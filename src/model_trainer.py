"""LightGBM 训练与模型落盘。"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMRanker, early_stopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .config import (
    BEST_LGB_PARAMS_JSON,
    FEATURE_COLUMNS,
    LGB_PARAMS,
    LGB_RANKER_DEFAULT_PARAMS,
    MODEL_PATH,
)
from .data_fetcher import fetch_daily_hist, has_enough_history
from .database import fetch_latest_industry_by_codes, init_db, register_model_version
from .factor_calculator import (
    build_stock_panel_features,
    clean_cross_sectional_features,
    normalize_industry_label,
)

# LambdaRank 默认 label_gain 长度 31，relevance 合法索引仅为 0..30（见 LightGBM 文档）
LAMBDARANK_MAX_REL_INDEX = 30


def calculate_ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
    """Spearman 秩相关，衡量因子与未来收益截面单调性。"""
    m = factor_values.notna() & future_returns.notna()
    if int(m.sum()) < 30:
        return float("nan")
    return float(factor_values[m].corr(future_returns[m], method="spearman"))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def calculate_ir(ic_series: pd.Series) -> float:
    """IR = IC 序列均值 / 标准差。"""
    s = ic_series.dropna()
    if len(s) < 2 or float(s.std()) < 1e-12:
        return float("nan")
    return float(s.mean() / s.std())


def prepare_ranking_frame(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """按交易日升序排列；同日按 stock_code 稳定次序，满足 LightGBM group 连续性。"""
    if df.empty:
        return df
    keys = [date_col]
    if "stock_code" in df.columns:
        keys.append("stock_code")
    return df.sort_values(keys, ascending=True).reset_index(drop=True)


def ranking_group_sizes(sorted_df: pd.DataFrame, date_col: str = "date") -> np.ndarray:
    """与 ``prepare_ranking_frame`` 输出对齐的每组 query 规模（每个交易日一行组）。"""
    return sorted_df.groupby(date_col, sort=False).size().to_numpy(dtype=np.int64)


def relevance_labels_int_from_returns(
    sorted_df: pd.DataFrame,
    label_col: str,
    date_col: str,
    *,
    max_rel_index: int = LAMBDARANK_MAX_REL_INDEX,
) -> np.ndarray:
    """
    LightGBM ``lambdarank``：label 为 **整型 relevance**，且必须落在默认 ``label_gain`` 长度内（0..30）。

    在每个交易日内先按 ``label_ret`` 得到次序秩，再 **线性映射** 到 ``0 .. max_rel_index``，
    保持收益越高 relevance 越大（截面单调性不变）。
    """
    df = sorted_df.reset_index(drop=True)
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=np.float64)
    dates = df[date_col].astype(str).to_numpy()
    n = len(df)
    cap = max(int(max_rel_index), 1)
    out = np.zeros(n, dtype=np.int32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        chunk = y[i:j]
        cnt = j - i
        if cnt <= 1:
            out[i:j] = 0
        else:
            rk = pd.Series(chunk).rank(method="first", ascending=True).to_numpy(dtype=np.float64) - 1.0
            scaled = np.floor(rk * float(cap) / float(cnt - 1)).astype(np.int32)
            np.clip(scaled, 0, cap, out=scaled)
            out[i:j] = scaled
        i = j
    return out


def mean_cross_sectional_rank_ic(
    pred: np.ndarray,
    dates: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    *,
    min_names: int = 10,
) -> float:
    """验证集上逐日截面 Spearman(pred, label) 的均值（Rank IC）。"""
    dd = pd.Series(dates).astype(str).str[:10]
    frame = pd.DataFrame({"d": dd, "p": np.asarray(pred).ravel(), "y": np.asarray(labels).ravel()})
    ics: list[float] = []
    for _, sub in frame.groupby("d", sort=False):
        if len(sub) < min_names:
            continue
        ic = sub["p"].corr(sub["y"], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def merge_ranker_params(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """以 ``LGB_RANKER_DEFAULT_PARAMS`` 为底合并 Optuna / JSON 超参。

    ``eval_at`` / ``n_estimators`` 不得进入字典：训练函数不对 Ranker 传 ``eval_at``（避免 sklearn 告警），
    ``n_estimators`` 由 ``train_lgbm_ranker(..., n_estimators=...)`` 单独指定。
    """
    skip = frozenset({"eval_at", "n_estimators"})
    out = dict(LGB_RANKER_DEFAULT_PARAMS)
    if overrides:
        for k, v in overrides.items():
            if v is None or k in skip:
                continue
            out[k] = v
    return out


def save_ranker_params_json(params: dict[str, Any], path: Path | None = None) -> Path:
    """保存 LambdaRank 超参 JSON（建议勿含 eval_at / n_estimators）。"""
    p = path or BEST_LGB_PARAMS_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(params)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return p


def load_ranker_params_json(path: Path | None = None) -> dict[str, Any] | None:
    """读取 JSON；缺失或损坏则返回 None。"""
    p = path or BEST_LGB_PARAMS_JSON
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def train_lgbm_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    params: dict[str, Any],
    feature_cols: list[str] | None = None,
    date_col: str = "date",
    label_col: str = "label_ret",
    n_estimators: int = 800,
    early_stopping_rounds: int = 50,
    verbose: bool = False,
) -> tuple[LGBMRanker, dict[str, Any]]:
    """
    LambdaRank：样本必须按 ``date`` 升序且 ``group`` 与行顺序一致。
    """
    cols = feature_cols or list(FEATURE_COLUMNS)
    tr = prepare_ranking_frame(train_df, date_col)
    va = prepare_ranking_frame(val_df, date_col)

    X_tr = tr[cols].apply(pd.to_numeric, errors="coerce")
    y_tr = pd.to_numeric(tr[label_col], errors="coerce")
    X_va = va[cols].apply(pd.to_numeric, errors="coerce")
    y_va = pd.to_numeric(va[label_col], errors="coerce")

    mt = np.isfinite(X_tr.to_numpy()).all(axis=1) & np.isfinite(y_tr.to_numpy())
    mv = np.isfinite(X_va.to_numpy()).all(axis=1) & np.isfinite(y_va.to_numpy())

    tr = tr.loc[mt].reset_index(drop=True)
    va = va.loc[mv].reset_index(drop=True)
    X_tr = X_tr.loc[mt].reset_index(drop=True)
    y_tr = y_tr.loc[mt].reset_index(drop=True)
    X_va = X_va.loc[mv].reset_index(drop=True)
    y_va = y_va.loc[mv].reset_index(drop=True)

    g_tr = ranking_group_sizes(tr, date_col)
    g_va = ranking_group_sizes(va, date_col)

    y_tr_rel = relevance_labels_int_from_returns(tr, label_col, date_col)
    y_va_rel = relevance_labels_int_from_returns(va, label_col, date_col)
    # sklearn / LightGBM C API 要求 ranking label 为整数数组，禁止 float64 视图
    y_fit_tr = np.ascontiguousarray(y_tr_rel, dtype=np.int32)
    y_fit_va = np.ascontiguousarray(y_va_rel, dtype=np.int32)

    # eval_at 只出现一次：写入展开参数，勿再传 ``eval_at=`` 关键字（否则会与内部 params 重复告警）
    ranker_params = {k: v for k, v in dict(params).items() if v is not None}
    ranker_params.pop("n_estimators", None)
    ranker_params.pop("eval_at", None)
    # 不传 eval_at：LightGBM sklearn 会把构造函数里的 eval_at 再写入 Booster params，触发重复告警；
    # 使用库内置 NDCG 截断（默认含多个 @k）即可早停与记录验证指标。

    model = LGBMRanker(**ranker_params, n_estimators=n_estimators)
    model.fit(
        X_tr,
        y_fit_tr,
        group=g_tr,
        eval_set=[(X_va, y_fit_va)],
        eval_group=[g_va],
        callbacks=[early_stopping(early_stopping_rounds, verbose=verbose)],
    )

    pred_va = model.predict(X_va)
    ic_val = mean_cross_sectional_rank_ic(
        pred_va, va[date_col].to_numpy(), y_va.to_numpy(dtype=float)
    )
    metrics: dict[str, Any] = {
        "mean_rank_ic_val": ic_val,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
    }
    try:
        er = getattr(model, "evals_result_", None) or {}
        v0 = er.get("valid_0", {})
        for key in sorted(v0.keys()):
            if key.startswith("ndcg") and v0[key]:
                metrics[f"val_{key}_last"] = float(v0[key][-1])
    except Exception:
        pass

    return model, metrics


def optuna_tune_lgbm_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_trials: int = 20,
    date_col: str = "date",
    label_col: str = "label_ret",
    feature_cols: list[str] | None = None,
    n_estimators_trial: int = 400,
    early_stopping_rounds: int = 50,
    verbose: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Optuna 最大化验证集平均 Rank IC；返回合并后的完整 LambdaRank 参数字典。"""
    import optuna

    cols = feature_cols or list(FEATURE_COLUMNS)

    def objective(trial: optuna.Trial) -> float:
        p = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "verbosity": -1,
            "seed": seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "bagging_freq": 1,
        }
        try:
            _, metrics = train_lgbm_ranker(
                train_df,
                val_df,
                params=p,
                feature_cols=cols,
                date_col=date_col,
                label_col=label_col,
                n_estimators=n_estimators_trial,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        except Exception:
            return -1e9
        ic = metrics.get("mean_rank_ic_val")
        if ic is None or not math.isfinite(ic):
            return -1e9
        return float(ic)

    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=verbose,
    )
    merged = merge_ranker_params(study.best_params)
    return merged


def collect_training_samples(
    stock_pairs: list[tuple[str, str]],
    train_end_date: str,
    start_date: str = "20180101",
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """
    多只股票纵向合并训练样本；只保留 date <= train_end_date 的行。

    progress=True 时每只股票拉取前后打印一行，避免长时间无输出误以为卡死。
    """
    parts: list[pd.DataFrame] = []
    end_compact = train_end_date.replace("-", "")
    init_db()
    pair_codes = [str(c).strip().zfill(6) for c, _ in stock_pairs]
    industry_by_code = fetch_latest_industry_by_codes(pair_codes)
    total = len(stock_pairs)
    if progress and total:
        print(
            "[数据采集] 共 "
            f"{total} 只股票；每条请求含超时与有限次重试，首只可能较慢请稍候…",
            flush=True,
        )
    for idx, (code, name) in enumerate(stock_pairs, start=1):
        if progress:
            print(f"[数据采集] ({idx}/{total}) {code} {name[:16]} 拉取日线…", flush=True)
        hist = fetch_daily_hist(code, start_date=start_date, end_date=end_compact)
        if not has_enough_history(hist):
            if progress:
                print(f"          → 跳过（K 线不足或无数据），累计有效 {len(parts)} 只", flush=True)
            continue
        panel = build_stock_panel_features(hist, code, name)
        code6 = str(code).strip().zfill(6)
        panel["industry"] = normalize_industry_label(industry_by_code.get(code6))
        panel = panel[panel["date"] <= train_end_date]
        if len(panel) > 0:
            parts.append(panel)
            if progress:
                print(
                    f"          → 纳入训练 样本行 {len(panel)}，累计有效 {len(parts)} 只",
                    flush=True,
                )
        elif progress:
            print(f"          → 跳过（截至日无样本），累计有效 {len(parts)} 只", flush=True)
    if progress and total:
        print(f"[数据采集] 完成：有效股票 {len(parts)}/{total}，正在合并面板…", flush=True)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def train_lgbm_regressor(
    df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[LGBMRegressor, dict[str, Any]]:
    """
    使用带列名的 DataFrame 训练，保证底层 Booster 记录 feature_name，
    落盘后的 pkl 可在 Streamlit 中通过 booster.feature_importance() / feature_name_ 读取重要性。
    """
    work = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["label_ret"]
    )
    X_df = work[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y_s = pd.to_numeric(work["label_ret"], errors="coerce")
    x_vals = X_df.to_numpy(dtype=np.float64)
    y_vals = y_s.to_numpy(dtype=np.float64)
    finite = np.isfinite(x_vals).all(axis=1) & np.isfinite(y_vals)
    X_df = X_df.loc[finite].reset_index(drop=True)
    y = y_vals[finite]
    if len(y) < 100:
        raise RuntimeError(
            f"有效训练样本过少（{len(y)}），多为异常价或缺失导致；请换截止日期或增大股票池。"
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X_df,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    model = LGBMRegressor(**LGB_PARAMS, n_estimators=500)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )
    pred_val = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    ic_by_feature: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        ic_by_feature[col] = calculate_ic(X_val[col], pd.Series(y_val))
    ic_series = pd.Series(ic_by_feature)
    ir_val = calculate_ir(ic_series)
    metrics = {
        "rmse_val": rmse,
        "n_samples": int(len(y)),
        "ic_mean": float(ic_series.mean()) if ic_series.notna().any() else float("nan"),
        "ir_val": ir_val,
        "ic_by_feature": ic_by_feature,
    }
    return model, metrics


def feature_importance_table(
    model: Any,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    从已拟合的 LGBMRegressor 读取特征重要性，供 UI 展示。
    名称以 Booster.feature_name() 为准；若为 Column_* 且长度与 FEATURE_COLUMNS 一致则回退为配置列名。
    """
    booster = model.booster_
    names = list(booster.feature_name())
    imp = booster.feature_importance(importance_type=importance_type)
    if (
        names
        and all(str(n).startswith("Column_") for n in names)
        and len(names) == len(FEATURE_COLUMNS)
    ):
        names = list(FEATURE_COLUMNS)
    out = pd.DataFrame({"feature": names, "importance": imp.astype(float)})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model: Any, path: Path | None = None) -> Path:
    """序列化 LightGBM 模型（回归或 LambdaRank）。训练时请传入列名为 FEATURE_COLUMNS 的 DataFrame。"""
    p = path or MODEL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    return p


def load_model(path: Path | None = None) -> Any:
    p = path or MODEL_PATH
    return joblib.load(p)


def train_panel_and_register(
    panel_df: pd.DataFrame,
    train_end_date: str,
    version: str | None = None,
    model_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """对已合并、并完成截面清洗的训练面板执行 LambdaRank 训练、落盘与注册。"""
    if version is None:
        version = "v" + datetime.now().strftime("%Y%m%d.%H%M")
    if panel_df.empty:
        raise RuntimeError("训练样本为空，请检查网络、股票池与日期区间。")
    work = panel_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["label_ret", "date"]
    )
    dates = sorted(work["date"].astype(str).unique())
    if len(dates) < 30:
        train_df, val_df = work.copy(), work.copy()
    else:
        split_date = dates[-max(1, len(dates) // 10)]
        train_df = work[work["date"].astype(str) < split_date].copy()
        val_df = work[work["date"].astype(str) >= split_date].copy()
    if train_df.empty or val_df.empty:
        train_df, val_df = work.copy(), work.copy()
    params = merge_ranker_params(load_ranker_params_json())
    model, metrics = train_lgbm_ranker(
        train_df,
        val_df,
        params=params,
        n_estimators=500,
        early_stopping_rounds=50,
        verbose=False,
    )
    save_model(model, model_path)
    register_model_version(
        version=version,
        train_end_date=train_end_date,
        features=list(FEATURE_COLUMNS),
        metrics=_json_safe({"objective": "lambdarank", **metrics}),
        set_active=True,
    )
    return model, {"version": version, **metrics, "rows": len(panel_df)}


def train_and_register(
    stock_pairs: list[tuple[str, str]],
    train_end_date: str,
    version: str | None = None,
    model_path: Path | None = None,
    *,
    collect_progress: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """collect → 截面清洗 → train_panel_and_register（供脚本一键调用）。"""
    all_features = collect_training_samples(
        stock_pairs,
        train_end_date=train_end_date,
        progress=collect_progress,
    )
    if all_features.empty:
        raise RuntimeError("训练样本为空，请检查网络、股票池与日期区间。")
    all_features = clean_cross_sectional_features(all_features)
    return train_panel_and_register(
        all_features,
        train_end_date=train_end_date,
        version=version,
        model_path=model_path,
    )
