"""LightGBM 训练与模型落盘。"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, LGB_PARAMS, MODEL_PATH
from .data_fetcher import fetch_daily_hist, has_enough_history
from .database import fetch_latest_industry_by_codes, init_db, register_model_version
from .factor_calculator import (
    build_stock_panel_features,
    clean_cross_sectional_features,
    normalize_industry_label,
)


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
    model: LGBMRegressor,
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


def save_model(model: LGBMRegressor, path: Path | None = None) -> Path:
    """序列化 LGBMRegressor（含 Booster 与特征名）。训练时请传入列名为 FEATURE_COLUMNS 的 DataFrame。"""
    p = path or MODEL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    return p


def load_model(path: Path | None = None) -> LGBMRegressor:
    p = path or MODEL_PATH
    return joblib.load(p)


def train_panel_and_register(
    panel_df: pd.DataFrame,
    train_end_date: str,
    version: str | None = None,
    model_path: Path | None = None,
) -> tuple[LGBMRegressor, dict[str, Any]]:
    """对已合并、并完成截面清洗的训练面板执行训练、落盘与注册。"""
    if version is None:
        version = "v" + datetime.now().strftime("%Y%m%d.%H%M")
    if panel_df.empty:
        raise RuntimeError("训练样本为空，请检查网络、股票池与日期区间。")
    model, metrics = train_lgbm_regressor(panel_df)
    save_model(model, model_path)
    register_model_version(
        version=version,
        train_end_date=train_end_date,
        features=list(FEATURE_COLUMNS),
        metrics=_json_safe(metrics),
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
) -> tuple[LGBMRegressor, dict[str, Any]]:
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
