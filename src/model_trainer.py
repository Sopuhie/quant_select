"""LightGBM 训练与模型落盘。"""
from __future__ import annotations

import gc
import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMRanker, early_stopping
from xgboost import XGBRanker
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .config import (
    BEST_LGB_PARAMS_JSON,
    CATBOOST_MODEL_PATH,
    FEATURE_COLUMNS,
    LGB_PARAMS,
    LGB_RANKER_DEFAULT_PARAMS,
    META_STACKER_PATH,
    MODEL_PATH,
    RANK_DRAWDOWN_WEIGHT_MULT,
    RANK_DRAWDOWN_WEIGHT_THRESH,
    RANK_META_OOF_FOLDS,
    RANK_SAMPLE_WEIGHT_EXTREME_RET,
    RANK_SAMPLE_WEIGHT_NEG_THRESH,
    XGB_MODEL_PATH,
    get_quant_config_merged,
)
from .data_fetcher import fetch_daily_hist, has_enough_history
from .database import fetch_latest_industry_by_codes, init_db, register_model_version
from .factor_calculator import (
    build_stock_panel_features,
    normalize_industry_label,
    prepare_ranking_cross_section_pipeline,
)
from .utils import permutation_importance_rank_ic_delta

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


def ranking_sample_weights_extreme_loss(
    sorted_df: pd.DataFrame,
    label_col: str,
) -> np.ndarray:
    """
    对标签收益率显著低于阈值的样本提高权重，使 LambdaRank 更重视「避雷」尾部。
    权重来自 ``config.json`` → ``quant.ranking``（缺省见 ``get_quant_config_merged``）。
    """
    y = pd.to_numeric(sorted_df[label_col], errors="coerce").to_numpy(dtype=np.float64)
    q = get_quant_config_merged().get("ranking", {})
    alpha = float(q.get("extreme_loss_weight", RANK_SAMPLE_WEIGHT_EXTREME_RET))
    thr = float(q.get("extreme_loss_thresh", RANK_SAMPLE_WEIGHT_NEG_THRESH))
    w = np.ones(len(y), dtype=np.float64)
    for i in range(len(y)):
        if np.isfinite(y[i]) and y[i] < thr:
            depth = float(thr - y[i])
            w[i] = 1.0 + alpha * min(depth / max(abs(thr), 1e-6), 5.0)
    dd_mult = float(q.get("drawdown_penalty_mult", RANK_DRAWDOWN_WEIGHT_MULT))
    dd_thr = float(q.get("drawdown_penalty_thresh", RANK_DRAWDOWN_WEIGHT_THRESH))
    if "factor_drawdown_60d" in sorted_df.columns:
        dd = pd.to_numeric(sorted_df["factor_drawdown_60d"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for i in range(len(y)):
            if np.isfinite(dd[i]) and dd[i] < dd_thr:
                w[i] *= dd_mult
    return w


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """PSI：衡量两组标量（如训练集 vs 验证集模型打分）分布漂移。"""
    e = np.asarray(expected, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    nb = max(3, int(n_bins))
    if len(e) < nb * 5 or len(a) < nb * 5:
        return float("nan")
    qs = np.unique(np.quantile(e, np.linspace(0.0, 1.0, nb + 1)))
    if len(qs) < 3:
        return float("nan")
    exp_c, _ = np.histogram(e, bins=qs)
    act_c, _ = np.histogram(a, bins=qs)
    exp_p = exp_c.astype(float) / max(float(exp_c.sum()), 1.0)
    act_p = act_c.astype(float) / max(float(act_c.sum()), 1.0)
    exp_p = np.where(exp_p <= 0, 1e-6, exp_p)
    act_p = np.where(act_p <= 0, 1e-6, act_p)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


def fit_meta_stacker_ridge(
    lgb_model: Any,
    xgb_model: Any,
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    y_tr: pd.Series,
    y_va: pd.Series,
    dates_va: np.ndarray,
) -> tuple[Ridge | None, dict[str, Any]]:
    """
    Stacking：以 LightGBM / XGBoost 一级得分为特征，Ridge 回归 ``label_ret``，
    预测侧再对元模型输出做截面分位秩以与旧融合尺度对齐。
    """
    try:
        z_tr = np.column_stack(
            [
                np.asarray(lgb_model.predict(X_tr), dtype=float).ravel(),
                np.asarray(xgb_model.predict(X_tr), dtype=float).ravel(),
            ]
        )
        z_va = np.column_stack(
            [
                np.asarray(lgb_model.predict(X_va), dtype=float).ravel(),
                np.asarray(xgb_model.predict(X_va), dtype=float).ravel(),
            ]
        )
    except Exception:
        return None, {}
    yt = pd.to_numeric(y_tr, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y_va, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(z_tr).all(axis=1) & np.isfinite(yt)
    if int(m.sum()) < 200:
        return None, {}
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(z_tr[m], yt[m])
    pred_va = ridge.predict(z_va)
    ic_meta = mean_cross_sectional_rank_ic(pred_va, dates_va, yv)
    return ridge, {"meta_mean_rank_ic_val": ic_meta, "meta_n_base": 2.0}


def _tail_date_split(
    df: pd.DataFrame,
    date_col: str,
    *,
    tail_frac: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按交易日尾部切出验证子集（用于 OOF 折内早停）。"""
    if df.empty:
        return df, df.iloc[:0].copy()
    uds = sorted(pd.unique(df[date_col].astype(str)))
    if len(uds) < 15:
        return df, df.iloc[:0].copy()
    cut_i = max(1, int(len(uds) * (1.0 - float(tail_frac))) - 1)
    cut = uds[min(cut_i, len(uds) - 1)]
    tr = df[df[date_col].astype(str) <= cut].copy()
    va = df[df[date_col].astype(str) > cut].copy()
    if tr.empty or va.empty:
        return df, df.iloc[:0].copy()
    return tr, va


def _catboost_thread_count_from_env() -> int | None:
    """与 OMP/CatBoost 环境变量对齐，减轻 Windows 上多库 OpenMP 超订导致的极慢或假死。"""
    raw = (
        os.environ.get("CATBOOST_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or ""
    ).strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def train_catboost_ranker_optional(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    best_params: dict[str, Any],
    feature_cols: list[str] | None = None,
    date_col: str = "date",
    label_col: str = "label_ret",
    n_estimators: int = 200,
    early_stopping_rounds: int = 40,
    random_state: int = 42,
    verbose: bool = False,
    oof_worker: bool = False,
) -> tuple[Any | None, dict[str, Any]]:
    """可选 CatBoost YetiRank；未安装库或失败时返回 (None, skip 信息)。

    - ``QUANT_CATBOOST_USE_GPU=1`` 时优先 ``task_type=GPU``（见 ``QUANT_CATBOOST_DEVICES``），
      失败则回退 CPU；``oof_worker=True`` 时强制 CPU（避免多进程 OOF 争用同一块 GPU）。
    - CPU 训练加速：``QUANT_CATBOOST_SPEEDOPT=1`` 时使用 ``Plain`` / ``rsm`` / 较低 ``border_count``。
    """
    cols = list(feature_cols or FEATURE_COLUMNS)
    try:
        from catboost import CatBoostRanker, Pool  # type: ignore[import-untyped]
    except Exception as exc:
        return None, {"catboost_skip": str(exc)}

    tr = prepare_ranking_frame(train_df, date_col)
    va = prepare_ranking_frame(val_df, date_col)
    X_tr = tr[cols].apply(pd.to_numeric, errors="coerce")
    y_tr = pd.to_numeric(tr[label_col], errors="coerce")
    X_va = va[cols].apply(pd.to_numeric, errors="coerce")
    y_va = pd.to_numeric(va[label_col], errors="coerce")
    # CatBoost C++ 按物理列序对齐，须与 FEATURE_COLUMNS 顺序完全一致
    X_tr = X_tr[cols]
    X_va = X_va[cols]
    mt = np.isfinite(X_tr.to_numpy()).all(axis=1) & np.isfinite(y_tr.to_numpy())
    mv = np.isfinite(X_va.to_numpy()).all(axis=1) & np.isfinite(y_va.to_numpy())
    tr = tr.loc[mt].reset_index(drop=True)
    va = va.loc[mv].reset_index(drop=True)
    X_tr = X_tr.loc[mt].reset_index(drop=True)
    y_tr = y_tr.loc[mt].reset_index(drop=True)
    X_va = X_va.loc[mv].reset_index(drop=True)
    y_va = y_va.loc[mv].reset_index(drop=True)
    if len(tr) < 300 or len(va) < 30:
        return None, {"catboost_skip": "too_few_rows"}

    y_tr_rel = relevance_labels_int_from_returns(tr, label_col, date_col)
    y_va_rel = relevance_labels_int_from_returns(va, label_col, date_col)
    y_fit_tr = np.ascontiguousarray(y_tr_rel, dtype=np.int32)
    y_fit_va = np.ascontiguousarray(y_va_rel, dtype=np.int32)
    _sw = ranking_sample_weights_extreme_loss(tr, label_col)
    _ = _sw
    g_tr = tr[date_col].astype(str).str[:10].tolist()
    g_va = va[date_col].astype(str).str[:10].tolist()

    bp = dict(best_params)
    lr = float(bp.get("learning_rate", 0.05))
    depth = int(np.clip(int(bp.get("num_leaves", 31)) // 8, 4, 8))
    # YetiRank 按日分组 + 全市场行数时，单轮迭代比 LGBM/XGB 的 histogram 排序慢很多；
    # Plain + 特征子采样 + 略降分箱数，通常明显加速（可用 QUANT_CATBOOST_SPEEDOPT=0 关闭）。
    _speed = (os.environ.get("QUANT_CATBOOST_SPEEDOPT", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    train_pool = Pool(
        data=X_tr,
        label=y_fit_tr,
        group_id=g_tr,
    )
    eval_pool = Pool(
        data=X_va,
        label=y_fit_va,
        group_id=g_va,
    )

    def _cat_metrics(model: Any) -> tuple[Any, dict[str, Any]]:
        pred_va = model.predict(X_va)
        ic_val = mean_cross_sectional_rank_ic(
            pred_va, va[date_col].to_numpy(), y_va.to_numpy(dtype=float)
        )
        metrics: dict[str, Any] = {
            "cat_mean_rank_ic_val": ic_val,
            "cat_n_train": int(len(y_tr)),
            "cat_n_val": int(len(y_va)),
        }
        return model, metrics

    try_gpu = (not oof_worker) and (
        os.environ.get("QUANT_CATBOOST_USE_GPU", "").strip().lower() in ("1", "true", "yes", "on")
    )
    if try_gpu:
        devices = os.environ.get("QUANT_CATBOOST_DEVICES", "0").strip() or "0"
        gpu_preprocess_threads = int(os.environ.get("QUANT_CATBOOST_GPU_THREAD_COUNT", "-1"))
        gpu_kw: dict[str, Any] = {
            "loss_function": "YetiRank",
            "iterations": int(n_estimators),
            "learning_rate": lr,
            "depth": depth,
            "random_seed": int(random_state),
            "verbose": bool(verbose),
            "early_stopping_rounds": int(early_stopping_rounds),
            "allow_writing_files": False,
            "task_type": "GPU",
            "devices": devices,
            "thread_count": gpu_preprocess_threads,
            "bootstrap_type": "Bernoulli",
            "subsample": float(os.environ.get("QUANT_CATBOOST_SUBSAMPLE", "0.66")),
        }
        try:
            gpu_model = CatBoostRanker(**gpu_kw)
            gpu_model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
            return _cat_metrics(gpu_model)
        except Exception as exc:
            warnings.warn(
                f"CatBoost GPU 不可用或训练失败，已回退 CPU: {exc}",
                UserWarning,
                stacklevel=2,
            )

    cb_kw: dict[str, Any] = {
        "loss_function": "YetiRank",
        "iterations": int(n_estimators),
        "learning_rate": lr,
        "depth": depth,
        "random_seed": int(random_state),
        "verbose": bool(verbose),
        "early_stopping_rounds": int(early_stopping_rounds),
        "allow_writing_files": False,
        "task_type": "CPU",
    }
    if _speed:
        cb_kw["boosting_type"] = "Plain"
        cb_kw["rsm"] = 0.85
        cb_kw["border_count"] = 128
    _tc = _catboost_thread_count_from_env()
    if _tc is not None:
        cb_kw["thread_count"] = _tc
    cpu_model = CatBoostRanker(**cb_kw)
    try:
        cpu_model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    except Exception as exc:
        return None, {"catboost_skip": str(exc)}
    return _cat_metrics(cpu_model)


def _resolve_oof_max_workers(n_tasks: int) -> int:
    """OOF 并行进程数上限（不超过折数）。``QUANT_OOF_MAX_WORKERS=1`` / ``seq`` / ``serial`` 强制串行。"""
    raw = (os.environ.get("QUANT_OOF_MAX_WORKERS", "") or "").strip().lower()
    if raw in ("0", "1", "seq", "serial", "none"):
        return 1
    k = max(1, int(n_tasks))
    cpu = os.cpu_count() or 8
    if not raw:
        return min(k, max(2, cpu // 4))
    try:
        n = int(raw)
    except ValueError:
        return 1
    return max(1, min(k, n))


def _walkforward_oof_fold_job(
    payload: tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        pd.DataFrame,
        tuple[str, ...],
        dict[str, Any],
        str,
        str,
        int,
        int,
        int,
        int,
        int,
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool] | None:
    """单折 OOF：子进程入口（须为模块级函数以便 pickle）。返回 ``(pos, plgb, pxgb, pcat, cat_ok)``。"""
    (
        tr_i,
        va_i,
        pos,
        X_te,
        cols_t,
        rank_params,
        date_col,
        label_col,
        n_lgb,
        n_xgb,
        _n_cat_oof_unused,
        es_oof,
        omp_threads,
    ) = payload
    cols = list(cols_t)
    if omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ["MKL_NUM_THREADS"] = str(omp_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
        os.environ["CATBOOST_NUM_THREADS"] = str(omp_threads)
    try:
        lgb_m, _ = train_lgbm_ranker(
            tr_i,
            va_i,
            params=rank_params,
            feature_cols=cols,
            date_col=date_col,
            label_col=label_col,
            n_estimators=n_lgb,
            early_stopping_rounds=es_oof,
            verbose=False,
        )
        xgb_m, _ = train_xgb_ranker(
            tr_i,
            va_i,
            best_params=rank_params,
            feature_cols=cols,
            date_col=date_col,
            label_col=label_col,
            n_estimators=n_xgb,
            early_stopping_rounds=es_oof,
            verbose=False,
        )
    except Exception:
        return None
    try:
        plgb = np.asarray(lgb_m.predict(X_te), dtype=float).ravel()
        pxgb = np.asarray(xgb_m.predict(X_te), dtype=float).ravel()
    except Exception:
        del lgb_m, xgb_m
        gc.collect()
        return None
    cat_ok = False
    pcat: np.ndarray | None = None
    # OOF 折内 CatBoost 仅作元特征趋势，固定极小迭代以缩短 Stacking 总耗时（与 n_cat 入参解耦）
    cb_m, _ = train_catboost_ranker_optional(
        tr_i,
        va_i,
        best_params=rank_params,
        feature_cols=cols,
        date_col=date_col,
        label_col=label_col,
        n_estimators=50,
        early_stopping_rounds=10,
        verbose=False,
        oof_worker=True,
    )
    if cb_m is not None:
        try:
            pcat = np.asarray(cb_m.predict(X_te), dtype=float).ravel()
            cat_ok = True
        except Exception:
            pcat = None
    del lgb_m, xgb_m, cb_m
    gc.collect()
    return (pos, plgb, pxgb, pcat, cat_ok)


def walkforward_oof_base_predictions(
    train_df: pd.DataFrame,
    *,
    date_col: str,
    cols: list[str],
    rank_params: dict[str, Any],
    n_folds: int,
    label_col: str = "label_ret",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    训练集内按交易日分块 Walk-forward，重训轻量 LGB / XGB / Cat 以构造 OOF 一级打分，
    供 Ridge 元学习器拟合，降低元特征与标签的同期泄漏。

    并行：默认 ``ProcessPoolExecutor``（``QUANT_OOF_MAX_WORKERS`` 未设时按 CPU 与折数自动取并发数；
    设为 ``1`` / ``seq`` / ``serial`` 则串行）。子进程内 CatBoost 强制 CPU，避免多进程争用 GPU。

    OOF 内树轮偏少（仅构元特征趋势）：``QUANT_OOF_LGB_ESTIMATORS`` / ``QUANT_OOF_XGB_ESTIMATORS`` /
    ``QUANT_OOF_CAT_ESTIMATORS``（默认 56/56/52），``QUANT_OOF_EARLY_STOPPING``（默认 24）。
    每进程 OpenMP 上限：``QUANT_OOF_WORKER_OMP_THREADS``；未设时为 ``max(1, cpu//workers//2)``。
    """
    work = prepare_ranking_frame(train_df, date_col).reset_index(drop=True)
    uds = sorted(pd.unique(work[date_col].astype(str)))
    if len(uds) < 30:
        n = len(work)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), None
    K = max(2, min(int(n_folds), max(2, len(uds) // 12)))
    blocks = np.array_split(np.array(uds, dtype=object), K)
    n_lgb_oof = int(os.environ.get("QUANT_OOF_LGB_ESTIMATORS", "56"))
    n_xgb_oof = int(os.environ.get("QUANT_OOF_XGB_ESTIMATORS", "56"))
    n_cat_oof = int(os.environ.get("QUANT_OOF_CAT_ESTIMATORS", "52"))
    es_oof = int(os.environ.get("QUANT_OOF_EARLY_STOPPING", "24"))

    cols_t = tuple(cols)
    raw_tasks: list[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            np.ndarray,
            pd.DataFrame,
            tuple[str, ...],
            dict[str, Any],
            str,
            str,
            int,
            int,
            int,
            int,
        ]
    ] = []
    for ki in range(K):
        te_dates = set(blocks[ki].tolist())
        tr_df = work.loc[~work[date_col].isin(te_dates)].copy()
        te_df = work.loc[work[date_col].isin(te_dates)].copy()
        if len(tr_df) < 800 or len(te_df) < 40:
            continue
        tr_i, va_i = _tail_date_split(tr_df, date_col, tail_frac=0.12)
        if len(tr_i) < 400 or len(va_i) < 30:
            tr_i = tr_df
            va_i = _tail_date_split(tr_df, date_col, tail_frac=0.25)[1]
        if len(va_i) < 25:
            continue
        m_te = work[date_col].isin(te_dates).to_numpy()
        w_te = work.loc[m_te].copy()
        pos = w_te.index.to_numpy()
        X_te = w_te[cols].apply(pd.to_numeric, errors="coerce")
        X_te = X_te[cols]
        raw_tasks.append(
            (
                tr_i,
                va_i,
                pos,
                X_te,
                cols_t,
                dict(rank_params),
                date_col,
                label_col,
                n_lgb_oof,
                n_xgb_oof,
                n_cat_oof,
                es_oof,
            )
        )

    oof_lgb = np.full(len(work), np.nan, dtype=float)
    oof_xgb = np.full(len(work), np.nan, dtype=float)
    oof_cat = np.full(len(work), np.nan, dtype=float)
    cat_any = False

    n_tasks = len(raw_tasks)
    if n_tasks == 0:
        for arr in (oof_lgb, oof_xgb):
            m = ~np.isfinite(arr)
            if m.any():
                arr[m] = (
                    float(np.nanmean(arr[np.isfinite(arr)]))
                    if np.isfinite(arr).any()
                    else 0.0
                )
        gc.collect()
        return oof_lgb, oof_xgb, None

    max_workers = min(_resolve_oof_max_workers(n_tasks), n_tasks)
    cpu = os.cpu_count() or 8
    wt = int(os.environ.get("QUANT_OOF_WORKER_OMP_THREADS", "0"))
    if wt <= 0:
        wt = max(1, cpu // max(max_workers, 1) // 2)
    tasks = [(*t, wt) for t in raw_tasks]

    def _merge_fold(r: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool]) -> None:
        nonlocal cat_any
        pos, plgb, pxgb, pcat, cok = r
        oof_lgb[pos] = plgb
        oof_xgb[pos] = pxgb
        if pcat is not None:
            oof_cat[pos] = pcat
            cat_any = cat_any or cok

    if max_workers <= 1:
        for t in tasks:
            try:
                r = _walkforward_oof_fold_job(t)
            except Exception:
                continue
            if r is None:
                continue
            _merge_fold(r)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_walkforward_oof_fold_job, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception:
                    continue
                if r is None:
                    continue
                _merge_fold(r)

    for arr in (oof_lgb, oof_xgb):
        m = ~np.isfinite(arr)
        if m.any():
            arr[m] = float(np.nanmean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
    if cat_any:
        m = ~np.isfinite(oof_cat)
        if m.any():
            oof_cat[m] = (
                float(np.nanmean(oof_cat[np.isfinite(oof_cat)]))
                if np.isfinite(oof_cat).any()
                else 0.0
            )
    else:
        oof_cat = None
    gc.collect()
    return oof_lgb, oof_xgb, oof_cat


def fit_meta_stacker_ridge_oof(
    lgb_model: Any,
    xgb_model: Any,
    cat_model: Any | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    date_col: str,
    cols: list[str],
    rank_params: dict[str, Any],
) -> tuple[Ridge | None, dict[str, Any]]:
    """
    以 Walk-forward OOF 一级打分为训练特征拟合 Ridge；验证集使用已训好的全量一级模型打分评估。
    """
    tr_sorted, X_tr_m, y_tr_m = _prepare_rank_xy(train_df, date_col=date_col, feature_cols=cols)
    va_r, X_va_m, y_va_m = _prepare_rank_xy(val_df, date_col=date_col, feature_cols=cols)
    qconf = get_quant_config_merged().get("ranking", {})
    kfolds = int(qconf.get("meta_oof_folds", RANK_META_OOF_FOLDS))
    if kfolds <= 1:
        return fit_meta_stacker_ridge(
            lgb_model,
            xgb_model,
            X_tr_m,
            X_va_m,
            y_tr_m,
            y_va_m,
            va_r["date"].to_numpy(),
        )
    oof_lgb, oof_xgb, oof_cat = walkforward_oof_base_predictions(
        train_df,
        date_col=date_col,
        cols=cols,
        rank_params=rank_params,
        n_folds=kfolds,
    )
    yt = pd.to_numeric(y_tr_m, errors="coerce").to_numpy(dtype=float)
    if oof_cat is not None and cat_model is not None:
        z_tr = np.column_stack([oof_lgb, oof_xgb, oof_cat])
        z_va = np.column_stack(
            [
                np.asarray(lgb_model.predict(X_va_m), dtype=float).ravel(),
                np.asarray(xgb_model.predict(X_va_m), dtype=float).ravel(),
                np.asarray(cat_model.predict(X_va_m), dtype=float).ravel(),
            ]
        )
        n_base = 3
    else:
        z_tr = np.column_stack([oof_lgb, oof_xgb])
        z_va = np.column_stack(
            [
                np.asarray(lgb_model.predict(X_va_m), dtype=float).ravel(),
                np.asarray(xgb_model.predict(X_va_m), dtype=float).ravel(),
            ]
        )
        n_base = 2
    m = np.isfinite(z_tr).all(axis=1) & np.isfinite(yt)
    if int(m.sum()) < 200:
        return None, {"meta_skip": "too_few_oof_rows"}
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(z_tr[m], yt[m])
    pred_va = ridge.predict(z_va)
    yv = pd.to_numeric(y_va_m, errors="coerce").to_numpy(dtype=float)
    ic_meta = mean_cross_sectional_rank_ic(
        pred_va, va_r["date"].to_numpy(), yv
    )
    return ridge, {"meta_mean_rank_ic_val": ic_meta, "meta_n_base": float(n_base)}


def save_meta_stacker(
    model: Ridge | None,
    path: Path | None = None,
    *,
    n_base: int = 2,
) -> Path | None:
    p = path or META_STACKER_PATH
    if model is None:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kind": "ridge_return",
            "model": model,
            "n_base": int(n_base),
        },
        p,
    )
    return p


def load_meta_stacker_optional(path: Path | None = None) -> dict[str, Any] | None:
    p = path or META_STACKER_PATH
    if not p.exists():
        return None
    try:
        raw = joblib.load(p)
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


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

    sw_tr = ranking_sample_weights_extreme_loss(tr, label_col)

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
        sample_weight=sw_tr,
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


def train_xgb_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    best_params: dict[str, Any],
    feature_cols: list[str] | None = None,
    date_col: str = "date",
    label_col: str = "label_ret",
    n_estimators: int = 200,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[XGBRanker, dict[str, Any]]:
    """
    与 ``train_lgbm_ranker`` 相同的按日分组与 relevance 标签，训练 ``rank:pairwise`` 的 XGBRanker。
    超参与 Optuna/LightGBM 结果对齐：树深度由 ``num_leaves`` 映射，学习率/子采样等直接沿用。
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
    y_fit_tr = np.ascontiguousarray(y_tr_rel, dtype=np.int32)
    y_fit_va = np.ascontiguousarray(y_va_rel, dtype=np.int32)

    bp = dict(best_params)
    lr = float(bp.get("learning_rate", 0.05))
    num_leaves = int(bp.get("num_leaves", 31))
    max_depth = int(np.clip(num_leaves // 4, 3, 8))
    subsample = float(bp.get("subsample", 0.8))
    colsample_bytree = float(bp.get("colsample_bytree", 0.8))

    model = XGBRanker(
        objective="rank:pairwise",
        eval_metric=["ndcg@3", "ndcg@5"],
        learning_rate=lr,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=1 if verbose else 0,
    )
    model.fit(
        X_tr,
        y_fit_tr,
        group=g_tr,
        eval_set=[(X_va, y_fit_va)],
        eval_group=[g_va],
        verbose=verbose,
    )

    pred_va = model.predict(X_va)
    ic_val = mean_cross_sectional_rank_ic(
        pred_va, va[date_col].to_numpy(), y_va.to_numpy(dtype=float)
    )
    metrics: dict[str, Any] = {
        "xgb_mean_rank_ic_val": ic_val,
        "xgb_n_train": int(len(y_tr)),
        "xgb_n_val": int(len(y_va)),
        "xgb_max_depth": max_depth,
    }
    try:
        er = getattr(model, "evals_result_", None) or {}
        v0 = er.get("validation_0", er.get("valid_0", {}))
        for key in sorted(v0.keys()):
            if "ndcg" in key.lower() and v0[key]:
                metrics[f"xgb_val_{key}_last"] = float(v0[key][-1])
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
    """序列化模型（joblib）：LightGBM / XGBoost Ranker 等。预测特征列须与 ``FEATURE_COLUMNS`` 一致。"""
    p = path or MODEL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    return p


def load_model(path: Path | None = None) -> Any:
    p = path or MODEL_PATH
    return joblib.load(p)


def load_xgb_ranker_optional(path: Path | None = None) -> Any | None:
    """若 ``models/xgb_model.pkl`` 不存在或损坏则返回 None（预测侧仅 LightGBM）。"""
    p = path or XGB_MODEL_PATH
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def load_catboost_ranker_optional(path: Path | None = None) -> Any | None:
    """若 ``models/cat_ranker.cbm`` 不存在或 CatBoost 未安装则返回 None。"""
    p = path or CATBOOST_MODEL_PATH
    if not p.exists():
        return None
    try:
        from catboost import CatBoostRanker  # type: ignore[import-untyped]

        m = CatBoostRanker()
        m.load_model(str(p))
        return m
    except Exception:
        return None


def estimate_ranker_n_features(model: Any) -> int | None:
    """推断排序模型训练时的特征列数（sklearn / LightGBM Booster / CatBoost）。"""
    n = getattr(model, "n_features_in_", None)
    if n is not None and int(n) > 0:
        return int(n)
    try:
        booster = model.booster_
        return int(booster.num_feature())
    except Exception:
        pass
    try:
        fn = getattr(model, "feature_names_", None)
        if fn is not None:
            return len(list(fn))
    except Exception:
        pass
    return None


def assert_feature_matrix_matches_rankers(
    X: pd.DataFrame,
    *,
    lgb_model: Any,
    xgb_model: Any | None = None,
    cat_model: Any | None = None,
    feature_cols: list[str] | None = None,
    context: str = "",
) -> None:
    """
    预测前校验：当前 ``FEATURE_COLUMNS`` 维数须与各已加载 Ranker 落盘时一致，否则给出明确重训指引。
    """
    cols = feature_cols or list(FEATURE_COLUMNS)
    if int(X.shape[1]) != len(cols):
        raise ValueError(
            f"特征矩阵列数 {X.shape[1]} 与 FEATURE_COLUMNS 长度 {len(cols)} 不一致。"
        )
    prefix = f"{context}: " if context else ""
    for label, m in (
        ("LightGBM", lgb_model),
        ("XGBoost", xgb_model),
        ("CatBoost", cat_model),
    ):
        if m is None:
            continue
        exp = estimate_ranker_n_features(m)
        if exp is None:
            continue
        if int(exp) != int(X.shape[1]):
            raise ValueError(
                f"{prefix}{label} 模型期望 {exp} 个特征，当前为 {X.shape[1]} 个（代码已扩展 FEATURE_COLUMNS，"
                "与旧模型不兼容）。请在项目根目录重新训练：python train_model.py"
            )


def _prepare_rank_xy(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """与 ``train_lgbm_ranker`` 相同的排序与有限值过滤，供 PSI / Stacking 复用。"""
    cols = list(feature_cols or FEATURE_COLUMNS)
    tr = prepare_ranking_frame(df, date_col)
    X = tr[cols].apply(pd.to_numeric, errors="coerce")
    X = X[cols]
    y = pd.to_numeric(tr["label_ret"], errors="coerce")
    m = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
    tr = tr.loc[m].reset_index(drop=True)
    X = X.loc[m].reset_index(drop=True)
    y = y.loc[m].reset_index(drop=True)
    return tr, X, y


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
    xgb_model, xgb_metrics = train_xgb_ranker(
        train_df,
        val_df,
        best_params=params,
        n_estimators=200,
        early_stopping_rounds=50,
        verbose=False,
    )
    save_model(xgb_model, XGB_MODEL_PATH)
    metrics = {**metrics, **xgb_metrics}

    cat_model, cat_metrics = train_catboost_ranker_optional(
        train_df,
        val_df,
        best_params=params,
        n_estimators=180,
        early_stopping_rounds=40,
        verbose=False,
    )
    metrics.update({k: v for k, v in cat_metrics.items() if v is not None})
    if cat_model is not None:
        try:
            cat_model.save_model(str(CATBOOST_MODEL_PATH))
            metrics["cat_model_path"] = str(CATBOOST_MODEL_PATH)
        except Exception as exc:
            metrics["cat_model_save_error"] = str(exc)

    tr_r, X_tr_m, y_tr_m = _prepare_rank_xy(train_df, date_col="date")
    va_r, X_va_m, y_va_m = _prepare_rank_xy(val_df, date_col="date")
    pred_tr_lgb = np.asarray(model.predict(X_tr_m), dtype=float).ravel()
    pred_va_lgb = np.asarray(model.predict(X_va_m), dtype=float).ravel()
    qconf = get_quant_config_merged()
    psi_cfg = qconf.get("psi", {})
    psi_bins = int(psi_cfg.get("bins", 12))
    psi_val = population_stability_index(
        pred_tr_lgb, pred_va_lgb, n_bins=psi_bins
    )
    metrics["psi_lgb_pred_train_vs_val"] = psi_val
    psi_thr = float(psi_cfg.get("alert_threshold", 0.25))
    if np.isfinite(psi_val) and psi_val > psi_thr:
        metrics["psi_alert"] = f"PSI={psi_val:.4f} 超过阈值 {psi_thr}"

    yv_arr = y_va_m.to_numpy(dtype=float)
    dt_va = va_r["date"].to_numpy()

    def _baseline_ic_lgb() -> float:
        p0 = np.asarray(model.predict(X_va_m), dtype=float).ravel()
        return float(mean_cross_sectional_rank_ic(p0, dt_va, yv_arr))

    try:
        pi_lgb = permutation_importance_rank_ic_delta(
            lambda xdf: np.asarray(model.predict(xdf), dtype=float).ravel(),
            X_va_m,
            yv_arr,
            dt_va,
            list(FEATURE_COLUMNS),
            baseline_ic_fn=_baseline_ic_lgb,
            n_repeats=1,
            seed=42,
        )
        metrics["perm_importance_lgb_mean_ic_drop"] = _json_safe(pi_lgb)
    except Exception as exc:
        metrics["perm_importance_lgb_error"] = str(exc)

    meta_model, meta_m = fit_meta_stacker_ridge_oof(
        model,
        xgb_model,
        cat_model,
        train_df,
        val_df,
        date_col="date",
        cols=list(FEATURE_COLUMNS),
        rank_params=params,
    )
    metrics.update(meta_m)
    n_base = int(float(meta_m.get("meta_n_base", 2))) if meta_m else 2
    save_meta_stacker(meta_model, n_base=n_base)

    register_model_version(
        version=version,
        train_end_date=train_end_date,
        features=list(FEATURE_COLUMNS),
        metrics=_json_safe({"objective": "lambdarank", **metrics}),
        set_active=True,
    )
    gc.collect()
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
    all_features = prepare_ranking_cross_section_pipeline(
        all_features, date_col="date"
    )
    return train_panel_and_register(
        all_features,
        train_end_date=train_end_date,
        version=version,
        model_path=model_path,
    )
