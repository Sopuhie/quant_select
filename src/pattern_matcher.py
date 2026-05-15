"""
图形相似度匹配引擎。
使用本地 SQLite 行情：皮尔逊 / RMSE 粗筛 + 可选 DTW；DTW 支持「收盘价+成交量」
双序列拼接后的多维 DTW（等价于每步代价为价、量平方误差加权和）。
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .config import get_quant_config_merged
from .database import get_connection


def get_normalized_series(series: pd.Series) -> np.ndarray:
    """将序列 Min-Max 标准化到 [0, 1] 区间，消除价格绝对尺度差异。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.array([])
    min_val = float(s.min())
    max_val = float(s.max())
    if abs(max_val - min_val) < 1e-8:
        return np.zeros(len(s))
    return ((s.values - min_val) / (max_val - min_val)).astype(float)


def _row_minmax_norm(X: np.ndarray) -> np.ndarray:
    """对二维矩阵按行 Min-Max 到 [0,1]。"""
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    rng = maxs - mins
    out = np.zeros_like(X, dtype=np.float64)
    np.divide(X - mins, rng, out=out, where=rng >= 1e-8)
    return out


DEFAULT_DTW_SAKOE_CHIBA_RADIUS = 12


def _dtw_accumulated_sq_cost_full_2row(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        ai = a[i - 1]
        delta_sq = (ai - b) ** 2
        for j in range(1, m + 1):
            curr[j] = delta_sq[j - 1] + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    val = prev[m]
    return float(val) if np.isfinite(val) else float("inf")


def _dtw_accumulated_sq_cost_banded_2row(
    a: np.ndarray,
    b: np.ndarray,
    radius: int,
) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    r = max(0, int(radius))
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[:] = np.inf
        lo = max(1, i - r)
        hi = min(m, i + r)
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            c = (ai - b[j - 1]) ** 2
            curr[j] = c + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    val = prev[m]
    return float(val) if np.isfinite(val) else float("inf")


def _dtw_euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sakoe_chiba_radius: int | None,
) -> float:
    rad = (
        int(sakoe_chiba_radius)
        if sakoe_chiba_radius is not None
        else DEFAULT_DTW_SAKOE_CHIBA_RADIUS
    )
    cost_sq = _dtw_accumulated_sq_cost_banded_2row(a, b, rad)
    if not np.isfinite(cost_sq):
        cost_sq = _dtw_accumulated_sq_cost_full_2row(a, b)
    return float(np.sqrt(cost_sq)) if np.isfinite(cost_sq) else float("inf")


def _dtw_distances_ref_to_rows(
    ref: np.ndarray,
    X: np.ndarray,
    *,
    sakoe_chiba_radius: int | None,
) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = _dtw_euclidean_distance(ref, X[i], sakoe_chiba_radius=sakoe_chiba_radius)
    return out


def _pearson_rows(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    ref_mean = ref.mean()
    X_mean = X.mean(axis=1, keepdims=True)
    rc = ref - ref_mean
    Xc = X - X_mean
    num = (Xc * rc).sum(axis=1)
    den_ref = np.sqrt((rc**2).sum())
    den_x = np.sqrt((Xc**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = num / (den_ref * den_x)
    bad = ~np.isfinite(corr) | (den_ref < 1e-15) | (den_x < 1e-15)
    corr = np.where(bad, -1.0, corr)
    return corr.astype(np.float64)


def _rmse_rows(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = X - ref
    return np.sqrt(np.mean(diff**2, axis=1)).astype(np.float64)


def find_similar_patterns(
    target_code: str,
    start_date: str,
    end_date: str,
    *,
    compare_days: int = 120,
    limit_results: int = 5,
    algorithm: str = "pearson",
    dtw_sakoe_chiba_radius: int | None = None,
) -> list[dict]:
    """
    在本地行情库中，寻找与目标股票给定区间形态最相似的个股。

    - **粗筛**：皮尔逊（收盘）+ 皮尔逊（成交量）+ RMSE（收盘）综合排序，仅保留 Top K（见 ``config`` quant.pattern）。
    - **DTW**：在 Top K 上，对 ``[归一化收盘 || w·归一化成交量]`` 拼接序列做 DTW（量价协同）。
    """
    qpat = get_quant_config_merged().get("pattern", {})
    coarse_top_k = max(20, int(qpat.get("coarse_top_k", 100)))
    vol_w = float(qpat.get("dtw_volume_weight", 1.0))

    algo = str(algorithm).strip().lower()
    if algo not in ("pearson", "dtw"):
        algo = "pearson"

    target_code = str(target_code).strip().zfill(6)
    start_date = str(start_date).strip()[:10]
    end_date = str(end_date).strip()[:10]

    sql_target = """
        SELECT date, close, volume FROM stock_daily_kline
        WHERE stock_code = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
    """
    with get_connection() as conn:
        target_df = pd.read_sql_query(
            sql_target, conn, params=(target_code, start_date, end_date)
        )

        if target_df.empty or len(target_df) < 5:
            return []

        window_size = len(target_df)
        target_series = get_normalized_series(target_df["close"])
        target_vol = get_normalized_series(target_df["volume"])
        if len(target_series) != window_size or len(target_vol) != window_size:
            return []

        end_dt = pd.Timestamp(end_date)
        calendar_slack = max(int(compare_days), window_size * 3, 60)
        cutoff = (end_dt - timedelta(days=calendar_slack)).strftime("%Y-%m-%d")

        sql_all = """
            SELECT date, stock_code, stock_name, close, volume FROM stock_daily_kline
            WHERE stock_code != ? AND date >= ?
            ORDER BY stock_code, date ASC
        """
        all_df = pd.read_sql_query(sql_all, conn, params=(target_code, cutoff))

    if all_df.empty:
        return []

    all_df["close"] = pd.to_numeric(all_df["close"], errors="coerce")
    all_df["volume"] = pd.to_numeric(all_df["volume"], errors="coerce")
    all_df = all_df.dropna(subset=["close", "volume"])

    stock_codes: list[str] = []
    stock_names: list[str] = []
    raw_blocks: list[np.ndarray] = []
    raw_vol_blocks: list[np.ndarray] = []
    dates_blocks: list[list[str]] = []
    raw_price_blocks: list[list[float]] = []

    grouped = all_df.groupby("stock_code", sort=False)
    for code, group in grouped:
        code_z = str(code).strip().zfill(6)
        if code_z == target_code:
            continue
        if len(group) < window_size:
            continue

        candidate_window = group.tail(window_size)
        closes = candidate_window["close"].to_numpy(dtype=np.float64, copy=False)
        vols = candidate_window["volume"].to_numpy(dtype=np.float64, copy=False)
        if closes.shape[0] != window_size or not np.all(np.isfinite(closes)):
            continue
        if vols.shape[0] != window_size or not np.all(np.isfinite(vols)) or np.any(vols <= 0):
            continue

        candidate_name = str(candidate_window["stock_name"].iloc[0] or "").strip()
        dates_fmt = (
            pd.to_datetime(candidate_window["date"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        if len(dates_fmt) != window_size:
            continue

        stock_codes.append(code_z)
        stock_names.append(candidate_name)
        raw_blocks.append(closes)
        raw_vol_blocks.append(vols)
        dates_blocks.append(dates_fmt)
        raw_price_blocks.append(closes.astype(float).tolist())

    if not raw_blocks:
        return []

    X_raw = np.vstack(raw_blocks)
    X_vol_raw = np.vstack(raw_vol_blocks)
    X_norm = _row_minmax_norm(X_raw)
    X_vol_norm = _row_minmax_norm(X_vol_raw)
    ref = np.asarray(target_series, dtype=np.float64)
    ref_v = np.asarray(target_vol, dtype=np.float64)

    corr_c = _pearson_rows(ref, X_norm)
    corr_v = _pearson_rows(ref_v, X_vol_norm)
    rmse_c = _rmse_rows(ref, X_norm)
    rmse_score = 1.0 / (1.0 + rmse_c)
    coarse = 0.45 * corr_c + 0.25 * corr_v + 0.30 * rmse_score

    limit = int(limit_results)
    k_coarse = min(coarse_top_k, len(coarse))

    if algo == "dtw":
        top_idx = np.argsort(-coarse)[:k_coarse]
        ref_cat = np.concatenate([ref, vol_w * ref_v])
        X_sub = np.concatenate(
            [X_norm[top_idx], vol_w * X_vol_norm[top_idx]], axis=1
        )
        dtw_d = _dtw_distances_ref_to_rows(
            ref_cat,
            X_sub,
            sakoe_chiba_radius=dtw_sakoe_chiba_radius,
        )
        similarity_sub = np.where(
            np.isfinite(dtw_d) & (dtw_d >= 0),
            100.0 / (1.0 + dtw_d),
            0.0,
        )
        order_sub = np.argsort(-similarity_sub)[:limit]
        order = top_idx[order_sub.astype(np.int64)]
        similarity = np.full(len(stock_codes), -1.0, dtype=np.float64)
        similarity[top_idx] = similarity_sub
    else:
        similarity = np.maximum(0.0, (coarse + 1.0) / 2.0 * 85.0 + rmse_score * 15.0)
        order = np.argsort(-similarity)[:limit]

    tgt_list = ref.tolist()
    tgt_vol_list = ref_v.tolist()
    results: list[dict] = []
    for k in order:
        i = int(k)
        row_norm = X_norm[i]
        row_v_norm = X_vol_norm[i]
        results.append(
            {
                "stock_code": stock_codes[i],
                "stock_name": stock_names[i],
                "similarity": float(similarity[i]),
                "target_trajectory": tgt_list,
                "target_volume_trajectory": tgt_vol_list,
                "candidate_trajectory": row_norm.tolist(),
                "candidate_volume_trajectory": row_v_norm.tolist(),
                "dates": dates_blocks[i],
                "raw_prices": raw_price_blocks[i],
            }
        )

    return results
