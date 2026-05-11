"""
图形相似度匹配引擎。
使用本地 SQLite 行情：可选皮尔逊相关 + 形状距离（向量化），或纯 NumPy DTW（双行 DP，可选 Sakoe-Chiba 带状）。
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

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
    """对二维收盘价矩阵按行 Min-Max 到 [0,1]，与 ``get_normalized_series`` 规则一致。"""
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    rng = maxs - mins
    out = np.zeros_like(X, dtype=np.float64)
    np.divide(X - mins, rng, out=out, where=rng >= 1e-8)
    return out


# Sakoe-Chiba 带宽（|i−j|≤R）；R 为常数时单条序列约 O(W·R)，全候选合计 O(N·W·R)≈O(N·W)。
DEFAULT_DTW_SAKOE_CHIBA_RADIUS = 12


def _dtw_accumulated_sq_cost_full_2row(a: np.ndarray, b: np.ndarray) -> float:
    """
    完整 DTW：累计平方欧式路径代价（未开方）。
    双行滚动数组，时间 O(n·m)，空间 O(m)；无 (n+1)×(m+1) 全矩阵分配。
    """
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
    """
    Sakoe-Chiba 带状 DTW（|i−j|≤radius），双行滚动。
    固定 radius 且 n≈m=W 时，时间约 O(W·radius)。
    """
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
    """欧式 DTW 距离；优先带状加速，累计代价非有限时回退完整 DTW。"""
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
    """样板 ``ref`` (W,) 与候选矩阵 ``X`` (N,W) 的 DTW 距离向量 (N,)。"""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = _dtw_euclidean_distance(ref, X[i], sakoe_chiba_radius=sakoe_chiba_radius)
    return out


def _pearson_rows(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``ref`` 形状 (W,)，``X`` (N, W)；返回每只候选与 ``ref`` 的皮尔逊相关系数 (N,)。"""
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
    在本地行情库中，寻找与目标股票给定区间收盘价形态最相似的个股（最近窗口长度与模板一致）。

    Args:
        target_code: 模板股票代码
        start_date / end_date: 模板区间（YYYY-MM-DD）
        compare_days: 日历跨度近似控制候选数据加载起点（越大越耗内存，默认覆盖约一季以上交易日）
        limit_results: 返回前几条
        algorithm: ``pearson`` | ``dtw``；DTW 适合时间轴伸缩的形态比较。
        dtw_sakoe_chiba_radius: DTW 的 Sakoe-Chiba 带宽 R（``|i-j|≤R``）；``None`` 时用模块默认值。
            固定 R 时复杂度约 ``O(N·W·R)``；过紧时会自动回退完整双行 DTW。
    """
    algo = str(algorithm).strip().lower()
    if algo not in ("pearson", "dtw"):
        algo = "pearson"

    target_code = str(target_code).strip().zfill(6)
    start_date = str(start_date).strip()[:10]
    end_date = str(end_date).strip()[:10]

    sql_target = """
        SELECT date, close FROM stock_daily_kline
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
        if len(target_series) != window_size:
            return []

        end_dt = pd.Timestamp(end_date)
        calendar_slack = max(int(compare_days), window_size * 3, 60)
        cutoff = (end_dt - timedelta(days=calendar_slack)).strftime("%Y-%m-%d")

        sql_all = """
            SELECT date, stock_code, stock_name, close FROM stock_daily_kline
            WHERE stock_code != ? AND date >= ?
            ORDER BY stock_code, date ASC
        """
        all_df = pd.read_sql_query(sql_all, conn, params=(target_code, cutoff))

    if all_df.empty:
        return []

    all_df["close"] = pd.to_numeric(all_df["close"], errors="coerce")
    all_df = all_df.dropna(subset=["close"])

    stock_codes: list[str] = []
    stock_names: list[str] = []
    raw_blocks: list[np.ndarray] = []
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
        if closes.shape[0] != window_size or not np.all(np.isfinite(closes)):
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
        dates_blocks.append(dates_fmt)
        raw_price_blocks.append(closes.astype(float).tolist())

    if not raw_blocks:
        return []

    X_raw = np.vstack(raw_blocks)
    X_norm = _row_minmax_norm(X_raw)
    ref = np.asarray(target_series, dtype=np.float64)

    if algo == "dtw":
        dtw_d = _dtw_distances_ref_to_rows(
            ref,
            X_norm,
            sakoe_chiba_radius=dtw_sakoe_chiba_radius,
        )
        similarity = np.where(
            np.isfinite(dtw_d) & (dtw_d >= 0),
            100.0 / (1.0 + dtw_d),
            0.0,
        )
    else:
        corr = _pearson_rows(ref, X_norm)
        diff = X_norm - ref
        dist = np.sqrt(np.mean(diff**2, axis=1))
        dist_score = 1.0 / (1.0 + dist)
        similarity = np.maximum(0.0, (corr + 1.0) / 2.0 * 80.0 + dist_score * 20.0)

    limit = int(limit_results)
    order = np.argsort(-similarity)[:limit]

    tgt_list = ref.tolist()
    results: list[dict] = []
    for k in order:
        i = int(k)
        row_norm = X_norm[i]
        results.append(
            {
                "stock_code": stock_codes[i],
                "stock_name": stock_names[i],
                "similarity": float(similarity[i]),
                "target_trajectory": tgt_list,
                "candidate_trajectory": row_norm.tolist(),
                "dates": dates_blocks[i],
                "raw_prices": raw_price_blocks[i],
            }
        )

    return results
