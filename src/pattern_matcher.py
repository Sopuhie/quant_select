"""
图形相似度匹配引擎。
使用本地 SQLite 行情，通过皮尔逊相关系数与归一化序列欧氏距离辅助项对全市场进行形态检索。
"""
from __future__ import annotations

import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import DB_PATH


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


def find_similar_patterns(
    target_code: str,
    start_date: str,
    end_date: str,
    *,
    compare_days: int = 120,
    limit_results: int = 5,
) -> list[dict]:
    """
    在本地行情库中，寻找与目标股票给定区间收盘价形态最相似的个股（最近窗口长度与模板一致）。

    Args:
        target_code: 模板股票代码
        start_date / end_date: 模板区间（YYYY-MM-DD）
        compare_days: 日历跨度近似控制候选数据加载起点（越大越耗内存，默认覆盖约一季以上交易日）
        limit_results: 返回前几条
    """
    target_code = str(target_code).strip().zfill(6)
    start_date = str(start_date).strip()[:10]
    end_date = str(end_date).strip()[:10]

    conn = sqlite3.connect(str(DB_PATH))
    sql_target = """
        SELECT date, close FROM stock_daily_kline
        WHERE stock_code = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
    """
    target_df = pd.read_sql_query(
        sql_target, conn, params=(target_code, start_date, end_date)
    )

    if target_df.empty or len(target_df) < 5:
        conn.close()
        return []

    window_size = len(target_df)
    target_series = get_normalized_series(target_df["close"])
    if len(target_series) != window_size:
        conn.close()
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
    conn.close()

    if all_df.empty:
        return []

    all_df["close"] = pd.to_numeric(all_df["close"], errors="coerce")
    all_df = all_df.dropna(subset=["close"])

    results: list[dict] = []
    grouped = all_df.groupby("stock_code", sort=False)
    for code, group in grouped:
        if str(code).zfill(6) == target_code:
            continue
        if len(group) < window_size:
            continue

        candidate_window = group.tail(window_size).copy()
        candidate_name = str(candidate_window["stock_name"].iloc[0] or "").strip()
        candidate_prices = candidate_window["close"]

        candidate_series = get_normalized_series(candidate_prices)
        if len(candidate_series) != window_size:
            continue

        corr = np.corrcoef(target_series, candidate_series)[0, 1]
        if np.isnan(corr):
            corr = -1.0

        distance = float(np.sqrt(np.mean((target_series - candidate_series) ** 2)))
        dist_score = 1.0 / (1.0 + distance)

        similarity_pct = max(
            0.0, (corr + 1.0) / 2.0 * 80.0 + dist_score * 20.0
        )

        dates_fmt = (
            pd.to_datetime(candidate_window["date"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        results.append(
            {
                "stock_code": str(code).zfill(6),
                "stock_name": candidate_name,
                "similarity": similarity_pct,
                "target_trajectory": target_series.tolist(),
                "candidate_trajectory": candidate_series.tolist(),
                "dates": dates_fmt,
                "raw_prices": candidate_prices.astype(float).tolist(),
            }
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[: int(limit_results)]
