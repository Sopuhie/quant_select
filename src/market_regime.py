"""
沪深 300 大盘环境分（0~100），与 ``ThemeAlphaStrategy.get_market_score`` 规则对齐。
用于 run_daily 择时熔断与题材选股环境过滤。
"""
from __future__ import annotations

import pandas as pd

from .config import DB_PATH, MARKET_REGIME_MIN_SCORE
from .database import get_connection, init_db

_SCORE_CACHE: dict[str, int] = {}


def get_hs300_market_environment_score(
    target_date: str,
    db_path=None,
) -> int:
    """
    返回 0~100 的大盘环境分：收盘价高于 20 日均线则 +40 基础分，否则 +0，再加 20 底分 → 60 或 20。
    无指数数据时默认 60（不阻断），与题材模块一致。
    """
    td = str(target_date).strip()[:10]
    if td in _SCORE_CACHE:
        return int(_SCORE_CACHE[td])
    score_out = 60
    path = db_path or DB_PATH
    init_db(path)
    try:
        with get_connection(path) as conn:
            df_idx = pd.read_sql_query(
                """
                SELECT date, close FROM index_daily
                WHERE index_code = '000300' AND date <= ?
                ORDER BY date DESC
                LIMIT 200
                """,
                conn,
                params=[td],
            )
        if len(df_idx) >= 20:
            df_idx = df_idx.sort_values("date")
            ma20 = float(df_idx["close"].rolling(20).mean().iloc[-1])
            last_close = float(df_idx["close"].iloc[-1])
            cond = last_close > ma20
            score_out = int((20 if cond else 0) + 40)
    except Exception:
        score_out = 60
    _SCORE_CACHE[td] = score_out
    return score_out


def market_environment_allows_trading(
    target_date: str,
    *,
    min_score: int | None = None,
    db_path=None,
) -> tuple[bool, int]:
    """``(是否允许选股, 环境分)``。"""
    score = get_hs300_market_environment_score(target_date, db_path=db_path)
    threshold = int(MARKET_REGIME_MIN_SCORE if min_score is None else min_score)
    return score >= threshold, score
