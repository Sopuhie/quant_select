"""
沪深 300 大盘环境分（0~100），与 ``ThemeAlphaStrategy.get_market_score`` 规则对齐。
用于 run_daily 择时熔断、题材选股环境过滤与 Ranker 融合权重。
"""
from __future__ import annotations

import pandas as pd

from .config import (
    DB_PATH,
    MARKET_REGIME_MIN_SCORE,
    MARKET_REGIME_MISSING_DATA_SCORE,
)
from .database import get_connection, init_db

_SCORE_CACHE: dict[str, int] = {}
_MISSING_INDEX_WARNED: set[str] = set()


def load_hs300_index_daily_local(
    target_date: str | None = None,
    *,
    db_path=None,
    limit: int = 200,
) -> pd.DataFrame:
    """
    从本地 ``index_daily`` 读取沪深300（000300）日线，列 ``date``、``close``（升序）。
    无数据时返回空 DataFrame。
    """
    path = db_path or DB_PATH
    init_db(path)
    td = str(target_date).strip()[:10] if target_date else None
    try:
        with get_connection(path) as conn:
            if td:
                df = pd.read_sql_query(
                    """
                    SELECT date, close FROM index_daily
                    WHERE index_code = '000300' AND date <= ?
                    ORDER BY date DESC
                    LIMIT ?
                    """,
                    conn,
                    params=[td, int(limit)],
                )
            else:
                df = pd.read_sql_query(
                    """
                    SELECT date, close FROM index_daily
                    WHERE index_code = '000300'
                    ORDER BY date DESC
                    LIMIT ?
                    """,
                    conn,
                    params=[int(limit)],
                )
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"])


def get_hs300_market_environment_score(
    target_date: str,
    db_path=None,
) -> int:
    """
    返回 0~100 的大盘环境分：收盘价高于 20 日均线则 +40 基础分，否则 +0，再加 20 底分 → 60 或 20。
    无足够指数数据时使用 ``MARKET_REGIME_MISSING_DATA_SCORE``（默认 50，触发熔断），并打印一次警告。
    """
    td = str(target_date).strip()[:10]
    if td in _SCORE_CACHE:
        return int(_SCORE_CACHE[td])
    score_out = int(MARKET_REGIME_MISSING_DATA_SCORE)
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
        else:
            if td not in _MISSING_INDEX_WARNED:
                _MISSING_INDEX_WARNED.add(td)
                print(
                    f"[大盘环境] {td} 本地沪深300指数不足 20 根，"
                    f"环境分={score_out}（QUANT_MARKET_REGIME_MISSING_SCORE，"
                    f"默认低于熔断线 {MARKET_REGIME_MIN_SCORE}）。"
                    "请同步 index_daily。",
                    flush=True,
                )
    except Exception:
        if td not in _MISSING_INDEX_WARNED:
            _MISSING_INDEX_WARNED.add(td)
            print(
                f"[大盘环境] {td} 读取 index_daily 失败，环境分={score_out}（保守熔断）。",
                flush=True,
            )
    _SCORE_CACHE[td] = score_out
    return score_out


def compute_market_regime_score(
    target_date: str,
    db_path=None,
) -> int:
    """``run_daily`` / 题材模块统一入口：沪深300 大盘多头环境分（0~100）。"""
    return get_hs300_market_environment_score(target_date, db_path=db_path)


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
