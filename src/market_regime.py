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


def get_market_environment_score(
    target_date: str,
    *,
    index_code: str = "000300",
    db_path=None,
) -> int:
    """
    返回 0~100 的大盘环境分：收盘价高于 20 日均线则 +40 基础分，否则 +0，再加 20 底分 → 60 或 20。
    无足够指数数据时使用 ``MARKET_REGIME_MISSING_DATA_SCORE``（默认 50，触发熔断），并打印一次警告。
    """
    code = str(index_code).strip().zfill(6)
    cache_key = f"{code}:{str(target_date).strip()[:10]}"
    if cache_key in _SCORE_CACHE:
        return int(_SCORE_CACHE[cache_key])
    score_out = int(MARKET_REGIME_MISSING_DATA_SCORE)
    td = str(target_date).strip()[:10]
    path = db_path or DB_PATH
    init_db(path)
    index_label = "中证1000" if code == "000852" else ("沪深300" if code == "000300" else code)
    try:
        with get_connection(path) as conn:
            df_idx = pd.read_sql_query(
                """
                SELECT date, close FROM index_daily
                WHERE index_code = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 200
                """,
                conn,
                params=[code, td],
            )
        if len(df_idx) >= 20:
            df_idx = df_idx.sort_values("date")
            ma20 = float(df_idx["close"].rolling(20).mean().iloc[-1])
            last_close = float(df_idx["close"].iloc[-1])
            cond = last_close > ma20
            score_out = int((20 if cond else 0) + 40)
        else:
            if cache_key not in _MISSING_INDEX_WARNED:
                _MISSING_INDEX_WARNED.add(cache_key)
                print(
                    f"[大盘环境] {td} 本地{index_label}({code})指数不足 20 根，"
                    f"环境分={score_out}（QUANT_MARKET_REGIME_MISSING_SCORE，"
                    f"默认低于熔断线 {MARKET_REGIME_MIN_SCORE}）。"
                    "请运行 scripts/update_local_data.py（或 --only-index-sync）同步 index_daily。",
                    flush=True,
                )
    except Exception:
        if cache_key not in _MISSING_INDEX_WARNED:
            _MISSING_INDEX_WARNED.add(cache_key)
            print(
                f"[大盘环境] {td} 读取 index_daily({code}) 失败，环境分={score_out}（保守熔断）。"
                "请运行 scripts/update_local_data.py（或 --only-index-sync）。",
                flush=True,
            )
    _SCORE_CACHE[cache_key] = score_out
    return score_out


def get_hs300_market_environment_score(
    target_date: str,
    db_path=None,
) -> int:
    """沪深300 大盘环境分（兼容旧入口）。"""
    return get_market_environment_score(target_date, index_code="000300", db_path=db_path)


def compute_market_regime_score(
    target_date: str,
    db_path=None,
) -> int:
    """``run_daily`` / 题材模块统一入口：沪深300 大盘多头环境分（0~100）。"""
    return get_hs300_market_environment_score(target_date, db_path=db_path)


def get_index_momentum_return(
    target_date: str,
    *,
    index_code: str = "000852",
    lookback_days: int = 5,
    db_path=None,
) -> float | None:
    """指数 N 日收益率（小数），数据不足时返回 None。"""
    code = str(index_code).strip().zfill(6)
    days = max(1, int(lookback_days))
    td = str(target_date).strip()[:10]
    path = db_path or DB_PATH
    init_db(path)
    try:
        with get_connection(path) as conn:
            df_idx = pd.read_sql_query(
                """
                SELECT date, close FROM index_daily
                WHERE index_code = ? AND date <= ?
                ORDER BY date DESC
                LIMIT ?
                """,
                conn,
                params=[code, td, days + 1],
            )
    except Exception:
        return None
    if len(df_idx) < days + 1:
        return None
    df_idx = df_idx.sort_values("date")
    last_close = float(df_idx["close"].iloc[-1])
    base_close = float(df_idx["close"].iloc[-(days + 1)])
    if base_close <= 0:
        return None
    return (last_close / base_close) - 1.0


def short_term_market_allows_trading(
    target_date: str,
    *,
    min_score: int = 60,
    index_code: str = "000852",
    momentum_days: int = 5,
    momentum_min: float = 0.0,
    db_path=None,
) -> tuple[bool, int, float | None]:
    """短线专用：MA20 环境分 + 指数 N 日动量双过滤。Returns (允许, 环境分, 动量)。"""
    score = get_market_environment_score(
        target_date, index_code=index_code, db_path=db_path
    )
    mom = get_index_momentum_return(
        target_date,
        index_code=index_code,
        lookback_days=momentum_days,
        db_path=db_path,
    )
    allows = score >= int(min_score) and mom is not None and mom > float(momentum_min)
    return allows, score, mom


def market_environment_allows_trading(
    target_date: str,
    *,
    min_score: int | None = None,
    index_code: str = "000300",
    db_path=None,
) -> tuple[bool, int]:
    """``(是否允许选股, 环境分)``。"""
    score = get_market_environment_score(
        target_date, index_code=index_code, db_path=db_path
    )
    threshold = int(MARKET_REGIME_MIN_SCORE if min_score is None else min_score)
    return score >= threshold, score
