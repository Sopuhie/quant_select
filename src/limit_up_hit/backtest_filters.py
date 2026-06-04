# -*- coding: utf-8 -*-
"""打板回测：大盘环境、涨停家数、同概念板块共振过滤（纯日线）。"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.factor_calculator import is_bar_limit_up

from .config import (
    LUH_BT_INDEX_RET_DAYS,
    LUH_BT_MIN_CONCEPT_LIMIT_UP,
    LUH_BT_MIN_INDEX_RET,
    LUH_BT_MIN_MARKET_LIMIT_UP,
    LUH_MARKET_INDEX_CODE,
)

_LIMIT_UP_COUNT_CACHE: dict[str, int] = {}
_INDEX_RET_CACHE: dict[str, float | None] = {}


def _limit_move_ratio(stock_code: str) -> float:
    c = str(stock_code).strip().zfill(6)
    if c.startswith(("300", "301", "688")):
        return 0.20
    if c.startswith(("8", "4")):
        return 0.30
    return 0.10


def is_limit_up_close(
    close: float,
    prev_close: float,
    stock_code: str,
) -> bool:
    return is_bar_limit_up(float(close), float(prev_close), stock_code)


def count_market_limit_up_stocks(
    conn: sqlite3.Connection,
    trade_date: str,
    *,
    use_cache: bool = True,
) -> int:
    """信号日全市场收盘涨停家数（按板块涨跌幅规则）。"""
    td = str(trade_date).strip()[:10]
    if use_cache and td in _LIMIT_UP_COUNT_CACHE:
        return int(_LIMIT_UP_COUNT_CACHE[td])

    df = pd.read_sql_query(
        """
        SELECT k.stock_code, k.close, p.close AS prev_close
        FROM stock_daily_kline k
        INNER JOIN stock_daily_kline p
            ON p.stock_code = k.stock_code
           AND p.date = (
               SELECT MAX(date) FROM stock_daily_kline
               WHERE stock_code = k.stock_code AND date < k.date
           )
        WHERE k.date = ?
          AND k.volume > 0
        """,
        conn,
        params=[td],
    )
    n = 0
    if not df.empty:
        for _, row in df.iterrows():
            code = str(row["stock_code"]).strip().zfill(6)
            try:
                c = float(row["close"])
                p = float(row["prev_close"])
            except (TypeError, ValueError):
                continue
            if is_limit_up_close(c, p, code):
                n += 1
    if use_cache:
        _LIMIT_UP_COUNT_CACHE[td] = int(n)
    return int(n)


def index_n_day_return(
    conn: sqlite3.Connection,
    trade_date: str,
    *,
    index_code: str | None = None,
    days: int | None = None,
    use_cache: bool = True,
) -> float | None:
    """锚定指数 N 日收益率（小数）。"""
    code = str(index_code or LUH_MARKET_INDEX_CODE).strip().zfill(6)
    td = str(trade_date).strip()[:10]
    n_days = int(days if days is not None else LUH_BT_INDEX_RET_DAYS)
    cache_key = f"{code}:{td}:{n_days}"
    if use_cache and cache_key in _INDEX_RET_CACHE:
        return _INDEX_RET_CACHE[cache_key]

    rows = conn.execute(
        """
        SELECT close FROM index_daily
        WHERE index_code = ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
        """,
        (code, td, n_days + 1),
    ).fetchall()
    ret: float | None = None
    if len(rows) >= n_days + 1:
        try:
            last_c = float(rows[0][0])
            base_c = float(rows[n_days][0])
        except (TypeError, ValueError):
            base_c = 0.0
            last_c = 0.0
        if base_c > 0:
            ret = (last_c / base_c) - 1.0
    if use_cache:
        _INDEX_RET_CACHE[cache_key] = ret
    return ret


def backtest_market_allows_trading(
    conn: sqlite3.Connection,
    trade_date: str,
) -> tuple[bool, dict[str, Any]]:
    """
    回测大盘闸：指数 20 日涨幅 > 0 且 全市场涨停家数 > 阈值。
    """
    td = str(trade_date).strip()[:10]
    idx_ret = index_n_day_return(conn, td)
    limit_up_n = count_market_limit_up_stocks(conn, td)
    min_ret = float(LUH_BT_MIN_INDEX_RET)
    min_lu = int(LUH_BT_MIN_MARKET_LIMIT_UP)
    allows = (
        idx_ret is not None
        and float(idx_ret) > min_ret
        and limit_up_n >= min_lu
    )
    detail = {
        "index_ret_20d": idx_ret,
        "market_limit_up_count": limit_up_n,
        "min_index_ret": min_ret,
        "min_market_limit_up": min_lu,
    }
    return bool(allows), detail


def load_stock_concept_boards(
    conn: sqlite3.Connection,
    stock_code: str,
) -> list[str]:
    code = str(stock_code).strip().zfill(6)
    try:
        rows = conn.execute(
            "SELECT DISTINCT board_name FROM stock_concept_boards WHERE stock_code = ?",
            (code,),
        ).fetchall()
        if rows:
            return [str(r[0]).strip() for r in rows if r[0]]
    except sqlite3.Error:
        pass
    try:
        from src.board_stocks import get_stock_boards

        boards = get_stock_boards(code)
        if boards:
            return boards
    except Exception:
        pass
    return []


def max_concept_limit_up_count(
    conn: sqlite3.Connection,
    trade_date: str,
    stock_code: str,
    *,
    limit_up_codes: set[str] | None = None,
) -> int:
    """个股所属概念板块中，信号日涨停家数最大值。"""
    boards = load_stock_concept_boards(conn, stock_code)
    if not boards:
        return 0
    td = str(trade_date).strip()[:10]
    if limit_up_codes is None:
        limit_up_codes = limit_up_codes_on_date(conn, td)
    if not limit_up_codes:
        return 0

    best = 0
    for board in boards:
        try:
            members = conn.execute(
                "SELECT DISTINCT stock_code FROM stock_concept_boards WHERE board_name = ?",
                (board,),
            ).fetchall()
            codes = {str(r[0]).strip().zfill(6) for r in members if r[0]}
        except sqlite3.Error:
            codes = set()
        if not codes:
            try:
                from src.board_stocks import BOARD_STOCKS_PATH
                import json

                if BOARD_STOCKS_PATH.exists():
                    data = json.loads(BOARD_STOCKS_PATH.read_text(encoding="utf-8"))
                    boards_map = (data.get("boards") or {}).get(board) or []
                    codes = {str(c).strip().zfill(6) for c in boards_map}
            except Exception:
                codes = set()
        n = len(codes & limit_up_codes)
        best = max(best, n)
    return int(best)


def limit_up_codes_on_date(conn: sqlite3.Connection, trade_date: str) -> set[str]:
    td = str(trade_date).strip()[:10]
    df = pd.read_sql_query(
        """
        SELECT k.stock_code, k.close, p.close AS prev_close
        FROM stock_daily_kline k
        INNER JOIN stock_daily_kline p
            ON p.stock_code = k.stock_code
           AND p.date = (
               SELECT MAX(date) FROM stock_daily_kline
               WHERE stock_code = k.stock_code AND date < k.date
           )
        WHERE k.date = ? AND k.volume > 0
        """,
        conn,
        params=[td],
    )
    out: set[str] = set()
    for _, row in df.iterrows():
        code = str(row["stock_code"]).strip().zfill(6)
        try:
            if is_limit_up_close(float(row["close"]), float(row["prev_close"]), code):
                out.add(code)
        except (TypeError, ValueError):
            continue
    return out


def backtest_concept_allows(
    conn: sqlite3.Connection,
    trade_date: str,
    stock_code: str,
    *,
    limit_up_codes: set[str] | None = None,
) -> tuple[bool, int]:
    """同概念板块涨停数 ≥ 阈值；无概念数据时不拦截。"""
    boards = load_stock_concept_boards(conn, stock_code)
    if not boards:
        return True, 0
    n = max_concept_limit_up_count(
        conn, trade_date, stock_code, limit_up_codes=limit_up_codes
    )
    return n >= int(LUH_BT_MIN_CONCEPT_LIMIT_UP), n


def clear_backtest_filter_cache() -> None:
    _LIMIT_UP_COUNT_CACHE.clear()
    _INDEX_RET_CACHE.clear()
