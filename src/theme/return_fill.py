# -*- coding: utf-8 -*-
"""回填 theme_daily_selections 的 1/5/10/60 日收盘收益率。"""
from __future__ import annotations

import sqlite3
from typing import Any

from pathlib import Path

from ..config import DB_PATH
from ..database import fetch_stock_daily_bars_until
from ..return_updater import RETURN_OFFSETS, _returns_from_hist
from ..utils import get_kline_incremental_end_trade_date
from .db import ensure_theme_tables, update_theme_selection_returns

_RETURN_COLS = tuple(c for c, _ in RETURN_OFFSETS)


def fill_theme_returns_for_date(
    conn: sqlite3.Connection,
    trade_date: str,
    *,
    db_path: Path | None = None,
    commit: bool = False,
) -> dict[str, int]:
    """为指定信号日补齐缺失的持有期收益。返回各字段更新计数。"""
    ensure_theme_tables(conn)
    td = str(trade_date).strip()[:10]
    cols_sql = ", ".join(_RETURN_COLS)
    where = " OR ".join(f"{col} IS NULL" for col in _RETURN_COLS)
    rows = conn.execute(
        f"""
        SELECT trade_date, stock_code, close_price, {cols_sql}
        FROM theme_daily_selections
        WHERE trade_date = ? AND ({where})
        """,
        (td,),
    ).fetchall()
    if not rows:
        return {"rows_touched": 0}

    end_anchor = get_kline_incremental_end_trade_date()
    hist_cache: dict[str, Any] = {}
    touched = 0

    for r in rows:
        code = str(r["stock_code"]).strip().zfill(6)
        if code not in hist_cache:
            hist_cache[code] = fetch_stock_daily_bars_until(
                code, end_anchor, db_path=db_path or DB_PATH, connection=conn
            )
        hist = hist_cache[code]
        close_hint = r["close_price"]
        computed = _returns_from_hist(
            hist if hist is not None and not hist.empty else None,
            td,
            float(close_hint) if close_hint is not None else None,
        )
        kwargs: dict[str, float] = {}
        for col, _off in RETURN_OFFSETS:
            if r[col] is None and computed.get(col) is not None:
                kwargs[col] = float(computed[col])
        if not kwargs:
            continue
        update_theme_selection_returns(
            conn,
            td,
            code,
            next_day_return=kwargs.get("next_day_return"),
            hold_5d_return=kwargs.get("hold_5d_return"),
            hold_10d_return=kwargs.get("hold_10d_return"),
            hold_60d_return=kwargs.get("hold_60d_return"),
        )
        touched += 1

    if commit:
        conn.commit()
    return {"rows_touched": touched}
