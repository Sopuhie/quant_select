# -*- coding: utf-8 -*-
"""打板复盘价：信号日 T 后的 T+1、T+2 开收盘价。"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

REVIEW_PRICE_KEYS = ("t1_open", "t1_close", "t2_open", "t2_close")


def resolve_t1_t2_dates(
    signal_trade_date: str,
    conn: sqlite3.Connection,
) -> tuple[str | None, str | None]:
    td = str(signal_trade_date).strip()[:10]
    rows = conn.execute(
        """
        SELECT DISTINCT date FROM stock_daily_kline
        WHERE date > ?
        ORDER BY date ASC
        LIMIT 2
        """,
        (td,),
    ).fetchall()
    if not rows:
        return None, None
    t1 = str(rows[0][0]).strip()[:10]
    t2 = str(rows[1][0]).strip()[:10] if len(rows) > 1 else None
    return t1, t2


def fetch_review_ohlc_map(
    conn: sqlite3.Connection,
    signal_trade_date: str,
    stock_codes: list[str],
) -> dict[str, dict[str, float | None]]:
    td = str(signal_trade_date).strip()[:10]
    t1, t2 = resolve_t1_t2_dates(td, conn)
    if not t1 or not stock_codes:
        return {}

    codes = [str(c).strip().zfill(6) for c in stock_codes]
    dates = [t1] + ([t2] if t2 else [])
    placeholders_c = ",".join("?" * len(codes))
    placeholders_d = ",".join("?" * len(dates))
    cur = conn.execute(
        f"""
        SELECT stock_code, date, open, close
        FROM stock_daily_kline
        WHERE stock_code IN ({placeholders_c})
          AND date IN ({placeholders_d})
        """,
        [*codes, *dates],
    )
    out: dict[str, dict[str, float | None]] = {
        c: {k: None for k in REVIEW_PRICE_KEYS} for c in codes
    }
    for row in cur.fetchall():
        code = str(row[0]).strip().zfill(6)
        d = str(row[1]).strip()[:10]
        try:
            o, c = float(row[2]), float(row[3])
        except (TypeError, ValueError):
            continue
        if code not in out:
            continue
        if d == t1:
            out[code]["t1_open"] = o if np.isfinite(o) else None
            out[code]["t1_close"] = c if np.isfinite(c) else None
        elif t2 and d == t2:
            out[code]["t2_open"] = o if np.isfinite(o) else None
            out[code]["t2_close"] = c if np.isfinite(c) else None
    return out


def enrich_rows_with_review_prices(
    conn: sqlite3.Connection,
    trade_date: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    codes = [str(r.get("stock_code", "")).strip().zfill(6) for r in rows]
    ohlc_map = fetch_review_ohlc_map(conn, trade_date, codes)
    for row in rows:
        code = str(row.get("stock_code", "")).strip().zfill(6)
        prices = ohlc_map.get(code, {})
        for k in REVIEW_PRICE_KEYS:
            if prices.get(k) is not None:
                row[k] = prices[k]
    return rows


def auto_fill_review_prices(
    conn: sqlite3.Connection,
    trade_date: str | None,
    *,
    commit: bool = True,
) -> dict[str, int]:
    from .db import ensure_luh_tables

    ensure_luh_tables(conn)
    if trade_date:
        cur = conn.execute(
            """
            SELECT id, stock_code FROM luh_daily_selections
            WHERE trade_date = ?
            """,
            (str(trade_date).strip()[:10],),
        )
    else:
        cur = conn.execute(
            "SELECT id, stock_code, trade_date FROM luh_daily_selections"
        )
    rows = cur.fetchall()
    updated = 0
    for row in rows:
        if trade_date:
            rid, code = row[0], row[1]
            td = str(trade_date).strip()[:10]
        else:
            rid, code, td = row[0], row[1], row[2]
        ohlc = fetch_review_ohlc_map(conn, td, [code]).get(
            str(code).strip().zfill(6), {}
        )
        if not any(ohlc.get(k) is not None for k in REVIEW_PRICE_KEYS):
            continue
        conn.execute(
            """
            UPDATE luh_daily_selections
            SET t1_open = ?, t1_close = ?, t2_open = ?, t2_close = ?
            WHERE id = ?
            """,
            (
                ohlc.get("t1_open"),
                ohlc.get("t1_close"),
                ohlc.get("t2_open"),
                ohlc.get("t2_close"),
                rid,
            ),
        )
        updated += 1
    if commit:
        conn.commit()
    return {"rows": updated}
