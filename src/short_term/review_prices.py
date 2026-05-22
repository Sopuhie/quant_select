# -*- coding: utf-8 -*-
"""短线复盘价：信号日 T 后的 T+1、T+2 开收盘价（来自本地 K 线）。"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from src.utils import next_trade_day_after

REVIEW_PRICE_KEYS = ("t1_open", "t1_close", "t2_open", "t2_close")


def calc_t1_buy_t2_sell_return(
    t1_open: float | None,
    t1_close: float | None,
    t2_close: float | None,
) -> float | None:
    """
    T+1 买入 → T+2 收盘卖出的区间涨跌幅（小数，如 0.05 表示 5%）。

    买入价优先 T+1 开盘价；无开盘价则用 T+1 收盘价。卖出价为 T+2 收盘价。
    """
    buy: float | None = None
    for px in (t1_open, t1_close):
        if px is not None:
            try:
                v = float(px)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                buy = v
                break
    if buy is None or t2_close is None:
        return None
    try:
        sell = float(t2_close)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(sell) or sell <= 0:
        return None
    return (sell - buy) / buy


def resolve_t1_t2_dates_from_kline(
    conn: sqlite3.Connection,
    signal_trade_date: str,
) -> tuple[str | None, str | None]:
    """从本地 ``stock_daily_kline`` 取信号日之后第 1、第 2 个有数据的交易日。"""
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


def resolve_t1_t2_dates(
    signal_trade_date: str,
    conn: sqlite3.Connection | None = None,
) -> tuple[str | None, str | None]:
    """
    信号日 T → (T+1 交易日, T+2 交易日)。

    优先用本地 K 线库中实际存在的后续交易日；无数据时退回新浪/工作日历。
    """
    td = str(signal_trade_date).strip()[:10]
    if conn is not None:
        t1, t2 = resolve_t1_t2_dates_from_kline(conn, td)
        if t1:
            return t1, t2
    t1 = next_trade_day_after(td)
    if not t1:
        return None, None
    t2 = next_trade_day_after(t1)
    return t1, t2


def fetch_review_ohlc_map(
    conn: sqlite3.Connection,
    signal_trade_date: str,
    stock_codes: list[str],
) -> dict[str, dict[str, float | None]]:
    """
    返回 ``{code: {t1_open, t1_close, t2_open, t2_close}}``，无 K 线则为 None。
    """
    codes = sorted({str(c).strip().zfill(6) for c in stock_codes if str(c).strip()})
    out: dict[str, dict[str, float | None]] = {
        c: {k: None for k in REVIEW_PRICE_KEYS} for c in codes
    }
    if not codes:
        return out

    t1, t2 = resolve_t1_t2_dates(signal_trade_date, conn)
    if not t1:
        return out

    dates = [t1]
    if t2:
        dates.append(t2)

    placeholders_c = ",".join("?" * len(codes))
    placeholders_d = ",".join("?" * len(dates))
    sql = f"""
        SELECT stock_code, date, open, close
        FROM stock_daily_kline
        WHERE stock_code IN ({placeholders_c})
          AND date IN ({placeholders_d})
    """
    cur = conn.execute(sql, [*codes, *dates])
    by_code_date: dict[tuple[str, str], dict[str, float]] = {}
    for row in cur.fetchall():
        code = str(row[0]).strip().zfill(6)
        d = str(row[1]).strip()[:10]
        try:
            o = float(row[2]) if row[2] is not None else float("nan")
            c = float(row[3]) if row[3] is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if np.isfinite(o) and np.isfinite(c):
            by_code_date[(code, d)] = {"open": o, "close": c}

    for code in codes:
        if t1 and (code, t1) in by_code_date:
            out[code]["t1_open"] = by_code_date[(code, t1)]["open"]
            out[code]["t1_close"] = by_code_date[(code, t1)]["close"]
        if t2 and (code, t2) in by_code_date:
            out[code]["t2_open"] = by_code_date[(code, t2)]["open"]
            out[code]["t2_close"] = by_code_date[(code, t2)]["close"]

    return out


def enrich_rows_with_review_prices(
    conn: sqlite3.Connection,
    signal_trade_date: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """为待写入行附加 T1/T2 开收盘价字段。"""
    codes = [str(r.get("stock_code", "")) for r in rows]
    ohlc_map = fetch_review_ohlc_map(conn, signal_trade_date, codes)
    for r in rows:
        code = str(r.get("stock_code", "")).strip().zfill(6)
        px = ohlc_map.get(code, {})
        for k in REVIEW_PRICE_KEYS:
            val = px.get(k)
            if val is not None:
                r[k] = val
    return rows


def auto_fill_review_prices(
    conn: sqlite3.Connection,
    trade_date: str | None = None,
    *,
    commit: bool = True,
) -> dict[str, int]:
    """
    从本地 K 线自动回填 ``short_daily_selections`` 的 T1/T2 开收盘价。

    ``trade_date`` 为 None 时处理库内全部信号日。
    返回 ``{"dates": n, "rows": n, "fields": n}``。
    """
    from .db import ensure_short_term_tables, refresh_short_review_prices

    ensure_short_term_tables(conn)
    if trade_date:
        dates = [str(trade_date).strip()[:10]]
    else:
        rows = conn.execute(
            "SELECT DISTINCT trade_date FROM short_daily_selections ORDER BY trade_date"
        ).fetchall()
        dates = [str(r[0]).strip()[:10] for r in rows if r[0]]

    total_rows = 0
    total_fields = 0
    for td in dates:
        n = refresh_short_review_prices(conn, td, commit=False)
        total_rows += n
        if n:
            cur = conn.execute(
                """
                SELECT COUNT(*) FROM short_daily_selections
                WHERE trade_date = ?
                  AND (t1_open IS NOT NULL OR t1_close IS NOT NULL
                       OR t2_open IS NOT NULL OR t2_close IS NOT NULL)
                """,
                (td,),
            ).fetchone()
            total_fields += int(cur[0] or 0) if cur else 0
    if commit:
        conn.commit()
    return {"dates": len(dates), "rows": total_rows, "filled_rows": total_fields}
