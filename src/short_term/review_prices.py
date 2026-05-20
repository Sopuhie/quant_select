# -*- coding: utf-8 -*-
"""短线复盘价：信号日 T 后的 T+1、T+2 开收盘价（来自本地 K 线）。"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from src.utils import next_trade_day_after

REVIEW_PRICE_KEYS = ("t1_open", "t1_close", "t2_open", "t2_close")


def resolve_t1_t2_dates(signal_trade_date: str) -> tuple[str | None, str | None]:
    """信号日 T → (T+1 交易日, T+2 交易日)。"""
    td = str(signal_trade_date).strip()[:10]
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

    t1, t2 = resolve_t1_t2_dates(signal_trade_date)
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
            r[k] = px.get(k)
    return rows
