# -*- coding: utf-8 -*-
"""短线选股结果表（独立建表，不改动 database.SCHEMA_SQL）。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

from .config import SHORT_HOLDING_DAYS
from .review_prices import REVIEW_PRICE_KEYS, enrich_rows_with_review_prices, fetch_review_ohlc_map

SHORT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS short_daily_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    rank INTEGER,
    rule_score REAL,
    close_price REAL,
    day_change_pct REAL,
    vol_ratio_5d REAL,
    kdj_j REAL,
    macd_bar REAL,
    hold_plan TEXT,
    advice_text TEXT,
    detail_json TEXT,
    t1_open REAL,
    t1_close REAL,
    t2_open REAL,
    t2_close REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_short_sel_date ON short_daily_selections(trade_date);
CREATE INDEX IF NOT EXISTS idx_short_sel_date_rank ON short_daily_selections(trade_date, rank);
"""

_REVIEW_MIGRATE_COLS = (
    ("t1_open", "REAL"),
    ("t1_close", "REAL"),
    ("t2_open", "REAL"),
    ("t2_close", "REAL"),
)


def ensure_short_term_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SHORT_SCHEMA_SQL)
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(short_daily_selections)")
    }
    for col, typ in _REVIEW_MIGRATE_COLS:
        if col not in existing:
            conn.execute(
                f"ALTER TABLE short_daily_selections ADD COLUMN {col} {typ}"
            )
    conn.commit()


def short_selection_exists(conn: sqlite3.Connection, trade_date: str) -> bool:
    ensure_short_term_tables(conn)
    row = conn.execute(
        "SELECT 1 FROM short_daily_selections WHERE trade_date = ? LIMIT 1",
        (str(trade_date).strip()[:10],),
    ).fetchone()
    return row is not None


def delete_short_selections_for_date(
    conn: sqlite3.Connection,
    trade_date: str,
    *,
    commit: bool = True,
) -> None:
    ensure_short_term_tables(conn)
    conn.execute(
        "DELETE FROM short_daily_selections WHERE trade_date = ?",
        (str(trade_date).strip()[:10],),
    )
    if commit:
        conn.commit()


def insert_short_daily_selections(
    conn: sqlite3.Connection,
    trade_date: str,
    rows: list[dict[str, Any]],
    *,
    commit: bool = True,
) -> int:
    ensure_short_term_tables(conn)
    td = str(trade_date).strip()[:10]
    rows = enrich_rows_with_review_prices(conn, td, rows)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for r in rows:
        detail = r.get("detail") or {}
        conn.execute(
            """
            INSERT OR REPLACE INTO short_daily_selections (
                trade_date, stock_code, stock_name, rank, rule_score,
                close_price, day_change_pct, vol_ratio_5d, kdj_j, macd_bar,
                hold_plan, advice_text, detail_json,
                t1_open, t1_close, t2_open, t2_close,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                td,
                str(r.get("stock_code", "")).zfill(6),
                r.get("stock_name"),
                r.get("rank"),
                r.get("rule_score"),
                r.get("close_price"),
                r.get("day_change_pct"),
                r.get("vol_ratio_5d"),
                r.get("kdj_j"),
                r.get("macd_bar"),
                r.get("hold_plan")
                or f"T+1 开盘买 → T+{1 + SHORT_HOLDING_DAYS} 开盘卖（持有 {SHORT_HOLDING_DAYS} 个交易日）",
                r.get("advice_text"),
                json.dumps(detail, ensure_ascii=False),
                r.get("t1_open"),
                r.get("t1_close"),
                r.get("t2_open"),
                r.get("t2_close"),
                now,
            ),
        )
        n += 1
    if commit:
        conn.commit()
    return n


def refresh_short_review_prices(
    conn: sqlite3.Connection,
    trade_date: str | None = None,
    *,
    commit: bool = True,
) -> int:
    """
    用本地 K 线回填/更新 T1/T2 开收盘价（复盘用）。
    ``trade_date`` 为 None 时处理全部信号日。
    """
    ensure_short_term_tables(conn)
    td = str(trade_date).strip()[:10] if trade_date else None
    if td:
        cur = conn.execute(
            """
            SELECT trade_date, stock_code FROM short_daily_selections
            WHERE trade_date = ?
            """,
            (td,),
        )
    else:
        cur = conn.execute(
            "SELECT trade_date, stock_code FROM short_daily_selections"
        )
    pairs = [(str(r[0])[:10], str(r[1]).zfill(6)) for r in cur.fetchall()]
    if not pairs:
        return 0

    by_date: dict[str, list[str]] = {}
    for d, code in pairs:
        by_date.setdefault(d, []).append(code)

    updated = 0
    for sig_date, codes in by_date.items():
        ohlc_map = fetch_review_ohlc_map(conn, sig_date, codes)
        for code in codes:
            px = ohlc_map.get(code, {})
            conn.execute(
                """
                UPDATE short_daily_selections
                SET t1_open = ?, t1_close = ?, t2_open = ?, t2_close = ?
                WHERE trade_date = ? AND stock_code = ?
                """,
                (
                    px.get("t1_open"),
                    px.get("t1_close"),
                    px.get("t2_open"),
                    px.get("t2_close"),
                    sig_date,
                    code,
                ),
            )
            updated += 1
    if commit:
        conn.commit()
    return updated


def load_short_selections_df(
    conn: sqlite3.Connection,
    trade_date: str,
) -> pd.DataFrame:
    ensure_short_term_tables(conn)
    return pd.read_sql_query(
        """
        SELECT trade_date AS 信号日, stock_code AS 股票代码, stock_name AS 股票名称,
               rank AS 排名, rule_score AS 规则得分, close_price AS 信号日收盘价,
               day_change_pct AS 日涨幅, vol_ratio_5d AS 五日量比,
               kdj_j AS KDJ_J, macd_bar AS MACD柱,
               t1_open AS T1开盘价, t1_close AS T1收盘价,
               t2_open AS T2开盘价, t2_close AS T2收盘价,
               hold_plan AS 持仓计划, advice_text AS 实盘建议
        FROM short_daily_selections
        WHERE trade_date = ?
        ORDER BY rank ASC
        """,
        conn,
        params=[str(trade_date).strip()[:10]],
    )
