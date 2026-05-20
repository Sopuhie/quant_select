# -*- coding: utf-8 -*-
"""短线选股结果表（独立建表，不改动 database.SCHEMA_SQL）。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

from .config import SHORT_HOLDING_DAYS

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
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_short_sel_date ON short_daily_selections(trade_date);
CREATE INDEX IF NOT EXISTS idx_short_sel_date_rank ON short_daily_selections(trade_date, rank);
"""


def ensure_short_term_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SHORT_SCHEMA_SQL)
    conn.commit()


def short_selection_exists(conn: sqlite3.Connection, trade_date: str) -> bool:
    ensure_short_term_tables(conn)
    row = conn.execute(
        "SELECT 1 FROM short_daily_selections WHERE trade_date = ? LIMIT 1",
        (str(trade_date).strip()[:10],),
    ).fetchone()
    return row is not None


def delete_short_selections_for_date(conn: sqlite3.Connection, trade_date: str) -> None:
    ensure_short_term_tables(conn)
    conn.execute(
        "DELETE FROM short_daily_selections WHERE trade_date = ?",
        (str(trade_date).strip()[:10],),
    )
    conn.commit()


def insert_short_daily_selections(
    conn: sqlite3.Connection,
    trade_date: str,
    rows: list[dict[str, Any]],
) -> int:
    ensure_short_term_tables(conn)
    td = str(trade_date).strip()[:10]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for r in rows:
        detail = r.get("detail") or {}
        conn.execute(
            """
            INSERT OR REPLACE INTO short_daily_selections (
                trade_date, stock_code, stock_name, rank, rule_score,
                close_price, day_change_pct, vol_ratio_5d, kdj_j, macd_bar,
                hold_plan, advice_text, detail_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                now,
            ),
        )
        n += 1
    conn.commit()
    return n


def load_short_selections_df(
    conn: sqlite3.Connection,
    trade_date: str,
) -> pd.DataFrame:
    ensure_short_term_tables(conn)
    return pd.read_sql_query(
        """
        SELECT trade_date AS 信号日, stock_code AS 股票代码, stock_name AS 股票名称,
               rank AS 排名, rule_score AS 规则得分, close_price AS 收盘价,
               day_change_pct AS 日涨幅, vol_ratio_5d AS 五日量比,
               kdj_j AS KDJ_J, macd_bar AS MACD柱, hold_plan AS 持仓计划,
               advice_text AS 实盘建议
        FROM short_daily_selections
        WHERE trade_date = ?
        ORDER BY rank ASC
        """,
        conn,
        params=[str(trade_date).strip()[:10]],
    )
