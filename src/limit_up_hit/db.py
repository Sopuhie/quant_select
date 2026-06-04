# -*- coding: utf-8 -*-
"""打板模块 SQLite 表：``luh_daily_selections``、``luh_order_tracker``。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from .config import LUH_HOLD_PLAN
from .review_prices import REVIEW_PRICE_KEYS, enrich_rows_with_review_prices

LUH_DAILY_SELECTIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS luh_daily_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    close_price REAL,
    pct_change REAL,
    board_streak INTEGER,
    seal_strength REAL,
    turnover REAL,
    board_score REAL,
    is_executed INTEGER DEFAULT 0,
    rank INTEGER,
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
"""

LUH_ORDER_TRACKER_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS luh_order_tracker (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    buy_date TEXT NOT NULL,
    buy_price REAL NOT NULL,
    sell_date TEXT,
    sell_price REAL,
    hold_days INTEGER DEFAULT 0,
    pnl_ratio REAL,
    status TEXT DEFAULT 'HOLDING',
    stop_loss_triggered INTEGER DEFAULT 0,
    signal_rank INTEGER,
    board_score REAL,
    board_streak INTEGER,
    exit_reason TEXT,
    hold_plan TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(buy_date, stock_code)
);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_luh_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(LUH_DAILY_SELECTIONS_TABLE_DDL)
    conn.executescript(LUH_ORDER_TRACKER_TABLE_DDL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_luh_sel_date "
        "ON luh_daily_selections(trade_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_luh_date_score "
        "ON luh_daily_selections(trade_date, board_score DESC)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uidx_luh_date_code "
        "ON luh_daily_selections(trade_date, stock_code)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_luh_order_buy_date "
        "ON luh_order_tracker(buy_date)"
    )
    conn.commit()


def luh_selection_exists(conn: sqlite3.Connection, trade_date: str) -> bool:
    ensure_luh_tables(conn)
    row = conn.execute(
        "SELECT 1 FROM luh_daily_selections WHERE trade_date = ? LIMIT 1",
        (str(trade_date).strip()[:10],),
    ).fetchone()
    return row is not None


def delete_luh_selections_for_date(
    conn: sqlite3.Connection,
    trade_date: str,
    *,
    commit: bool = True,
) -> None:
    ensure_luh_tables(conn)
    conn.execute(
        "DELETE FROM luh_daily_selections WHERE trade_date = ?",
        (str(trade_date).strip()[:10],),
    )
    if commit:
        conn.commit()


def delete_luh_orders_for_buy_date(
    conn: sqlite3.Connection,
    buy_date: str,
    *,
    commit: bool = True,
) -> None:
    ensure_luh_tables(conn)
    conn.execute(
        "DELETE FROM luh_order_tracker WHERE buy_date = ?",
        (str(buy_date).strip()[:10],),
    )
    if commit:
        conn.commit()


def insert_luh_daily_selections(
    conn: sqlite3.Connection,
    trade_date: str,
    rows: list[dict[str, Any]],
    *,
    commit: bool = True,
) -> int:
    ensure_luh_tables(conn)
    td = str(trade_date).strip()[:10]
    enrich_rows_with_review_prices(conn, td, rows)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for row in rows:
        detail = row.get("detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        conn.execute(
            """
            INSERT OR REPLACE INTO luh_daily_selections (
                trade_date, stock_code, stock_name, close_price, pct_change,
                board_streak, seal_strength, turnover, board_score,
                is_executed, rank, hold_plan, advice_text, detail_json,
                t1_open, t1_close, t2_open, t2_close, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                td,
                str(row.get("stock_code", "")).strip().zfill(6),
                row.get("stock_name"),
                row.get("close_price"),
                row.get("pct_change"),
                row.get("board_streak"),
                row.get("seal_strength"),
                row.get("turnover"),
                row.get("board_score"),
                int(row.get("is_executed") or 0),
                row.get("rank"),
                row.get("hold_plan") or LUH_HOLD_PLAN,
                row.get("advice_text"),
                json.dumps(detail, ensure_ascii=False),
                row.get("t1_open"),
                row.get("t1_close"),
                row.get("t2_open"),
                row.get("t2_close"),
                now,
            ),
        )
        n += 1
    if commit:
        conn.commit()
    return n


def upsert_luh_order(conn: sqlite3.Connection, order: dict[str, Any]) -> None:
    ensure_luh_tables(conn)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO luh_order_tracker (
            stock_code, stock_name, buy_date, buy_price, sell_date, sell_price,
            hold_days, pnl_ratio, status, stop_loss_triggered,
            signal_rank, board_score, board_streak, exit_reason, hold_plan,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(buy_date, stock_code) DO UPDATE SET
            stock_name = excluded.stock_name,
            buy_price = excluded.buy_price,
            sell_date = excluded.sell_date,
            sell_price = excluded.sell_price,
            hold_days = excluded.hold_days,
            pnl_ratio = excluded.pnl_ratio,
            status = excluded.status,
            stop_loss_triggered = excluded.stop_loss_triggered,
            signal_rank = excluded.signal_rank,
            board_score = excluded.board_score,
            board_streak = excluded.board_streak,
            exit_reason = excluded.exit_reason,
            hold_plan = excluded.hold_plan,
            updated_at = excluded.updated_at
        """,
        (
            str(order.get("stock_code", "")).strip().zfill(6),
            order.get("stock_name"),
            str(order.get("buy_date", "")).strip()[:10],
            order.get("buy_price"),
            order.get("sell_date"),
            order.get("sell_price"),
            order.get("hold_days"),
            order.get("pnl_ratio"),
            order.get("status"),
            int(order.get("stop_loss_triggered") or 0),
            order.get("signal_rank"),
            order.get("board_score"),
            order.get("board_streak"),
            order.get("exit_reason"),
            order.get("hold_plan") or LUH_HOLD_PLAN,
            now,
            now,
        ),
    )


def load_luh_orders_for_buy_date(
    conn: sqlite3.Connection,
    buy_date: str,
) -> list[dict[str, Any]]:
    ensure_luh_tables(conn)
    cur = conn.execute(
        """
        SELECT stock_code, stock_name, buy_date, buy_price, sell_date, sell_price,
               hold_days, pnl_ratio, status, stop_loss_triggered,
               signal_rank, board_score, board_streak, exit_reason, hold_plan
        FROM luh_order_tracker
        WHERE buy_date = ?
        ORDER BY signal_rank ASC
        """,
        (str(buy_date).strip()[:10],),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def list_luh_selection_trade_dates(conn: sqlite3.Connection) -> list[str]:
    ensure_luh_tables(conn)
    rows = conn.execute(
        """
        SELECT DISTINCT trade_date FROM luh_daily_selections
        ORDER BY trade_date DESC
        """
    ).fetchall()
    return [str(r[0]).strip()[:10] for r in rows if r[0]]


def load_luh_selections_for_date(
    conn: sqlite3.Connection,
    trade_date: str,
) -> list[dict[str, Any]]:
    ensure_luh_tables(conn)
    cur = conn.execute(
        """
        SELECT trade_date, stock_code, stock_name, close_price, pct_change,
               board_streak, seal_strength, turnover, board_score,
               is_executed, rank, hold_plan, advice_text, detail_json,
               t1_open, t1_close, t2_open, t2_close
        FROM luh_daily_selections
        WHERE trade_date = ?
        ORDER BY rank ASC
        """,
        (str(trade_date).strip()[:10],),
    )
    cols = [d[0] for d in cur.description]
    out: list[dict[str, Any]] = []
    for row in cur.fetchall():
        item = dict(zip(cols, row))
        dj = item.pop("detail_json", None)
        if dj:
            try:
                item["detail"] = json.loads(dj)
            except json.JSONDecodeError:
                item["detail"] = {}
        out.append(item)
    return out
