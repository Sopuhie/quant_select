# -*- coding: utf-8 -*-
"""热门题材选股历史表 ``theme_daily_selections``。"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

THEME_DAILY_SELECTIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS theme_daily_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    theme_tag TEXT,
    close_price REAL,
    vol_ratio_5d REAL,
    kdj_j REAL,
    macd_bar REAL,
    advice_text TEXT,
    market_score INTEGER,
    filter_keyword TEXT,
    rank INTEGER,
    next_day_return REAL,
    hold_5d_return REAL,
    hold_10d_return REAL,
    hold_60d_return REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, stock_code)
);
"""

_THEME_MIGRATE_COLS: tuple[tuple[str, str], ...] = (
    ("theme_tag", "TEXT"),
    ("close_price", "REAL"),
    ("vol_ratio_5d", "REAL"),
    ("kdj_j", "REAL"),
    ("macd_bar", "REAL"),
    ("advice_text", "TEXT"),
    ("market_score", "INTEGER"),
    ("filter_keyword", "TEXT"),
    ("rank", "INTEGER"),
    ("next_day_return", "REAL"),
    ("hold_5d_return", "REAL"),
    ("hold_10d_return", "REAL"),
    ("hold_60d_return", "REAL"),
    ("created_at", "TEXT"),
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_theme_tables(conn: sqlite3.Connection) -> None:
    conn.execute(THEME_DAILY_SELECTIONS_TABLE_DDL)
    cols = _table_columns(conn, "theme_daily_selections")
    for name, typ in _THEME_MIGRATE_COLS:
        if name not in cols:
            conn.execute(
                f"ALTER TABLE theme_daily_selections ADD COLUMN {name} {typ}"
            )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_theme_sel_date "
        "ON theme_daily_selections(trade_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_theme_sel_date_rank "
        "ON theme_daily_selections(trade_date, rank)"
    )


def _parse_price_cell(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        t = re.sub(r"[^\d.\-]", "", str(val))
        return float(t) if t else None
    except (TypeError, ValueError):
        return None


def _parse_vol_ratio_cell(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        t = str(val).replace("倍", "").strip()
        return float(t) if t else None
    except (TypeError, ValueError):
        return None


def save_theme_selections(
    conn: sqlite3.Connection,
    trade_date: str,
    theme_df: pd.DataFrame,
    *,
    market_score: int | None = None,
    filter_keyword: str | None = None,
) -> int:
    """覆盖写入某日题材选股结果，返回写入条数。"""
    ensure_theme_tables(conn)
    td = str(trade_date).strip()[:10]
    kw = str(filter_keyword or "").strip() or None
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("DELETE FROM theme_daily_selections WHERE trade_date = ?", (td,))
    if theme_df is None or theme_df.empty:
        return 0

    n = 0
    for rank, row in enumerate(theme_df.to_dict("records"), start=1):
        code = str(row.get("股票代码", "")).strip().zfill(6)
        if len(code) != 6:
            continue
        name = str(row.get("股票名称", "") or "").strip()
        tag = str(row.get("题材标签", "") or "").strip()
        close_px = _parse_price_cell(row.get("最新价格"))
        vr5 = _parse_vol_ratio_cell(row.get("当前量比"))
        try:
            kdj_j = float(row.get("KDJ_J值"))
        except (TypeError, ValueError):
            kdj_j = None
        try:
            macd_bar = float(row.get("MACD红柱"))
        except (TypeError, ValueError):
            macd_bar = None
        advice = str(row.get("实盘决策建议结论", "") or "").strip()
        conn.execute(
            """
            INSERT INTO theme_daily_selections (
                trade_date, stock_code, stock_name, theme_tag,
                close_price, vol_ratio_5d, kdj_j, macd_bar, advice_text,
                market_score, filter_keyword, rank, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                td,
                code,
                name,
                tag,
                close_px,
                vr5,
                kdj_j,
                macd_bar,
                advice,
                market_score,
                kw,
                rank,
                now,
            ),
        )
        n += 1
    return n


def update_theme_selection_returns(
    conn: sqlite3.Connection,
    trade_date: str,
    stock_code: str,
    *,
    next_day_return: float | None = None,
    hold_5d_return: float | None = None,
    hold_10d_return: float | None = None,
    hold_60d_return: float | None = None,
) -> int:
    sets: list[str] = []
    vals: list[Any] = []
    if next_day_return is not None:
        sets.append("next_day_return = ?")
        vals.append(next_day_return)
    if hold_5d_return is not None:
        sets.append("hold_5d_return = ?")
        vals.append(hold_5d_return)
    if hold_10d_return is not None:
        sets.append("hold_10d_return = ?")
        vals.append(hold_10d_return)
    if hold_60d_return is not None:
        sets.append("hold_60d_return = ?")
        vals.append(hold_60d_return)
    if not sets:
        return 0
    code = str(stock_code).strip().zfill(6)
    vals.extend([str(trade_date).strip()[:10], code])
    return int(
        conn.execute(
            f"UPDATE theme_daily_selections SET {', '.join(sets)} "
            "WHERE trade_date = ? AND stock_code = ?",
            vals,
        ).rowcount
        or 0
    )


def load_theme_selections_df(conn: sqlite3.Connection, trade_date: str) -> pd.DataFrame:
    ensure_theme_tables(conn)
    td = str(trade_date).strip()[:10]
    return pd.read_sql_query(
        """
        SELECT
            rank AS 排名,
            stock_code AS 代码,
            stock_name AS 名称,
            theme_tag AS 题材标签,
            close_price AS 选股日收盘价,
            vol_ratio_5d AS 五日量比,
            kdj_j AS KDJ_J,
            macd_bar AS MACD柱,
            advice_text AS 实盘决策建议结论,
            market_score AS 大盘环境分,
            filter_keyword AS 筛选关键词,
            next_day_return AS ret_1d,
            hold_5d_return AS ret_5d,
            hold_10d_return AS ret_10d,
            hold_60d_return AS ret_60d
        FROM theme_daily_selections
        WHERE trade_date = ?
        ORDER BY COALESCE(rank, 999), stock_code
        """,
        conn,
        params=[td],
    )
