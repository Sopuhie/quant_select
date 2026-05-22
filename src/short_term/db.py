# -*- coding: utf-8 -*-
"""
短线模块 SQLite 表：``short_daily_selections``、``short_order_tracker``。

由 ``src.database.init_db`` 在系统启动时调用 ``ensure_short_term_tables`` 自动创建/升级。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

from .config import SHORT_HOLD_PLAN
from .review_prices import REVIEW_PRICE_KEYS, enrich_rows_with_review_prices, fetch_review_ohlc_map

# ---------------------------------------------------------------------------
# 建表 DDL（与 A_Quant_Lite_Short_Term_Optimization_Guide 对齐）
# ---------------------------------------------------------------------------

# 仅建表（勿在同一段 script 里建依赖新列的索引，旧库会先命中 IF NOT EXISTS 而跳过建表）
SHORT_DAILY_SELECTIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS short_daily_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    close_price REAL,
    pct_change REAL,
    volume_ratio_5d REAL,
    macd_bar_improve REAL,
    j_slope REAL,
    final_score REAL,
    is_executed INTEGER DEFAULT 0,
    rank INTEGER,
    kdj_j REAL,
    macd_bar REAL,
    hold_plan TEXT,
    advice_text TEXT,
    detail_json TEXT,
    t1_open REAL,
    t1_close REAL,
    t2_open REAL,
    t2_close REAL,
    rule_score REAL,
    day_change_pct REAL,
    vol_ratio_5d REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, stock_code)
);
"""

SHORT_ORDER_TRACKER_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS short_order_tracker (
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
    rule_score REAL,
    exit_reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(buy_date, stock_code)
);
"""

# 旧库增量列（先建表再 ALTER，兼容历史库）
_SELECTION_MIGRATE_COLS: tuple[tuple[str, str], ...] = (
    ("pct_change", "REAL"),
    ("volume_ratio_5d", "REAL"),
    ("macd_bar_improve", "REAL"),
    ("j_slope", "REAL"),
    ("final_score", "REAL"),
    ("is_executed", "INTEGER DEFAULT 0"),
    ("rank", "INTEGER"),
    ("kdj_j", "REAL"),
    ("macd_bar", "REAL"),
    ("hold_plan", "TEXT"),
    ("advice_text", "TEXT"),
    ("detail_json", "TEXT"),
    ("t1_open", "REAL"),
    ("t1_close", "REAL"),
    ("t2_open", "REAL"),
    ("t2_close", "REAL"),
    ("rule_score", "REAL"),
    ("day_change_pct", "REAL"),
    ("vol_ratio_5d", "REAL"),
    ("created_at", "TEXT"),
)

_ORDER_MIGRATE_COLS: tuple[tuple[str, str], ...] = (
    ("stock_name", "TEXT"),
    ("sell_date", "TEXT"),
    ("sell_price", "REAL"),
    ("hold_days", "INTEGER DEFAULT 0"),
    ("pnl_ratio", "REAL"),
    ("status", "TEXT DEFAULT 'HOLDING'"),
    ("stop_loss_triggered", "INTEGER DEFAULT 0"),
    ("signal_rank", "INTEGER"),
    ("rule_score", "REAL"),
    ("exit_reason", "TEXT"),
    ("created_at", "TEXT"),
    ("updated_at", "TEXT"),
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})")
    }


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    col: str,
    col_type: str,
    existing: set[str],
) -> None:
    if col not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def _drop_broken_short_selection_indexes(conn: sqlite3.Connection) -> None:
    """
    删除可能由旧版 DDL 在半迁移状态下创建的无效索引。
    （旧脚本在尚无 final_score 列时创建 idx_short_date_score 会导致后续 init 失败。）
    """
    for name in (
        "idx_short_date_score",
        "uidx_short_date_code",
        "idx_short_sel_date_rank",
        "idx_short_sel_date",
    ):
        conn.execute(f"DROP INDEX IF EXISTS {name}")


def _ensure_selection_indexes(conn: sqlite3.Connection) -> None:
    """列迁移完成后再建索引，避免旧库缺少 final_score 时报错。"""
    cols = _table_columns(conn, "short_daily_selections")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_short_sel_date "
        "ON short_daily_selections(trade_date)"
    )
    if "final_score" in cols:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_short_date_score "
            "ON short_daily_selections(trade_date, final_score DESC)"
        )
    elif "rule_score" in cols:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_short_date_score "
            "ON short_daily_selections(trade_date, rule_score DESC)"
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uidx_short_date_code "
        "ON short_daily_selections(trade_date, stock_code)"
    )
    if "rank" in cols:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_short_sel_date_rank "
            "ON short_daily_selections(trade_date, rank)"
        )


def _ensure_order_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_short_order_status "
        "ON short_order_tracker(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_short_order_buy_date "
        "ON short_order_tracker(buy_date)"
    )


def _backfill_selection_legacy_columns(conn: sqlite3.Connection) -> None:
    """将历史列 day_change_pct / rule_score / vol_ratio_5d 回填到新字段。"""
    cols = _table_columns(conn, "short_daily_selections")
    if "day_change_pct" in cols and "pct_change" in cols:
        conn.execute(
            """
            UPDATE short_daily_selections
            SET pct_change = day_change_pct
            WHERE pct_change IS NULL AND day_change_pct IS NOT NULL
            """
        )
    if "vol_ratio_5d" in cols and "volume_ratio_5d" in cols:
        conn.execute(
            """
            UPDATE short_daily_selections
            SET volume_ratio_5d = vol_ratio_5d
            WHERE volume_ratio_5d IS NULL AND vol_ratio_5d IS NOT NULL
            """
        )
    if "rule_score" in cols and "final_score" in cols:
        conn.execute(
            """
            UPDATE short_daily_selections
            SET final_score = rule_score
            WHERE final_score IS NULL AND rule_score IS NOT NULL
            """
        )


def ensure_short_term_tables(conn: sqlite3.Connection) -> None:
    """
    创建/升级短线相关表与索引（幂等，可供 ``init_db`` 与业务层重复调用）。

    顺序：建表 → ALTER 补列 → 回填 → 建索引（旧库必须先补 final_score 再建 score 索引）。
    """
    # 旧库可能残留无效索引，必须先删再建表/补列
    if _table_columns(conn, "short_daily_selections"):
        _drop_broken_short_selection_indexes(conn)

    conn.executescript(SHORT_DAILY_SELECTIONS_TABLE_DDL)
    conn.executescript(SHORT_ORDER_TRACKER_TABLE_DDL)

    sel_cols = _table_columns(conn, "short_daily_selections")
    for col, typ in _SELECTION_MIGRATE_COLS:
        _add_column_if_missing(conn, "short_daily_selections", col, typ, sel_cols)
        sel_cols.add(col)

    order_cols = _table_columns(conn, "short_order_tracker")
    for col, typ in _ORDER_MIGRATE_COLS:
        _add_column_if_missing(conn, "short_order_tracker", col, typ, order_cols)
        order_cols.add(col)

    _backfill_selection_legacy_columns(conn)
    _ensure_selection_indexes(conn)
    _ensure_order_indexes(conn)
    conn.commit()


def _extract_guide_fields(row: dict[str, Any]) -> dict[str, Any]:
    """从策略/执行行解析指南要求的核心字段。"""
    detail = row.get("detail") or {}
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            detail = {}

    execution = detail.get("execution") or {}
    pct = row.get("pct_change")
    if pct is None:
        pct = row.get("day_change_pct")
    vr5 = row.get("volume_ratio_5d")
    if vr5 is None:
        vr5 = row.get("vol_ratio_5d")
    final = row.get("final_score")
    if final is None:
        final = row.get("rule_score")
    macd_imp = detail.get("macd_bar_improve")
    if macd_imp is None and row.get("macd_bar") is not None:
        macd_imp = detail.get("macd_bar_delta")
    j_slope = detail.get("j_slope")
    status = str(execution.get("status") or "").upper()
    is_executed = 1 if status == "CLOSED" else int(row.get("is_executed") or 0)

    return {
        "pct_change": pct,
        "volume_ratio_5d": vr5,
        "macd_bar_improve": macd_imp,
        "j_slope": j_slope,
        "final_score": final,
        "is_executed": is_executed,
        "rule_score": final,
        "day_change_pct": pct,
        "vol_ratio_5d": vr5,
    }


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
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except json.JSONDecodeError:
                detail = {}
        g = _extract_guide_fields({**r, "detail": detail})
        conn.execute(
            """
            INSERT OR REPLACE INTO short_daily_selections (
                trade_date, stock_code, stock_name,
                close_price, pct_change, volume_ratio_5d,
                macd_bar_improve, j_slope, final_score, is_executed,
                rank, kdj_j, macd_bar, hold_plan, advice_text, detail_json,
                t1_open, t1_close, t2_open, t2_close,
                rule_score, day_change_pct, vol_ratio_5d,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                td,
                str(r.get("stock_code", "")).zfill(6),
                r.get("stock_name"),
                r.get("close_price"),
                g["pct_change"],
                g["volume_ratio_5d"],
                g["macd_bar_improve"],
                g["j_slope"],
                g["final_score"],
                g["is_executed"],
                r.get("rank"),
                r.get("kdj_j"),
                r.get("macd_bar"),
                r.get("hold_plan") or SHORT_HOLD_PLAN,
                r.get("advice_text"),
                json.dumps(detail, ensure_ascii=False),
                r.get("t1_open"),
                r.get("t1_close"),
                r.get("t2_open"),
                r.get("t2_close"),
                g["rule_score"],
                g["day_change_pct"],
                g["vol_ratio_5d"],
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
    """用本地 K 线回填/更新 T1/T2 开收盘价（复盘用）。"""
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


def mark_selections_executed_for_buy_date(
    conn: sqlite3.Connection,
    buy_date: str,
    *,
    commit: bool = False,
) -> int:
    """按 ``short_order_tracker`` 中已 CLOSED 的订单，回写 ``is_executed=1``。"""
    ensure_short_term_tables(conn)
    td = str(buy_date).strip()[:10]
    cur = conn.execute(
        """
        UPDATE short_daily_selections
        SET is_executed = 1
        WHERE trade_date = ?
          AND stock_code IN (
              SELECT stock_code FROM short_order_tracker
              WHERE buy_date = ? AND status = 'CLOSED'
          )
        """,
        (td, td),
    )
    n = cur.rowcount
    if commit:
        conn.commit()
    return int(n)


def load_short_selections_df(
    conn: sqlite3.Connection,
    trade_date: str,
) -> pd.DataFrame:
    ensure_short_term_tables(conn)
    return pd.read_sql_query(
        """
        SELECT trade_date AS 信号日, stock_code AS 股票代码, stock_name AS 股票名称,
               rank AS 排名,
               COALESCE(final_score, rule_score) AS 规则得分,
               COALESCE(pct_change, day_change_pct) AS 日涨幅,
               COALESCE(volume_ratio_5d, vol_ratio_5d) AS 五日量比,
               macd_bar_improve AS MACD柱改善, j_slope AS J斜率,
               is_executed AS 已执行,
               kdj_j AS KDJ_J, macd_bar AS MACD柱,
               close_price AS 信号日收盘价,
               t1_open AS T1开盘价, t1_close AS T1收盘价,
               t2_open AS T2开盘价, t2_close AS T2收盘价,
               CASE
                 WHEN t2_close IS NOT NULL
                  AND COALESCE(t1_open, t1_close) IS NOT NULL
                  AND COALESCE(t1_open, t1_close) > 0
                 THEN (t2_close - COALESCE(t1_open, t1_close))
                      / COALESCE(t1_open, t1_close)
                 ELSE NULL
               END AS T1买T2卖涨跌幅,
               hold_plan AS 持仓计划, advice_text AS 实盘建议
        FROM short_daily_selections
        WHERE trade_date = ?
        ORDER BY COALESCE(final_score, rule_score) DESC, rank ASC
        """,
        conn,
        params=[str(trade_date).strip()[:10]],
    )


def delete_short_orders_for_buy_date(
    conn: sqlite3.Connection,
    buy_date: str,
    *,
    commit: bool = True,
) -> None:
    ensure_short_term_tables(conn)
    conn.execute(
        "DELETE FROM short_order_tracker WHERE buy_date = ?",
        (str(buy_date).strip()[:10],),
    )
    if commit:
        conn.commit()


def upsert_short_order(conn: sqlite3.Connection, order: dict[str, Any]) -> None:
    """写入或更新一条模拟订单。"""
    ensure_short_term_tables(conn)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO short_order_tracker (
            stock_code, stock_name, buy_date, buy_price,
            sell_date, sell_price, hold_days, pnl_ratio,
            status, stop_loss_triggered, signal_rank, rule_score,
            exit_reason, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            rule_score = excluded.rule_score,
            exit_reason = excluded.exit_reason,
            updated_at = excluded.updated_at
        """,
        (
            str(order.get("stock_code", "")).zfill(6),
            order.get("stock_name"),
            str(order.get("buy_date", "")).strip()[:10],
            order.get("buy_price"),
            order.get("sell_date"),
            order.get("sell_price"),
            order.get("hold_days"),
            order.get("pnl_ratio"),
            order.get("status"),
            int(order.get("stop_loss_triggered") or 0),
            order.get("rank") or order.get("signal_rank"),
            order.get("rule_score") or order.get("final_score"),
            order.get("exit_reason"),
            now,
            now,
        ),
    )


def load_short_orders_for_buy_date(
    conn: sqlite3.Connection,
    buy_date: str,
) -> list[dict[str, Any]]:
    ensure_short_term_tables(conn)
    cur = conn.execute(
        """
        SELECT order_id, stock_code, stock_name, buy_date, buy_price,
               sell_date, sell_price, hold_days, pnl_ratio, status,
               stop_loss_triggered, signal_rank, rule_score, exit_reason
        FROM short_order_tracker
        WHERE buy_date = ?
        ORDER BY signal_rank ASC, stock_code ASC
        """,
        (str(buy_date).strip()[:10],),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
