# -*- coding: utf-8 -*-
"""SQLite 数据库：建表、写入选股与预测、模型版本。"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import time

import numpy as np
import pandas as pd
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .config import DB_PATH

# 高并发读写时延长连接等待、WAL、busy_timeout，降低 "database is locked" 概率
_SQLITE_CONNECT_TIMEOUT = 30.0


def _sqlite_write_backoff_sleep(attempt: int) -> None:
    """指数退避 + 小幅随机抖动，缓解与后台线程并发写库时的锁竞争。"""
    base = 0.1 * (2 ** max(0, int(attempt)))
    time.sleep(base + random.uniform(0, 0.05))


def _retry_sqlite_locked(
    op: Callable[[], None],
    *,
    attempts: int = 5,
) -> None:
    last: sqlite3.OperationalError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            op()
            return
        except sqlite3.OperationalError as e:
            last = e
            msg = str(e).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            if attempt >= int(attempts) - 1:
                raise
            _sqlite_write_backoff_sleep(attempt)
    if last is not None:
        raise last


def _apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-64000;")
    conn.execute("PRAGMA mmap_size=268435456;")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS daily_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    rank INTEGER,
    score REAL,
    close_price REAL,
    next_day_return REAL,
    hold_5d_return REAL,
    hold_10d_return REAL,
    hold_60d_return REAL,
    selection_reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, stock_code)
);

CREATE TABLE IF NOT EXISTS daily_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    score REAL,
    rank_in_market INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, stock_code)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    train_end_date TEXT,
    features TEXT,
    metrics TEXT,
    is_active INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_selections(trade_date);
CREATE INDEX IF NOT EXISTS idx_predict_date ON daily_predictions(trade_date);
CREATE INDEX IF NOT EXISTS idx_predict_rank ON daily_predictions(rank_in_market);
-- 跑批 / Streamlit 高频查询联合索引（任务 4）
CREATE INDEX IF NOT EXISTS idx_daily_sel_date_rank ON daily_selections(trade_date, rank);
CREATE INDEX IF NOT EXISTS idx_predict_date_code ON daily_predictions(trade_date, stock_code);

CREATE TABLE IF NOT EXISTS stock_daily_kline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    industry TEXT,
    market_cap REAL,
    turnover_rate REAL,
    pe_ttm REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_kline_code_date ON stock_daily_kline(stock_code, date);
CREATE INDEX IF NOT EXISTS idx_kline_date ON stock_daily_kline(date);

CREATE TABLE IF NOT EXISTS stock_financial_data (
    stock_code TEXT NOT NULL,
    pub_date TEXT NOT NULL,
    report_date TEXT NOT NULL,
    roe REAL,
    net_profit_growth REAL,
    revenue_growth REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code, report_date)
);
CREATE INDEX IF NOT EXISTS idx_financial_code_pub ON stock_financial_data(stock_code, pub_date);

CREATE TABLE IF NOT EXISTS stock_money_flow_daily (
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    big_net_ratio REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_mf_date ON stock_money_flow_daily(trade_date);

CREATE TABLE IF NOT EXISTS stock_north_hold_daily (
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    hold_pct REAL,
    hold_pct_chg REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_nh_date ON stock_north_hold_daily(trade_date);

CREATE TABLE IF NOT EXISTS market_hsgt_flow_daily (
    trade_date TEXT NOT NULL PRIMARY KEY,
    net_inflow REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    status TEXT NOT NULL,
    run_time TEXT NOT NULL,
    parameters TEXT,
    log_output TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_logs_task_name ON system_logs(task_name);
CREATE INDEX IF NOT EXISTS idx_logs_created ON system_logs(created_at);

CREATE TABLE IF NOT EXISTS signal_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    signal_time TEXT NOT NULL,
    signal_price REAL,
    signal_type TEXT NOT NULL,
    reason TEXT,
    realtime_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_signal_code_time ON signal_history(stock_code, signal_time);
CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_dedup_minute
    ON signal_history(stock_code, signal_time, signal_type);

CREATE TABLE IF NOT EXISTS stock_concept_boards (
    stock_code TEXT NOT NULL,
    board_name TEXT NOT NULL,
    updated_date TEXT NOT NULL,
    PRIMARY KEY (stock_code, board_name)
);
CREATE INDEX IF NOT EXISTS idx_scb_board ON stock_concept_boards(board_name);
CREATE INDEX IF NOT EXISTS idx_scb_date ON stock_concept_boards(updated_date);
"""


def _ensure_stock_daily_kline_industry(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_daily_kline'")
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(stock_daily_kline)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "industry" not in cols:
        conn.execute("ALTER TABLE stock_daily_kline ADD COLUMN industry TEXT")


def _ensure_daily_selections_selection_reason(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_selections'")
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(daily_selections)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "selection_reason" not in cols:
        try:
            conn.execute("ALTER TABLE daily_selections ADD COLUMN selection_reason TEXT")
        except Exception:
            pass


def _ensure_daily_selections_hold_returns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_selections'")
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(daily_selections)")
    cols = {str(row[1]) for row in cur.fetchall()}
    for col in ("hold_10d_return", "hold_60d_return"):
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE daily_selections ADD COLUMN {col} REAL")
            except Exception:
                pass


def _ensure_stock_daily_kline_market_cap(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_daily_kline'")
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(stock_daily_kline)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "market_cap" not in cols:
        conn.execute("ALTER TABLE stock_daily_kline ADD COLUMN market_cap REAL")


def _ensure_stock_daily_kline_turnover_pe(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_daily_kline'")
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(stock_daily_kline)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "turnover_rate" not in cols:

        def _add_tr() -> None:
            conn.execute("ALTER TABLE stock_daily_kline ADD COLUMN turnover_rate REAL")

        _retry_sqlite_locked(_add_tr, attempts=8)
        cols.add("turnover_rate")
    if "pe_ttm" not in cols:

        def _add_pe() -> None:
            conn.execute("ALTER TABLE stock_daily_kline ADD COLUMN pe_ttm REAL")

        _retry_sqlite_locked(_add_pe, attempts=8)


def _ensure_market_hsgt_flow_daily(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_hsgt_flow_daily'")
    if cur.fetchone() is None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_hsgt_flow_daily (
                trade_date TEXT NOT NULL PRIMARY KEY,
                net_inflow REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def _ensure_performance_indexes(conn: sqlite3.Connection) -> None:
    """
    旧库升级：补全 SCHEMA 之后新增的联合索引与 K 线 (code,date) 索引。
    ``init_db`` 建库与每次连接前均会调用，避免仅依赖 executescript 时漏建。
    """
    for ddl in (
        "CREATE INDEX IF NOT EXISTS idx_daily_sel_date_rank ON daily_selections(trade_date, rank)",
        "CREATE INDEX IF NOT EXISTS idx_predict_date_code ON daily_predictions(trade_date, stock_code)",
        "CREATE INDEX IF NOT EXISTS idx_kline_code_date ON stock_daily_kline(stock_code, date)",
        "CREATE INDEX IF NOT EXISTS idx_kline_date ON stock_daily_kline(date)",
        "CREATE INDEX IF NOT EXISTS idx_mf_code_date ON stock_money_flow_daily(stock_code, trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_nh_code_date ON stock_north_hold_daily(stock_code, trade_date)",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass


def _ensure_stock_concept_boards(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_concept_boards'")
    if cur.fetchone() is None:
        conn.execute("CREATE TABLE IF NOT EXISTS stock_concept_boards (stock_code TEXT NOT NULL, board_name TEXT NOT NULL, updated_date TEXT NOT NULL, PRIMARY KEY (stock_code, board_name))")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scb_board ON stock_concept_boards(board_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scb_date ON stock_concept_boards(updated_date)")


def _ensure_short_term_tables(conn: sqlite3.Connection) -> None:
    """短线选股与模拟订单表（``short_daily_selections`` / ``short_order_tracker``）。"""
    from src.short_term.db import ensure_short_term_tables as _ensure_short

    _ensure_short(conn)


def _ensure_theme_tables(conn: sqlite3.Connection) -> None:
    """热门题材选股历史表（``theme_daily_selections``）。"""
    from src.theme.db import ensure_theme_tables as _ensure_theme

    _ensure_theme(conn)


def init_db(db_path: Path | None = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    try:
        _apply_sqlite_pragmas(conn)
        conn.executescript(SCHEMA_SQL)
        _ensure_stock_daily_kline_industry(conn)
        _ensure_stock_daily_kline_market_cap(conn)
        _ensure_stock_daily_kline_turnover_pe(conn)
        _ensure_daily_selections_selection_reason(conn)
        _ensure_daily_selections_hold_returns(conn)
        _ensure_stock_concept_boards(conn)
        _ensure_market_hsgt_flow_daily(conn)
        _ensure_performance_indexes(conn)
        _ensure_short_term_tables(conn)
        _ensure_theme_tables(conn)
        conn.commit()
    finally:
        conn.close()


def open_sqlite_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    return conn


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_stock_daily_klines(
    rows: Iterable[dict[str, Any]],
    db_path: Path | None = None,
    *,
    connection: sqlite3.Connection | None = None,
) -> None:
    sql = """
    INSERT INTO stock_daily_kline (
        date, stock_code, stock_name, industry, market_cap,
        turnover_rate, pe_ttm,
        open, high, low, close, volume
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(date, stock_code) DO UPDATE SET
        stock_name = excluded.stock_name,
        industry = COALESCE(NULLIF(TRIM(excluded.industry), ''), stock_daily_kline.industry),
        market_cap = COALESCE(excluded.market_cap, stock_daily_kline.market_cap),
        turnover_rate = COALESCE(excluded.turnover_rate, stock_daily_kline.turnover_rate),
        pe_ttm = COALESCE(excluded.pe_ttm, stock_daily_kline.pe_ttm),
        open = excluded.open,
        high = excluded.high,
        low = excluded.low,
        close = excluded.close,
        volume = excluded.volume
    """
    batch: list[tuple[Any, ...]] = []
    for r in rows:
        ind_raw = r.get("industry")
        industry_val = str(ind_raw).strip() if ind_raw is not None and str(ind_raw).strip() else ""
        mc_raw = r.get("market_cap")
        try:
            market_cap_val = float(mc_raw) if mc_raw is not None and pd.notna(mc_raw) else None
        except (TypeError, ValueError):
            market_cap_val = None

        def _opt_float(key: str) -> float | None:
            raw = r.get(key)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                return None
            try:
                v = float(raw)
                return v if math.isfinite(v) else None
            except (TypeError, ValueError):
                return None

        batch.append(
            (
                r["date"],
                str(r["stock_code"]).strip().zfill(6),
                str(r.get("stock_name") or "").strip(),
                industry_val,
                market_cap_val,
                _opt_float("turnover_rate"),
                _opt_float("pe_ttm"),
                float(r["open"]) if r.get("open") is not None else None,
                float(r["high"]) if r.get("high") is not None else None,
                float(r["low"]) if r.get("low") is not None else None,
                float(r["close"]) if r.get("close") is not None else None,
                float(r["volume"]) if r.get("volume") is not None else None,
            )
        )
    if not batch:
        return

    own = connection is None
    conn = connection if connection is not None else sqlite3.connect(str(db_path or DB_PATH), timeout=_SQLITE_CONNECT_TIMEOUT)
    if own:
        _apply_sqlite_pragmas(conn)

    # 深度重构: 分块写入结合微休眠，彻底释放高并发下的 SQLite 写锁，防止实盘信号漏标
    chunk_size = 2000
    try:
        for i in range(0, len(batch), chunk_size):
            sub_chunk = batch[i : i + chunk_size]

            def _execute_chunk(chunk_data=sub_chunk):
                conn.executemany(sql, chunk_data)

            _retry_sqlite_locked(_execute_chunk, attempts=5)
            if own:
                conn.commit()
                time.sleep(0.01)
    finally:
        if own:
            conn.close()


def insert_daily_selections(
    rows: Iterable[dict[str, Any]],
    db_path: Path | None = None,
) -> None:
    sql = """
    INSERT INTO daily_selections
    (trade_date, stock_code, stock_name, rank, score, close_price,
     next_day_return, hold_5d_return, selection_reason, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(trade_date, stock_code) DO UPDATE SET
        stock_name = excluded.stock_name,
        rank = excluded.rank,
        score = excluded.score,
        close_price = excluded.close_price,
        next_day_return = coalesce(next_day_return, excluded.next_day_return),
        hold_5d_return = coalesce(hold_5d_return, excluded.hold_5d_return),
        selection_reason = excluded.selection_reason
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    batch = []
    for r in rows:
        batch.append(
            (
                r["trade_date"],
                r["stock_code"],
                r.get("stock_name"),
                r["rank"],
                r["score"],
                r.get("close_price"),
                r.get("next_day_return"),
                r.get("hold_5d_return"),
                r.get("selection_reason"),
                r.get("created_at", now),
            )
        )
    if not batch:
        return

    def _do() -> None:
        with get_connection(db_path) as conn:
            conn.executemany(sql, batch)

    _retry_sqlite_locked(_do, attempts=5)


def bulk_set_stock_daily_industry_by_code(
    stock_code_to_industry: dict[str, str],
    db_path: Path | None = None,
) -> int:
    """
    按 6 位 ``stock_code`` 批量更新 ``stock_daily_kline.industry``（该代码下所有历史行）。

    Returns:
        ``sqlite3`` 连接上累计的 ``total_changes`` 增量（约等于被更新的行数）。
    """
    if not stock_code_to_industry:
        return 0
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    try:
        before = conn.total_changes
        payload: list[tuple[str, str]] = []
        for code, ind in stock_code_to_industry.items():
            c = str(code).strip().zfill(6)
            if len(c) == 6 and c.isdigit():
                payload.append((str(ind).strip(), c))
        conn.executemany(
            "UPDATE stock_daily_kline SET industry = ? WHERE stock_code = ?",
            payload,
        )
        conn.commit()
        return int(conn.total_changes - before)
    finally:
        conn.close()




def insert_daily_predictions(rows: Iterable[dict[str, Any]], db_path: Path | None = None) -> None:
    sql = """
    INSERT OR REPLACE INTO daily_predictions
    (trade_date, stock_code, stock_name, score, rank_in_market, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    batch = []
    for r in rows:
        batch.append((r["trade_date"], r["stock_code"], r.get("stock_name"), r["score"], r["rank_in_market"], r.get("created_at", now)))
    with get_connection(db_path) as conn:
        conn.executemany(sql, batch)


def insert_system_log(task_name: str, status: str, parameters: str | None, log_output: str | None, *, db_path: Path | None = None) -> None:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = (str(task_name).strip(), str(status).strip(), run_time, parameters if parameters is not None else "", log_output if log_output is not None else "")
    def _do() -> None:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO system_logs (task_name, status, run_time, parameters, log_output) VALUES (?, ?, ?, ?, ?)",
                params,
            )

    _retry_sqlite_locked(_do, attempts=5)


def register_model_version(version: str, train_end_date: str, features: list[str], metrics: dict[str, Any], set_active: bool = True, db_path: Path | None = None) -> None:
    with get_connection(db_path) as conn:
        if set_active:
            conn.execute("UPDATE model_versions SET is_active = 0")
        conn.execute("""
            INSERT INTO model_versions (version, train_end_date, features, metrics, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (version, train_end_date, json.dumps(features, ensure_ascii=False), json.dumps(metrics, ensure_ascii=False), 1 if set_active else 0, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))


def get_active_model_version(db_path: Path | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT * FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("features"): d["features"] = json.loads(d["features"])
    if d.get("metrics"): d["metrics"] = json.loads(d["metrics"])
    return d


def query_df(sql: str, params: tuple[Any, ...] = (), db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def fetch_selection_rows_for_date(trade_date: str, limit: int | None = None, db_path: Path | None = None) -> list[dict[str, Any]]:
    from .config import TOP_N_SELECTION
    lim = int(limit if limit is not None else TOP_N_SELECTION)
    df = query_df("SELECT rank, stock_code, stock_name, score, close_price, selection_reason FROM daily_selections WHERE trade_date = ? ORDER BY rank ASC LIMIT ?", (trade_date, lim), db_path)
    return df.to_dict("records")


def fetch_selection_rows_for_dingtalk_push(trade_date: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    from .config import TOP_N_SELECTION
    df = query_df("SELECT rank, stock_code, stock_name, score, close_price, selection_reason FROM daily_selections WHERE trade_date = ? ORDER BY rank ASC", (trade_date,), db_path)
    if df.empty: return []
    out = df[df["rank"].isin(range(1, TOP_N_SELECTION + 1))].copy()
    if out.empty: return []
    out = out.sort_values("score", ascending=False).drop_duplicates(subset=["rank"], keep="first")
    out = out.sort_values("rank").reset_index(drop=True).head(TOP_N_SELECTION)
    rows: list[dict[str, Any]] = []
    for _, r in out.iterrows():
        rows.append({
            "rank": int(r["rank"]),
            "stock_code": str(r["stock_code"]).strip(),
            "stock_name": str(r.get("stock_name") or "").strip(),
            "score": float(r["score"]) if pd.notna(r.get("score")) else None,
            "close_price": float(r["close_price"]) if pd.notna(r.get("close_price")) else None,
            "selection_reason": str(r.get("selection_reason") or "").strip()
        })
    return rows


def selection_exists_for_date(trade_date: str, db_path: Path | None = None) -> bool:
    with get_connection(db_path) as conn:
        return conn.execute("SELECT 1 FROM daily_selections WHERE trade_date = ? LIMIT 1", (trade_date,)).fetchone() is not None


def delete_daily_outputs_for_trade_date(trade_date: str, db_path: Path | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM daily_selections WHERE trade_date = ?", (trade_date,))
        conn.execute("DELETE FROM daily_predictions WHERE trade_date = ?", (trade_date,))


def update_selection_returns(
    trade_date: str,
    stock_code: str,
    next_day_return: float | None = None,
    hold_5d_return: float | None = None,
    hold_10d_return: float | None = None,
    hold_60d_return: float | None = None,
    db_path: Path | None = None,
) -> int:
    sets, vals = [], []
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
    raw = str(stock_code).strip()
    code_key = raw.zfill(6) if raw.isdigit() else raw
    if code_key == raw:
        vals.extend([trade_date, code_key])
        where = "trade_date = ? AND trim(replace(stock_code, ' ', '')) = ?"
    else:
        vals.extend([trade_date, code_key, raw])
        where = "trade_date = ? AND (trim(replace(stock_code, ' ', '')) = ? OR trim(replace(stock_code, ' ', '')) = ?)"
    with get_connection(db_path) as conn:
        return int(conn.execute(f"UPDATE daily_selections SET {', '.join(sets)} WHERE {where}", vals).rowcount or 0)


def fetch_stock_daily_bars_until(
    stock_code: str,
    end_date: str,
    *,
    db_path: Path | None = None,
    connection: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    code, end = str(stock_code).strip().zfill(6), str(end_date).strip()[:10]

    def _query(conn: sqlite3.Connection) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume, industry, market_cap, "
            "turnover_rate, pe_ttm FROM stock_daily_kline WHERE stock_code = ? AND date <= ? "
            "ORDER BY date ASC",
            conn,
            params=(code, end),
        )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close", "volume", "market_cap", "turnover_rate", "pe_ttm"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "industry" in df.columns:
            df["industry"] = df["industry"].fillna("").astype(str)
        return df.dropna(subset=["close"]).reset_index(drop=True)

    if connection is not None:
        return _query(connection)
    with get_connection(db_path or DB_PATH) as conn:
        return _query(conn)


def fetch_latest_industry_by_codes(stock_codes: Iterable[str], *, db_path: Path | None = None) -> dict[str, str]:
    uniq = list(set([str(c).strip().zfill(6) for c in stock_codes if len(str(c).strip().zfill(6)) == 6]))
    if not uniq: return {}
    out: dict[str, str] = {}
    chunk_size = 400
    with get_connection(db_path) as conn:
        for i in range(0, len(uniq), chunk_size):
            chunk = uniq[i : i + chunk_size]
            cur = conn.execute(f"SELECT k.stock_code, k.industry FROM stock_daily_kline k INNER JOIN (SELECT stock_code, MAX(date) AS mx FROM stock_daily_kline WHERE stock_code IN ({','.join('?'*len(chunk))}) GROUP BY stock_code) t ON k.stock_code = t.stock_code AND k.date = t.mx", chunk)
            for row in cur.fetchall(): out[str(row[0]).strip().zfill(6)] = "" if row[1] is None else str(row[1]).strip()
    return out


def fetch_latest_market_cap_by_codes(stock_codes: Iterable[str], *, db_path: Path | None = None) -> dict[str, float]:
    uniq = list(set([str(c).strip().zfill(6) for c in stock_codes if len(str(c).strip().zfill(6)) == 6]))
    if not uniq: return {}
    out: dict[str, float] = {}
    chunk_size = 400
    with get_connection(db_path) as conn:
        for i in range(0, len(uniq), chunk_size):
            chunk = uniq[i : i + chunk_size]
            cur = conn.execute(f"SELECT k.stock_code, k.market_cap FROM stock_daily_kline k INNER JOIN (SELECT stock_code, MAX(date) AS mx FROM stock_daily_kline WHERE stock_code IN ({','.join('?'*len(chunk))}) GROUP BY stock_code) t ON k.stock_code = t.stock_code AND k.date = t.mx", chunk)
            for row in cur.fetchall():
                if row[1] is not None:
                    try: out[str(row[0]).strip().zfill(6)] = float(row[1])
                    except ValueError: pass
    return out


def list_predict_universe_from_kline(trade_date: str, *, min_bars: int, max_count: int | None, db_path: Path | None = None) -> list[tuple[str, str]]:
    end, mb = str(trade_date).strip()[:10], max(1, int(min_bars))
    sql_body = "WITH eligible AS (SELECT stock_code FROM stock_daily_kline WHERE date <= ? GROUP BY stock_code HAVING COUNT(*) >= ?), latest AS (SELECT stock_code, MAX(date) AS mx FROM stock_daily_kline WHERE date <= ? GROUP BY stock_code) SELECT k.stock_code, k.stock_name FROM stock_daily_kline k INNER JOIN eligible e ON k.stock_code = e.stock_code INNER JOIN latest t ON k.stock_code = t.stock_code AND k.date = t.mx ORDER BY k.stock_code"
    use_limit = max_count is not None and int(max_count) > 0
    sql = sql_body + (" LIMIT ?" if use_limit else "")
    params = (end, mb, end, int(max_count)) if use_limit else (end, mb, end)
    with get_connection(db_path) as conn:
        raw = pd.read_sql_query(sql, conn, params=params)
    return [(str(row["stock_code"]).strip().zfill(6), str(row.get("stock_name") or "").strip()) for _, row in raw.iterrows()]


def stock_codes_with_local_bars(trade_date: str, min_bars: int, *, db_path: Path | None = None) -> set[str]:
    end, mb = str(trade_date).strip()[:10], max(1, int(min_bars))
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT stock_code FROM stock_daily_kline WHERE date <= ? GROUP BY stock_code HAVING COUNT(*) >= ?", (end, mb)).fetchall()
    return {str(r[0]).strip().zfill(6) for r in rows}


def fetch_market_hsgt_net_flow_map(dates: list[str], *, db_path: Path | None = None) -> dict[str, float]:
    ds = sorted({str(d).strip()[:10] for d in dates if d})
    if not ds: return {}
    with get_connection(db_path) as conn:
        cur = conn.execute(f"SELECT trade_date, net_inflow FROM market_hsgt_flow_daily WHERE trade_date IN ({','.join('?'*len(ds))}) AND net_inflow IS NOT NULL", tuple(ds))
        return {str(r[0]).strip()[:10]: float(r[1]) for r in cur.fetchall() if math.isfinite(float(r[1]))}


def fetch_market_hsgt_net_flow_up_to(end_date: str, *, db_path: Path | None = None) -> dict[str, float]:
    ed = str(end_date).strip()[:10]
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT trade_date, net_inflow FROM market_hsgt_flow_daily WHERE trade_date <= ? AND net_inflow IS NOT NULL ORDER BY trade_date ASC", (ed,))
        return {str(r[0]).strip()[:10]: float(r[1]) for r in cur.fetchall() if math.isfinite(float(r[1]))}


def upsert_market_hsgt_flow_rows(rows: Iterable[dict[str, Any]], *, db_path: Path | None = None) -> int:
    batch = [(str(r["trade_date"]).strip()[:10], float(r["net_inflow"])) for r in rows if r.get("trade_date") and r.get("net_inflow") is not None]
    if not batch: return 0
    def _do() -> None:
        with get_connection(db_path) as conn:
            conn.executemany("INSERT INTO market_hsgt_flow_daily (trade_date, net_inflow, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(trade_date) DO UPDATE SET net_inflow = excluded.net_inflow, updated_at = CURRENT_TIMESTAMP", batch)
    _retry_sqlite_locked(_do, attempts=5)
    return len(batch)


def fetch_auxiliary_feature_frames(dates: list[str], codes: list[str], *, db_path: Path | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not dates or not codes: return None, None
    ds, cs = [str(d).strip()[:10] for d in dates], [str(c).strip().zfill(6) for c in codes]
    params = tuple(ds + cs)
    with get_connection(db_path) as conn:
        mf = pd.read_sql_query(f"SELECT trade_date AS _k_date, stock_code AS _k_code, big_net_ratio FROM stock_money_flow_daily WHERE trade_date IN ({','.join('?'*len(ds))}) AND stock_code IN ({','.join('?'*len(cs))})", conn, params=params)
        nh = pd.read_sql_query(f"SELECT trade_date AS _k_date, stock_code AS _k_code, hold_pct_chg, hold_pct FROM stock_north_hold_daily WHERE trade_date IN ({','.join('?'*len(ds))}) AND stock_code IN ({','.join('?'*len(cs))})", conn, params=params)
    if not mf.empty:
        mf["_k_code"] = mf["_k_code"].astype(str).str.zfill(6)
        mf["_k_date"] = mf["_k_date"].astype(str).str[:10]
    else: mf = None
    if not nh.empty:
        nh["_k_code"] = nh["_k_code"].astype(str).str.zfill(6)
        nh["_k_date"] = nh["_k_date"].astype(str).str[:10]
    else: nh = None
    return mf, nh


def fetch_existing_auxiliary_key_sets(dates: list[str], codes: list[str], *, db_path: Path | None = None) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    mf_keys, nh_keys = set(), set()
    if not dates or not codes: return mf_keys, nh_keys
    ds, cs = [str(d).strip()[:10] for d in dates], [str(c).strip().zfill(6) for c in codes]
    params = tuple(ds + cs)
    with get_connection(db_path) as conn:
        cur1 = conn.execute(f"SELECT trade_date, stock_code FROM stock_money_flow_daily WHERE trade_date IN ({','.join('?'*len(ds))}) AND stock_code IN ({','.join('?'*len(cs))})", params)
        for r in cur1.fetchall(): mf_keys.add((str(r[0])[:10], str(r[1]).zfill(6)))
        cur2 = conn.execute(f"SELECT trade_date, stock_code FROM stock_north_hold_daily WHERE trade_date IN ({','.join('?'*len(ds))}) AND stock_code IN ({','.join('?'*len(cs))})", params)
        for r in cur2.fetchall(): nh_keys.add((str(r[0])[:10], str(r[1]).zfill(6)))
    return mf_keys, nh_keys


def upsert_stock_money_flow_rows(rows: list[tuple[str, str, float]], *, db_path: Path | None = None) -> None:
    if not rows: return
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    def _do() -> None:
        with get_connection(db_path) as conn:
            conn.executemany("INSERT INTO stock_money_flow_daily (trade_date, stock_code, big_net_ratio, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(trade_date, stock_code) DO UPDATE SET big_net_ratio=excluded.big_net_ratio, updated_at=excluded.updated_at", [(a[0], a[1], float(a[2]), now) for a in rows])
    _retry_sqlite_locked(_do, attempts=5)


def fetch_top3_selections_for_monitor(trade_date: str | None = None, db_path: Path | None = None) -> list[dict[str, Any]]:
    from .config import TOP_N_SELECTION
    if trade_date: td = str(trade_date).strip()[:10]
    else:
        with get_connection(db_path) as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM daily_selections").fetchone()
            if row is None or row[0] is None: return []
            td = str(row[0]).strip()[:10]
    df = query_df("SELECT rank, stock_code, stock_name, score, close_price FROM daily_selections WHERE trade_date = ? AND rank BETWEEN 1 AND ? ORDER BY rank ASC", (td, int(TOP_N_SELECTION)), db_path)
    return df.to_dict("records")


def insert_signal_record(*, stock_code: str, stock_name: str | None, signal_time: str, signal_price: float, signal_type: str, reason: str | None = None, realtime_score: float | None = None, db_path: Path | None = None) -> None:
    params = (str(stock_code).strip().zfill(6), stock_name, str(signal_time).strip()[:19], float(signal_price), str(signal_type).strip(), reason, float(realtime_score) if realtime_score is not None else None, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    def _do() -> None:
        with get_connection(db_path) as conn:
            conn.execute("INSERT OR IGNORE INTO signal_history (stock_code, stock_name, signal_time, signal_price, signal_type, reason, realtime_score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", params)
    _retry_sqlite_locked(_do, attempts=5)


def signal_exists_in_minute(stock_code: str, signal_time_minute: str, signal_type: str, db_path: Path | None = None) -> bool:
    with get_connection(db_path) as conn:
        return conn.execute("SELECT 1 FROM signal_history WHERE stock_code = ? AND substr(signal_time, 1, 16) = ? AND signal_type = ? LIMIT 1", (str(stock_code).strip().zfill(6), str(signal_time_minute).strip()[:16], str(signal_type).strip())).fetchone() is not None


def signal_exists_within_minutes(stock_code: str, signal_time: str, signal_type: str, window_minutes: int, *, db_path: Path | None = None) -> bool:
    if window_minutes <= 0: return False
    with get_connection(db_path) as conn:
        return conn.execute("SELECT 1 FROM signal_history WHERE stock_code = ? AND signal_type = ? AND datetime(signal_time) > datetime(?, ?) AND datetime(signal_time) <= datetime(?) LIMIT 1", (str(stock_code).strip().zfill(6), str(signal_type).strip(), str(signal_time).strip()[:19], f"-{max(1, int(window_minutes))} minutes", str(signal_time).strip()[:19])).fetchone() is not None


def fetch_signal_history_for_stock_on_date(stock_code: str, trade_date: str | None = None, db_path: Path | None = None) -> list[dict[str, Any]]:
    day = str(trade_date).strip()[:10] if trade_date else datetime.now().strftime("%Y-%m-%d")
    df = query_df("SELECT signal_time, signal_price, signal_type, reason, realtime_score FROM signal_history WHERE stock_code = ? AND signal_time LIKE ? ORDER BY signal_time ASC", (str(stock_code).strip().zfill(6), f"{day}%"), db_path)
    return df.to_dict("records")


def upsert_stock_north_hold_rows(rows: list[tuple[str, str, float, float]], *, db_path: Path | None = None) -> None:
    if not rows: return
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    def _do() -> None:
        with get_connection(db_path) as conn:
            conn.executemany("INSERT INTO stock_north_hold_daily (trade_date, stock_code, hold_pct, hold_pct_chg, updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(trade_date, stock_code) DO UPDATE SET hold_pct=excluded.hold_pct, hold_pct_chg=excluded.hold_pct_chg, updated_at=excluded.updated_at", [(a[0], a[1], float(a[2]), float(a[3]), now) for a in rows])
    _retry_sqlite_locked(_do, attempts=5)


def sync_concept_boards_from_json(db_path=None) -> int:
    from .config import DATA_DIR
    json_path = DATA_DIR / "board_stocks.json"
    if not json_path.exists(): return 0
    with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
    stb = data.get("stock_to_boards", {})
    if not stb: return 0
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    today, inserted = datetime.now().strftime("%Y-%m-%d"), 0
    try:
        conn.execute("BEGIN")
        for code, boards in stb.items():
            for board in boards:
                conn.execute("INSERT OR REPLACE INTO stock_concept_boards (stock_code, board_name, updated_date) VALUES (?, ?, ?)", (str(code).strip().zfill(6), str(board).strip(), today))
                inserted += 1
        conn.commit()
    finally: conn.close()
    return inserted


def fetch_concept_boards_by_stock(stock_code: str, db_path=None) -> list[str]:
    with get_connection(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT DISTINCT board_name FROM stock_concept_boards WHERE stock_code = ?", (str(stock_code).zfill(6),)).fetchall()]


def fetch_stocks_by_concept_board(board_name: str, db_path=None) -> list[str]:
    with get_connection(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_concept_boards WHERE board_name = ?", (str(board_name),)).fetchall()]


def fetch_joined_fundamental_moneyflow_panel(
    trade_date: str,
    codes: list[str],
    *,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """
    截面左连接：``stock_daily_kline``（锚定日）+ 最新 ``pub_date<=trade_date`` 财报
    + 当日 ``stock_money_flow_daily``。
    """
    td = str(trade_date).strip()[:10]
    uniq = sorted({str(c).strip().zfill(6) for c in codes if len(str(c).strip().zfill(6)) == 6})
    if not uniq:
        return pd.DataFrame(
            columns=[
                "date",
                "stock_code",
                "roe",
                "net_profit_growth",
                "revenue_growth",
                "big_net_ratio",
                "turnover_rate",
                "pe_ttm",
            ]
        )
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    try:
        cur_cols = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(stock_daily_kline)").fetchall()
        }
        tr_sel = (
            "k.turnover_rate"
            if "turnover_rate" in cur_cols
            else "NULL AS turnover_rate"
        )
        pe_sel = "k.pe_ttm" if "pe_ttm" in cur_cols else "NULL AS pe_ttm"
        ph = ",".join("?" * len(uniq))
        kline = pd.read_sql_query(
            f"""
            SELECT k.date, k.stock_code, {tr_sel}, {pe_sel}
            FROM stock_daily_kline k
            WHERE k.date = ? AND k.stock_code IN ({ph})
            """,
            conn,
            params=(td, *uniq),
        )
        mf = pd.read_sql_query(
            f"""
            SELECT stock_code, big_net_ratio
            FROM stock_money_flow_daily
            WHERE trade_date = ? AND stock_code IN ({ph})
            """,
            conn,
            params=(td, *uniq),
        )
        fin = pd.read_sql_query(
            f"""
            SELECT f.stock_code, f.pub_date, f.roe, f.net_profit_growth, f.revenue_growth
            FROM stock_financial_data f
            INNER JOIN (
                SELECT stock_code, MAX(pub_date) AS mx_pub
                FROM stock_financial_data
                WHERE pub_date <= ? AND stock_code IN ({ph})
                GROUP BY stock_code
            ) t ON f.stock_code = t.stock_code AND f.pub_date = t.mx_pub
            WHERE f.stock_code IN ({ph})
            """,
            conn,
            params=(td, *uniq, *uniq),
        )
    finally:
        conn.close()

    if kline.empty:
        base = pd.DataFrame({"stock_code": uniq, "date": td})
    else:
        base = kline.copy()
        base["stock_code"] = base["stock_code"].astype(str).str.zfill(6)
        base["date"] = base["date"].astype(str).str[:10]

    if not mf.empty:
        mf["stock_code"] = mf["stock_code"].astype(str).str.zfill(6)
        base = base.merge(mf, on="stock_code", how="left")
    elif "big_net_ratio" not in base.columns:
        base["big_net_ratio"] = np.nan

    if not fin.empty:
        fin["stock_code"] = fin["stock_code"].astype(str).str.zfill(6)
        fin = fin.drop_duplicates(subset=["stock_code"], keep="last")
        base = base.merge(
            fin[["stock_code", "roe", "net_profit_growth", "revenue_growth"]],
            on="stock_code",
            how="left",
        )
    else:
        for c in ("roe", "net_profit_growth", "revenue_growth"):
            if c not in base.columns:
                base[c] = np.nan

    for c in ("turnover_rate", "pe_ttm", "big_net_ratio"):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    return base


def fetch_stock_financial_panel(stock_code: str, *, db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT pub_date, report_date, roe, net_profit_growth, revenue_growth FROM stock_financial_data WHERE stock_code = ? ORDER BY pub_date ASC, report_date ASC", conn, params=(str(stock_code).strip().zfill(6),))


def upsert_stock_financial_rows(records: list[tuple[Any, ...]], db_path: Path | None = None) -> int:
    if not records: return 0
    def _do() -> None:
        with get_connection(db_path) as conn:
            conn.executemany("INSERT OR REPLACE INTO stock_financial_data (stock_code, pub_date, report_date, roe, net_profit_growth, revenue_growth) VALUES (?, ?, ?, ?, ?, ?)", records)
    _retry_sqlite_locked(_do, attempts=5)
    return len(records)


def fetch_stock_code_max_dates(db_path: Path | None = None) -> dict[str, str]:
    with get_connection(db_path) as conn:
        return {str(r[0]).strip().zfill(6): str(r[1]).strip()[:10] for r in conn.execute("SELECT stock_code, MAX(date) FROM stock_daily_kline GROUP BY stock_code").fetchall() if r[0] and r[1]}
