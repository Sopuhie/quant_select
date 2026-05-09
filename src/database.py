"""SQLite 数据库：建表、写入选股与预测、模型版本。"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import DB_PATH

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
"""


def init_db(db_path: Path | None = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_daily_selections(
    rows: Iterable[dict[str, Any]],
    db_path: Path | None = None,
) -> None:
    """写入 Top3；若同日同代码已存在且已有收益字段，则保留原收益（避免 --force 覆盖）。"""
    sql = """
    INSERT INTO daily_selections
    (trade_date, stock_code, stock_name, rank, score, close_price,
     next_day_return, hold_5d_return, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(trade_date, stock_code) DO UPDATE SET
        stock_name = excluded.stock_name,
        rank = excluded.rank,
        score = excluded.score,
        close_price = excluded.close_price,
        next_day_return = coalesce(next_day_return, excluded.next_day_return),
        hold_5d_return = coalesce(hold_5d_return, excluded.hold_5d_return)
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
                r.get("created_at", now),
            )
        )
    with get_connection(db_path) as conn:
        conn.executemany(sql, batch)


def insert_daily_predictions(
    rows: Iterable[dict[str, Any]],
    db_path: Path | None = None,
) -> None:
    sql = """
    INSERT OR REPLACE INTO daily_predictions
    (trade_date, stock_code, stock_name, score, rank_in_market, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    batch = []
    for r in rows:
        batch.append(
            (
                r["trade_date"],
                r["stock_code"],
                r.get("stock_name"),
                r["score"],
                r["rank_in_market"],
                r.get("created_at", now),
            )
        )
    with get_connection(db_path) as conn:
        conn.executemany(sql, batch)


def register_model_version(
    version: str,
    train_end_date: str,
    features: list[str],
    metrics: dict[str, Any],
    set_active: bool = True,
    db_path: Path | None = None,
) -> None:
    with get_connection(db_path) as conn:
        if set_active:
            conn.execute("UPDATE model_versions SET is_active = 0")
        conn.execute(
            """
            INSERT INTO model_versions
            (version, train_end_date, features, metrics, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                train_end_date,
                json.dumps(features, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                1 if set_active else 0,
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )


def get_active_model_version(db_path: Path | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("features"):
        d["features"] = json.loads(d["features"])
    if d.get("metrics"):
        d["metrics"] = json.loads(d["metrics"])
    return d


def query_df(sql: str, params: tuple[Any, ...] = (), db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def fetch_selection_rows_for_date(
    trade_date: str,
    limit: int | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """读取某日选股记录（默认 TopN 条），供钉钉推送等使用。"""
    from .config import TOP_N_SELECTION

    lim = int(limit if limit is not None else TOP_N_SELECTION)
    df = query_df(
        """
        SELECT rank, stock_code, stock_name, score, close_price
        FROM daily_selections
        WHERE trade_date = ?
        ORDER BY rank ASC
        LIMIT ?
        """,
        (trade_date, lim),
        db_path,
    )
    return df.to_dict("records")


def fetch_selection_rows_for_dingtalk_push(
    trade_date: str,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    读取指定交易日的选股记录，并按与 Streamlit「今日推荐」相同的规则清洗：
    仅 rank∈[1,TOP_N]、同 rank 保留 score 较高的一条，最终至多 TOP_N 条。
    用于钉钉推送，保证与页面「今日推荐」一致。
    """
    from .config import TOP_N_SELECTION

    df = query_df(
        """
        SELECT rank, stock_code, stock_name, score
        FROM daily_selections
        WHERE trade_date = ?
        ORDER BY rank ASC
        """,
        (trade_date,),
        db_path,
    )
    if df.empty:
        return []

    out = df[df["rank"].isin(range(1, TOP_N_SELECTION + 1))].copy()
    if out.empty:
        return []

    out = out.sort_values("score", ascending=False).drop_duplicates(
        subset=["rank"], keep="first"
    )
    out = out.sort_values("rank").reset_index(drop=True).head(TOP_N_SELECTION)

    rows: list[dict[str, Any]] = []
    for _, r in out.iterrows():
        try:
            rk = int(r["rank"])
        except (TypeError, ValueError):
            rk = r["rank"]
        rows.append(
            {
                "rank": rk,
                "stock_code": str(r["stock_code"]).strip(),
                "stock_name": str(r.get("stock_name") or "").strip(),
            }
        )
    return rows


def selection_exists_for_date(trade_date: str, db_path: Path | None = None) -> bool:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT 1 FROM daily_selections WHERE trade_date = ? LIMIT 1",
            (trade_date,),
        )
        return cur.fetchone() is not None


def delete_daily_outputs_for_trade_date(trade_date: str, db_path: Path | None = None) -> None:
    """删除某交易日的选股与全市场预测，便于同日重新写入时不会出现「多出一套 TopN」。"""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM daily_selections WHERE trade_date = ?", (trade_date,))
        conn.execute("DELETE FROM daily_predictions WHERE trade_date = ?", (trade_date,))


def update_selection_returns(
    trade_date: str,
    stock_code: str,
    next_day_return: float | None = None,
    hold_5d_return: float | None = None,
    db_path: Path | None = None,
) -> int:
    """按交易日 + 代码更新收益；代码统一为 6 位以匹配库内主键。返回受影响的行数。"""
    sets = []
    vals: list[Any] = []
    if next_day_return is not None:
        sets.append("next_day_return = ?")
        vals.append(next_day_return)
    if hold_5d_return is not None:
        sets.append("hold_5d_return = ?")
        vals.append(hold_5d_return)
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
    sql = f"UPDATE daily_selections SET {', '.join(sets)} WHERE {where}"
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, vals)
        return int(cur.rowcount if cur.rowcount is not None else 0)
