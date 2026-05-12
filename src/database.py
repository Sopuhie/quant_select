"""SQLite 数据库：建表、写入选股与预测、模型版本。"""
from __future__ import annotations

import json
import math
import sqlite3

import pandas as pd
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import DB_PATH

# 高并发读写时延长连接等待、WAL、busy_timeout，降低 "database is locked" 概率
_SQLITE_CONNECT_TIMEOUT = 30.0


def _apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")


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
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_kline_code_date ON stock_daily_kline(stock_code, date);

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
"""


def _ensure_stock_daily_kline_industry(conn: sqlite3.Connection) -> None:
    """旧库升级：为 ``stock_daily_kline`` 增加 ``industry`` 列。"""
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_daily_kline'"
    )
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(stock_daily_kline)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "industry" not in cols:
        conn.execute("ALTER TABLE stock_daily_kline ADD COLUMN industry TEXT")


def _ensure_daily_selections_selection_reason(conn: sqlite3.Connection) -> None:
    """旧库升级：为 ``daily_selections`` 增加 ``selection_reason``（入选原因）。"""
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_selections'"
    )
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(daily_selections)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "selection_reason" not in cols:
        print(
            "[DB] 检测到缺少 selection_reason 列，正在自动补丁 daily_selections …",
            flush=True,
        )
        try:
            conn.execute(
                "ALTER TABLE daily_selections ADD COLUMN selection_reason TEXT"
            )
            print("[DB] daily_selections.selection_reason 补丁已成功应用。", flush=True)
        except Exception as exc:
            print(
                f"[DB] ALTER TABLE selection_reason 失败（可能已存在）: {exc}",
                flush=True,
            )


def _ensure_stock_daily_kline_market_cap(conn: sqlite3.Connection) -> None:
    """旧库升级：为 ``stock_daily_kline`` 增加 ``market_cap``（总市值，元）。"""
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_daily_kline'"
    )
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(stock_daily_kline)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "market_cap" not in cols:
        conn.execute("ALTER TABLE stock_daily_kline ADD COLUMN market_cap REAL")


def init_db(db_path: Path | None = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    try:
        _apply_sqlite_pragmas(conn)
        conn.executescript(SCHEMA_SQL)
        _ensure_stock_daily_kline_industry(conn)
        _ensure_stock_daily_kline_market_cap(conn)
        _ensure_daily_selections_selection_reason(conn)
        conn.commit()
    finally:
        conn.close()


def fetch_stock_financial_panel(
    stock_code: str,
    *,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """
    读取单只股票在 ``stock_financial_data`` 中的财报行（按公告日、报告期升序）。
    用于与日线 ``merge_asof``：仅使用 ``pub_date`` 已过的记录，避免前视偏差。
    """
    code = str(stock_code).strip().zfill(6)
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    try:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_financial_data'"
        )
        if cur.fetchone() is None:
            return pd.DataFrame(
                columns=["pub_date", "report_date", "roe", "net_profit_growth", "revenue_growth"]
            )
        return pd.read_sql_query(
            """
            SELECT pub_date, report_date, roe, net_profit_growth, revenue_growth
            FROM stock_financial_data
            WHERE stock_code = ?
            ORDER BY pub_date ASC, report_date ASC
            """,
            conn,
            params=(code,),
        )
    finally:
        conn.close()


def upsert_stock_financial_rows(
    records: list[tuple[Any, ...]],
    db_path: Path | None = None,
) -> int:
    """批量写入/覆盖 ``stock_financial_data``。每元组：
    (stock_code, pub_date, report_date, roe, net_profit_growth, revenue_growth)
    """
    if not records:
        return 0
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(str(path), timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_financial_data
            (stock_code, pub_date, report_date, roe, net_profit_growth, revenue_growth)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    finally:
        conn.close()
    return len(records)


def fetch_stock_code_max_dates(db_path: Path | None = None) -> dict[str, str]:
    """返回 ``{stock_code(6位): 最新日线日期 YYYY-MM-DD}``，用于增量同步跳过已最新标的。"""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT stock_code, MAX(date) FROM stock_daily_kline GROUP BY stock_code"
        )
        out: dict[str, str] = {}
        for k, v in cur.fetchall():
            if k is None or v is None:
                continue
            out[str(k).strip().zfill(6)] = str(v).strip()[:10]
        return out


def open_sqlite_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """建立可写连接（WAL + 长超时）。调用方负责 ``commit`` / ``close``。"""
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


def insert_daily_selections(
    rows: Iterable[dict[str, Any]],
    db_path: Path | None = None,
) -> None:
    """写入 Top3；若同日同代码已存在且已有收益字段，则保留原收益（避免 --force 覆盖）。"""
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
    with get_connection(db_path) as conn:
        conn.executemany(sql, batch)


def upsert_stock_daily_klines(
    rows: Iterable[dict[str, Any]],
    db_path: Path | None = None,
    *,
    connection: sqlite3.Connection | None = None,
) -> None:
    """写入或更新本地日线缓存；冲突键为 (date, stock_code)。

    支持可选字段 ``industry``（申万/中信等行业文本）；空字符串不会覆盖库内已有行业
    （``COALESCE(NULLIF(TRIM(excluded.industry),''), …)``）。

    若传入 ``connection``，则不单独提交/关闭连接（由调用方 ``commit``）。
    """
    sql = """
    INSERT INTO stock_daily_kline (
        date, stock_code, stock_name, industry, market_cap,
        open, high, low, close, volume
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(date, stock_code) DO UPDATE SET
        stock_name = excluded.stock_name,
        industry = COALESCE(NULLIF(TRIM(excluded.industry), ''), stock_daily_kline.industry),
        market_cap = COALESCE(excluded.market_cap, stock_daily_kline.market_cap),
        open = excluded.open,
        high = excluded.high,
        low = excluded.low,
        close = excluded.close,
        volume = excluded.volume
    """
    batch: list[tuple[Any, ...]] = []
    for r in rows:
        ind_raw = r.get("industry")
        industry_val = (
            str(ind_raw).strip()
            if ind_raw is not None and str(ind_raw).strip()
            else ""
        )
        mc_raw = r.get("market_cap")
        try:
            market_cap_val = (
                float(mc_raw)
                if mc_raw is not None and pd.notna(mc_raw)
                else None
            )
        except (TypeError, ValueError):
            market_cap_val = None
        batch.append(
            (
                r["date"],
                str(r["stock_code"]).strip().zfill(6),
                str(r.get("stock_name") or "").strip(),
                industry_val,
                market_cap_val,
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
    if connection is not None:
        conn = connection
    else:
        conn = sqlite3.connect(str(db_path or DB_PATH), timeout=_SQLITE_CONNECT_TIMEOUT)
        _apply_sqlite_pragmas(conn)
    try:
        conn.executemany(sql, batch)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


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


def insert_system_log(
    task_name: str,
    status: str,
    parameters: str | None,
    log_output: str | None,
    *,
    db_path: Path | None = None,
) -> None:
    """写入一条系统任务日志（SUCCESS / FAILED 等）。用于控制台任务与前端联动追溯。"""
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO system_logs (task_name, status, run_time, parameters, log_output)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(task_name).strip(),
                str(status).strip(),
                run_time,
                parameters if parameters is not None else "",
                log_output if log_output is not None else "",
            ),
        )


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
        SELECT rank, stock_code, stock_name, score, close_price, selection_reason
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
        SELECT rank, stock_code, stock_name, score, close_price, selection_reason
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
        score_v = r.get("score")
        try:
            score_f = float(score_v) if score_v is not None and pd.notna(score_v) else None
        except (TypeError, ValueError):
            score_f = None
        cp_v = r.get("close_price")
        try:
            close_f = float(cp_v) if cp_v is not None and pd.notna(cp_v) else None
        except (TypeError, ValueError):
            close_f = None
        rows.append(
            {
                "rank": rk,
                "stock_code": str(r["stock_code"]).strip(),
                "stock_name": str(r.get("stock_name") or "").strip(),
                "score": score_f,
                "close_price": close_f,
                "selection_reason": str(r.get("selection_reason") or "").strip(),
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


def fetch_stock_daily_bars_until(
    stock_code: str,
    end_date: str,
    *,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """
    从 ``stock_daily_kline`` 读取单只股票截至 ``end_date``（含）的日线，列与在线行情对齐。
    """
    code = str(stock_code).strip().zfill(6)
    end = str(end_date).strip()[:10]
    path_obj = db_path or DB_PATH
    init_db(path_obj)
    path = str(path_obj)
    conn = sqlite3.connect(path, timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    try:
        df = pd.read_sql_query(
            """
            SELECT date, open, high, low, close, volume, industry, market_cap
            FROM stock_daily_kline
            WHERE stock_code = ? AND date <= ?
            ORDER BY date ASC
            """,
            conn,
            params=(code, end),
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume", "market_cap"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "industry" in df.columns:
        df["industry"] = df["industry"].fillna("").astype(str)
    return df.dropna(subset=["close"]).reset_index(drop=True)


def fetch_latest_industry_by_codes(
    stock_codes: Iterable[str],
    *,
    db_path: Path | None = None,
) -> dict[str, str]:
    """
    每只代码取 ``stock_daily_kline`` 中最新交易日一行的 ``industry``（原始库内值，可为空）。
    供在线拉 K 线训练路径与行业字段对齐；缺失代码不出现在字典中。
    """
    seen: set[str] = set()
    uniq: list[str] = []
    for c in stock_codes:
        code = str(c).strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        if code not in seen:
            seen.add(code)
            uniq.append(code)
    if not uniq:
        return {}
    out: dict[str, str] = {}
    chunk_size = 400
    with get_connection(db_path) as conn:
        for i in range(0, len(uniq), chunk_size):
            chunk = uniq[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            sql = f"""
            SELECT k.stock_code, k.industry
            FROM stock_daily_kline k
            INNER JOIN (
                SELECT stock_code, MAX(date) AS mx
                FROM stock_daily_kline
                WHERE stock_code IN ({placeholders})
                GROUP BY stock_code
            ) t ON k.stock_code = t.stock_code AND k.date = t.mx
            """
            cur = conn.execute(sql, chunk)
            for row in cur.fetchall():
                sc = str(row[0]).strip().zfill(6)
                ind = row[1]
                out[sc] = "" if ind is None else str(ind).strip()
    return out


def fetch_latest_market_cap_by_codes(
    stock_codes: Iterable[str],
    *,
    db_path: Path | None = None,
) -> dict[str, float]:
    """
    每只代码取 ``stock_daily_kline`` 最新交易日一行上的 ``market_cap``（元，>0）。
    缺失或非正的值不写入字典；供在线 K 线推断 ``factor_size_mcap`` 时广播。
    """
    seen: set[str] = set()
    uniq: list[str] = []
    for c in stock_codes:
        code = str(c).strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        if code not in seen:
            seen.add(code)
            uniq.append(code)
    if not uniq:
        return {}
    out: dict[str, float] = {}
    chunk_size = 400
    with get_connection(db_path) as conn:
        for i in range(0, len(uniq), chunk_size):
            chunk = uniq[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            sql = f"""
            SELECT k.stock_code, k.market_cap
            FROM stock_daily_kline k
            INNER JOIN (
                SELECT stock_code, MAX(date) AS mx
                FROM stock_daily_kline
                WHERE stock_code IN ({placeholders})
                GROUP BY stock_code
            ) t ON k.stock_code = t.stock_code AND k.date = t.mx
            """
            cur = conn.execute(sql, chunk)
            for row in cur.fetchall():
                sc = str(row[0]).strip().zfill(6)
                mc = row[1]
                try:
                    v = float(mc)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v > 0:
                    out[sc] = v
    return out


def list_predict_universe_from_kline(
    trade_date: str,
    *,
    min_bars: int,
    max_count: int | None,
    db_path: Path | None = None,
) -> list[tuple[str, str]]:
    """
    截至 ``trade_date``（含）本地 K 线条数 >= ``min_bars`` 的股票，
    名称取该 cutoff 下最新一根的名称；按代码排序后可选截断 ``max_count``
    （``None`` 或 ``<=0`` 表示不限制，返回库内全部满足条数的股票）。
    """
    end = str(trade_date).strip()[:10]
    mb = max(1, int(min_bars))
    path = str(db_path or DB_PATH)
    sql_body = """
    WITH eligible AS (
        SELECT stock_code
        FROM stock_daily_kline
        WHERE date <= ?
        GROUP BY stock_code
        HAVING COUNT(*) >= ?
    ),
    latest AS (
        SELECT stock_code, MAX(date) AS mx
        FROM stock_daily_kline
        WHERE date <= ?
        GROUP BY stock_code
    )
    SELECT k.stock_code, k.stock_name
    FROM stock_daily_kline k
    INNER JOIN eligible e ON k.stock_code = e.stock_code
    INNER JOIN latest L ON k.stock_code = L.stock_code AND k.date = L.mx
    ORDER BY k.stock_code
    """
    use_limit = max_count is not None and int(max_count) > 0
    sql = sql_body + (" LIMIT ?" if use_limit else "")
    params: tuple[Any, ...] = (end, mb, end)
    if use_limit:
        params = (*params, int(max_count))

    conn = sqlite3.connect(path, timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    try:
        raw = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    if raw.empty:
        return []
    out: list[tuple[str, str]] = []
    for _, row in raw.iterrows():
        c = str(row["stock_code"]).strip().zfill(6)
        n = str(row.get("stock_name") or "").strip()
        out.append((c, n))
    return out


def stock_codes_with_local_bars(
    trade_date: str,
    min_bars: int,
    *,
    db_path: Path | None = None,
) -> set[str]:
    """截至 trade_date 至少有 min_bars 根 K 线的 6 位代码集合。"""
    end = str(trade_date).strip()[:10]
    mb = max(1, int(min_bars))
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path, timeout=_SQLITE_CONNECT_TIMEOUT)
    _apply_sqlite_pragmas(conn)
    try:
        cur = conn.execute(
            """
            SELECT stock_code FROM stock_daily_kline
            WHERE date <= ?
            GROUP BY stock_code
            HAVING COUNT(*) >= ?
            """,
            (end, mb),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return {str(r[0]).strip().zfill(6) for r in rows}
