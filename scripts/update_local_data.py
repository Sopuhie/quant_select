"""
全量初始化与每日增量更新本地 SQLite 数据库日线行情。

在 quant_select 根目录执行:
  python scripts/update_local_data.py

逻辑：本地无该代码记录则拉取约 365 自然日；已有则从「最近收盘日」次日增量拉取并 UPSERT。
启动前一次性查询 ``SELECT stock_code, MAX(date) GROUP BY stock_code`` 建查找表；已对齐最近交易日的标的跳过网络请求。

并发策略（SQLite 友好）：
  - **阶段 1**：``ThreadPoolExecutor`` 内仅执行网络拉取，不写数据库；
  - **阶段 2**：线程池结束后，由 **主线程** 将内存中的 K 线分批 ``UPSERT`` 并 ``commit``，避免多线程竞争连接导致锁等待或死锁。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DB_PATH
from src.database import (
    fetch_stock_code_max_dates,
    init_db,
    open_sqlite_connection,
    upsert_stock_daily_klines,
)
from src.data_fetcher import (
    fetch_daily_hist,
    get_stock_pool,
    resolve_incremental_daily_fetch_window,
)
from src.utils import get_last_trading_date

DEFAULT_WORKERS = max(1, int(os.environ.get("QUANT_UPDATE_WORKERS", "8")))
# 主线程单次 executemany 行数上限（过大易占用内存；过小则事务开销大）
UPSERT_FLUSH_ROWS = max(1000, int(os.environ.get("QUANT_UPSERT_FLUSH_ROWS", "8000")))
COMMIT_EVERY_FLUSHES = max(1, int(os.environ.get("QUANT_COMMIT_EVERY_FLUSHES", "3")))


def _worker_fetch_only(
    code: str,
    name: str,
    last_date_raw: object | None,
    latest_trade: str,
    end_compact: str,
) -> tuple[list[dict] | None, str]:
    """
    仅在 worker 线程中执行网络请求与内存组装，**禁止**访问 SQLite。
    返回 (records, status)；status ∈ fetched | uptodate | bad_range | empty | error
    """
    code = str(code).strip().zfill(6)
    start_compact, kind = resolve_incremental_daily_fetch_window(
        last_date_raw,
        latest_trade,
        end_compact,
    )
    if kind != "fetch":
        return None, kind

    try:
        df = fetch_daily_hist(
            code,
            start_date=start_compact,
            end_date=end_compact,
            adjust="qfq",
        )
    except Exception:
        return None, "error"

    if df is None or df.empty:
        return None, "empty"

    records: list[dict] = []
    for _, row in df.iterrows():
        d = row["date"]
        ds = d if isinstance(d, str) else pd.Timestamp(d).strftime("%Y-%m-%d")
        records.append(
            {
                "date": ds,
                "stock_code": code,
                "stock_name": str(name).strip(),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
    return records, "fetched"


def update_database_kline(
    *,
    max_stocks: int = 6000,
    pool_type: str = "all",
    workers: int = DEFAULT_WORKERS,
) -> None:
    init_db()

    print("正在拉取全市场股票代码列表...", flush=True)
    stock_pool = get_stock_pool(pool_type=pool_type, max_count=max_stocks)
    if not stock_pool:
        print("错误: 无法获取股票池，请检查网络连接！", flush=True)
        raise SystemExit(1)

    t0 = time.perf_counter()
    date_mapping = fetch_stock_code_max_dates(DB_PATH)
    t_map = time.perf_counter()

    total = len(stock_pool)
    latest_trade = get_last_trading_date()
    end_compact = latest_trade.replace("-", "")

    print(
        f"全市场共获取到 {total} 只股票；本地已有缓存 {len(date_mapping)} 只。"
        f" MAX(date) 映射查询耗时 {(t_map - t0) * 1000:.1f} ms。"
        f" 增量截止(最近交易日): {latest_trade}；并发拉取线程: {workers}。",
        flush=True,
    )

    skipped_uptodate = 0
    pending: list[tuple[str, str, object | None]] = []
    for c, n in stock_pool:
        code = str(c).strip().zfill(6)
        last = date_mapping.get(code)
        if last is not None and str(last).strip()[:10] >= latest_trade:
            skipped_uptodate += 1
            continue
        pending.append((code, str(n).strip(), last))

    n_pending = len(pending)
    print(
        f"待拉取（未对齐 {latest_trade}）: {n_pending} 只；已跳过最新: {skipped_uptodate}。",
        flush=True,
    )

    # ---------- 阶段 1：仅并发拉取（零 SQLite） ----------
    t_fetch0 = time.perf_counter()
    outcomes: list[tuple[list[dict] | None, str]] = []
    workers_n = max(1, min(workers, 32))

    with ThreadPoolExecutor(max_workers=workers_n) as ex:
        futs = [
            ex.submit(
                _worker_fetch_only,
                code,
                name,
                last_raw,
                latest_trade,
                end_compact,
            )
            for code, name, last_raw in pending
        ]
        done = 0
        for fut in as_completed(futs):
            done += 1
            if n_pending and (done % 500 == 0 or done == n_pending):
                print(
                    f"  [拉取] {done}/{n_pending} 个任务已完成…",
                    flush=True,
                )
            try:
                outcomes.append(fut.result())
            except Exception:
                outcomes.append((None, "error"))

    t_fetch1 = time.perf_counter()
    print(
        f"阶段 1 结束：并发拉取耗时 {t_fetch1 - t_fetch0:.1f}s；开始主线程批量写入…",
        flush=True,
    )

    skipped_bad_range = 0
    empty_fetch = 0
    errors = 0
    fetch_jobs = 0
    pending_writes: list[list[dict]] = []

    for records, status in outcomes:
        if status == "uptodate":
            skipped_uptodate += 1
            continue
        if status == "bad_range":
            skipped_bad_range += 1
            continue
        if status == "empty":
            empty_fetch += 1
            continue
        if status == "error":
            errors += 1
            continue
        if status == "fetched" and records:
            fetch_jobs += 1
            pending_writes.append(records)

    # ---------- 阶段 2：主线程分批 UPSERT ----------
    stocks_batches = 0
    buf: list[dict] = []
    flush_ops = 0
    conn = open_sqlite_connection(DB_PATH)
    try:
        for recs in pending_writes:
            buf.extend(recs)
            stocks_batches += 1
            while len(buf) >= UPSERT_FLUSH_ROWS:
                chunk = buf[:UPSERT_FLUSH_ROWS]
                del buf[:UPSERT_FLUSH_ROWS]
                upsert_stock_daily_klines(chunk, connection=conn)
                flush_ops += 1
                if flush_ops % COMMIT_EVERY_FLUSHES == 0:
                    conn.commit()
                    print(
                        f"  [写入] 已执行 upsert 批次 {flush_ops} "
                        f"（约每批 {UPSERT_FLUSH_ROWS} 行）…",
                        flush=True,
                    )

        if buf:
            upsert_stock_daily_klines(buf, connection=conn)
            flush_ops += 1
        conn.commit()
    finally:
        conn.close()

    t_end = time.perf_counter()
    print(
        f"本轮结束：共 {total} 只股票；无需请求(已跟上最近交易日): {skipped_uptodate}；"
        f" 区间无效: {skipped_bad_range}；空返回: {empty_fetch}；拉取异常: {errors}；"
        f" 有效拉取股票数: {fetch_jobs}；参与写入的股票批次数: {stocks_batches}；"
        f" DB upsert 次数: {flush_ops}；总耗时 {t_end - t0:.1f}s。",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="同步本地 stock_daily_kline")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=6000,
        help="最多同步的股票数量上限",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="all",
        help="股票池类型：all | hs300 | zz500（默认 all）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发拉取线程数（默认 {DEFAULT_WORKERS}，可用环境变量 QUANT_UPDATE_WORKERS）",
    )
    args = parser.parse_args()
    update_database_kline(
        max_stocks=args.max_stocks,
        pool_type=args.pool,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
