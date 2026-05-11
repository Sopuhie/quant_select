"""
全量初始化与每日增量更新本地 SQLite 数据库日线行情。

在 quant_select 根目录执行:
  python scripts/update_local_data.py

逻辑：本地无该代码记录则拉取约 365 自然日；已有则从「最近收盘日」次日增量拉取并 UPSERT。
启动前一次性查询 ``SELECT stock_code, MAX(date) GROUP BY stock_code`` 建查找表；已对齐最近交易日的标的跳过网络请求。
并发拉取使用 ``ThreadPoolExecutor``（默认 8 线程，可用 ``--workers`` 或环境变量 ``QUANT_UPDATE_WORKERS`` 调整），写入仍在主线程串行提交以降低锁竞争。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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
from src.data_fetcher import fetch_daily_hist, get_stock_pool
from src.utils import get_last_trading_date, next_trade_day_after

DEFAULT_WORKERS = max(1, int(os.environ.get("QUANT_UPDATE_WORKERS", "8")))


def _parse_db_date(s: str) -> datetime:
    s = str(s).strip()
    if "-" in s:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    return datetime.strptime(s[:8], "%Y%m%d")


def _worker_fetch_incremental(
    code: str,
    name: str,
    last_date_raw: object | None,
    latest_trade: str,
    end_compact: str,
) -> tuple[list[dict] | None, str]:
    """
    网络拉取（线程内执行）。返回 (records, status)。
    status: fetched | uptodate | bad_range | empty | error
    """
    code = str(code).strip().zfill(6)
    if last_date_raw is not None:
        ld = str(last_date_raw).strip()[:10]
        if ld >= latest_trade:
            return None, "uptodate"

    if last_date_raw is None:
        start_compact = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    else:
        ld_raw = str(last_date_raw).strip()[:10]
        nxt_trade = next_trade_day_after(ld_raw)
        if nxt_trade:
            start_compact = nxt_trade.replace("-", "")[:8]
        else:
            try:
                nxt = _parse_db_date(ld_raw) + timedelta(days=1)
            except ValueError:
                nxt = datetime.now() - timedelta(days=365)
            start_compact = nxt.strftime("%Y%m%d")

    if start_compact > end_compact:
        return None, "bad_range"

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
        f" 增量截止(最近交易日): {latest_trade}；并发线程: {workers}。",
        flush=True,
    )

    skipped_uptodate = 0
    skipped_bad_range = 0
    empty_fetch = 0
    errors = 0
    stocks_saved = 0
    fetch_jobs = 0

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

    commit_every = 50
    conn = open_sqlite_connection(DB_PATH)
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 32))) as ex:
            futs = [
                ex.submit(
                    _worker_fetch_incremental,
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
                        f"进度: {done}/{n_pending} "
                        f"(已写入批次: {stocks_saved}, 有效拉取: {fetch_jobs})",
                        flush=True,
                    )
                try:
                    records, status = fut.result()
                except Exception:
                    errors += 1
                    continue

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
                    upsert_stock_daily_klines(records, connection=conn)
                    stocks_saved += 1
                    if stocks_saved % commit_every == 0:
                        conn.commit()

        conn.commit()
    finally:
        conn.close()

    elapsed = time.perf_counter() - t0
    print(
        f"本轮结束：共 {total} 只股票；无需请求(已跟上最近交易日): {skipped_uptodate}；"
        f" 区间无效跳过: {skipped_bad_range}；空返回: {empty_fetch}；异常: {errors}；"
        f" 实际写入批次: {stocks_saved}；总耗时 {elapsed:.1f}s。",
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
