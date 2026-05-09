"""
全量初始化与每日增量更新本地 SQLite 数据库日线行情。

在 quant_select 根目录执行:
  python scripts/update_local_data.py

逻辑：本地无该代码记录则拉取约 365 自然日；已有则从「最近收盘日」次日增量拉取并 UPSERT。
实际拉取走 ``fetch_daily_hist``（AkShare + 重试 + Baostock 兜底），与项目其余模块一致。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DB_PATH
from src.database import init_db, upsert_stock_daily_klines
from src.data_fetcher import fetch_daily_hist, get_stock_pool
from src.utils import get_last_trading_date, next_trade_day_after


def _parse_db_date(s: str) -> datetime:
    s = str(s).strip()
    if "-" in s:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    return datetime.strptime(s[:8], "%Y%m%d")


def update_database_kline(*, max_stocks: int = 6000, pool_type: str = "all") -> None:
    init_db()

    print("正在拉取全市场股票代码列表...", flush=True)
    stock_pool = get_stock_pool(pool_type=pool_type, max_count=max_stocks)
    if not stock_pool:
        print("错误: 无法获取股票池，请检查网络连接！", flush=True)
        raise SystemExit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        "SELECT stock_code, MAX(date) FROM stock_daily_kline GROUP BY stock_code"
    )
    date_mapping = {str(k).zfill(6): v for k, v in cur.fetchall()}

    total = len(stock_pool)
    # 用「最近交易日」而非日历「今天」判断增量：否则周末/休市日会误判为未更新，对全市场重复请求接口。
    latest_trade = get_last_trading_date()
    end_compact = latest_trade.replace("-", "")

    print(
        f"全市场共获取到 {total} 只股票；本地已有缓存 {len(date_mapping)} 只。"
        f" 增量截止日期(最近交易日): {latest_trade}",
        flush=True,
    )
    print("开始处理数据更新...", flush=True)

    skipped_no_fetch = 0
    fetch_calls = 0
    stocks_saved = 0

    commit_every = 50
    for idx, (code, name) in enumerate(stock_pool):
        code = str(code).strip().zfill(6)
        if idx % 100 == 0 or idx == total - 1:
            print(
                f"进度: {idx + 1}/{total} "
                f"(跳过无需请求: {skipped_no_fetch}, 已保存新数据: {stocks_saved}, 行情请求次数: {fetch_calls})",
                flush=True,
            )

        last_date = date_mapping.get(code)
        # 本地最新一根 K 线的日期已经不早于最近交易日 → 无需再请求网络
        if last_date:
            ld = str(last_date).strip()[:10]
            if ld >= latest_trade:
                skipped_no_fetch += 1
                continue

        if not last_date:
            start_compact = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        else:
            ld_raw = str(last_date).strip()[:10]
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
            skipped_no_fetch += 1
            continue

        try:
            fetch_calls += 1
            df = fetch_daily_hist(
                code,
                start_date=start_compact,
                end_date=end_compact,
                adjust="qfq",
            )
            if df is None or df.empty:
                continue

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

            upsert_stock_daily_klines(records, connection=conn)
            stocks_saved += 1
            if stocks_saved % commit_every == 0:
                conn.commit()

        except Exception:
            pass

    conn.commit()
    conn.close()
    print(
        f"本轮结束：共 {total} 只股票；无需请求(已跟上最近交易日): {skipped_no_fetch}；"
        f" 实际发起行情请求: {fetch_calls} 次；写入新 K 线批次: {stocks_saved} 只。",
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
    args = parser.parse_args()
    update_database_kline(max_stocks=args.max_stocks, pool_type=args.pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
