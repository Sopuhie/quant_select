"""
按缺失键从 AkShare 拉取个股资金流 / 北向持股并写入 SQLite（增量 upsert）。

默认从 ``stock_daily_kline`` 读取指定区间内的 (date, stock_code) 网格，与库中已有键求差后补全。

用法（项目根目录 quant_select/）:
  python scripts/sync_auxiliary_features.py --start-date 2024-01-01 --end-date 2024-06-30
  python scripts/sync_auxiliary_features.py --start-date 2024-01-01 --end-date 2024-01-31 --max-codes 50
  python scripts/sync_auxiliary_features.py --codes 000001,600519 --start-date 2024-05-01 --end-date 2024-05-31
  python scripts/sync_auxiliary_features.py --only-money-flow   # 不补北向
  set QUANT_AUX_SYNC_SLEEP=0.4 && python scripts/sync_auxiliary_features.py ...
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auxiliary_ak_sync import sync_auxiliary_for_date_code_grid
from src.config import DB_PATH
from src.database import init_db


def _distinct_dates_codes_from_kline(
    start_date: str,
    end_date: str,
    *,
    max_stocks: int | None,
    db_path: Path,
) -> tuple[list[str], list[str]]:
    s = str(start_date).strip()[:10]
    e = str(end_date).strip()[:10]
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            SELECT DISTINCT date FROM stock_daily_kline
            WHERE date >= ? AND date <= ?
            ORDER BY date
            """,
            (s, e),
        )
        dates = [str(r[0])[:10] for r in cur.fetchall()]
        cur = conn.execute(
            """
            SELECT DISTINCT stock_code FROM stock_daily_kline
            WHERE date >= ? AND date <= ?
            ORDER BY stock_code
            """,
            (s, e),
        )
        codes = [str(r[0]).strip().zfill(6) for r in cur.fetchall()]
    finally:
        conn.close()
    if max_stocks is not None and int(max_stocks) > 0:
        codes = codes[: int(max_stocks)]
    return dates, codes


def main() -> None:
    p = argparse.ArgumentParser(description="增量同步资金流与北向持股特征到 SQLite")
    p.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--codes",
        type=str,
        default="",
        help="逗号分隔 6 位代码；为空则从 K 线表取该区间内出现过的全部代码",
    )
    p.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="从 K 线表取代码时最多保留前 N 个（按代码排序）",
    )
    p.add_argument(
        "--max-codes",
        type=int,
        default=None,
        help="本轮最多对多少只股票发起 Ak 请求（其余缺失键留待下次）",
    )
    p.add_argument("--only-money-flow", action="store_true", help="仅补资金流")
    p.add_argument("--only-north", action="store_true", help="仅补北向持股")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    fill_mf = not args.only_north
    fill_nh = not args.only_money_flow
    if args.only_money_flow and args.only_north:
        raise SystemExit("不能同时指定 --only-money-flow 与 --only-north")

    init_db(DB_PATH)

    if args.codes.strip():
        s = str(args.start_date).strip()[:10]
        e = str(args.end_date).strip()[:10]
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.execute(
                """
                SELECT DISTINCT date FROM stock_daily_kline
                WHERE date >= ? AND date <= ?
                ORDER BY date
                """,
                (s, e),
            )
            dates = [str(r[0])[:10] for r in cur.fetchall()]
        finally:
            conn.close()
        codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
        if not dates:
            raise SystemExit("区间内 K 线无日期，请检查 --start-date / --end-date 与本地库。")
        if not codes:
            raise SystemExit("--codes 为空")
    else:
        dates, codes = _distinct_dates_codes_from_kline(
            args.start_date,
            args.end_date,
            max_stocks=args.max_stocks,
            db_path=DB_PATH,
        )
    if not dates or not codes:
        raise SystemExit("无可用日期或股票代码（请先同步 stock_daily_kline）。")

    stats = sync_auxiliary_for_date_code_grid(
        dates,
        codes,
        fill_money_flow=fill_mf,
        fill_north=fill_nh,
        sleep_sec=None,
        max_codes=args.max_codes,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print("完成:", stats, flush=True)


if __name__ == "__main__":
    main()
