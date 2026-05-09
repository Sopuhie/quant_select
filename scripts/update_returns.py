"""
每日 / 手动回填 daily_selections 的次日与 5 个交易日收益。

在 quant_select 根目录执行:
  python scripts/update_returns.py
  python scripts/update_returns.py --only next
  python scripts/update_returns.py --only h5
  python scripts/update_returns.py --start-date 20180101
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import init_db
from src.return_updater import (
    update_all_returns,
    update_hold_5d_returns,
    update_next_day_returns,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="回填选股记录收益")
    parser.add_argument(
        "--only",
        choices=("all", "next", "h5"),
        default="all",
        help="all=次日+5日；next=仅次日；h5=仅5日",
    )
    parser.add_argument(
        "--start-date",
        default="20150101",
        help="拉 K 线的起始日期 YYYYMMDD",
    )
    args = parser.parse_args()
    init_db()
    if args.only == "next":
        n = update_next_day_returns(start_date=args.start_date)
        print("已更新次日收益（影响行数累计）:", n)
    elif args.only == "h5":
        n = update_hold_5d_returns(start_date=args.start_date)
        print("已更新5日收益（影响行数累计）:", n)
    else:
        out = update_all_returns(start_date=args.start_date)
        print("回填结果:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
