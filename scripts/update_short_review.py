"""
回填短线选股记录的 T1/T2 开收盘价（本地 K 线就绪后执行）。

用法:
  python scripts/update_short_review.py
  python scripts/update_short_review.py --trade-date 2026-05-19
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DB_PATH
from src.database import get_connection, init_db
from src.short_term.db import ensure_short_term_tables
from src.short_term.execution import refresh_holding_short_orders
from src.short_term.review_prices import auto_fill_review_prices


def main() -> int:
    parser = argparse.ArgumentParser(
        description="回填短线 T1/T2 复盘价，并重新评估 HOLDING 订单的止损/平仓"
    )
    parser.add_argument("--trade-date", type=str, default=None, help="仅更新该信号日")
    args = parser.parse_args()

    init_db(DB_PATH)
    with get_connection(DB_PATH) as conn:
        ensure_short_term_tables(conn)
        fill = auto_fill_review_prices(conn, args.trade_date)
        h = refresh_holding_short_orders(conn, args.trade_date)
    print(
        f"已回填 T1/T2：处理 {fill.get('dates', 0)} 个信号日、"
        f"更新 {fill.get('rows', 0)} 条；重新评估 {h} 笔 HOLDING 订单。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
