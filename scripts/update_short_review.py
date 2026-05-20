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
from src.short_term.db import ensure_short_term_tables, refresh_short_review_prices


def main() -> int:
    parser = argparse.ArgumentParser(description="回填短线 T1/T2 复盘价")
    parser.add_argument("--trade-date", type=str, default=None, help="仅更新该信号日")
    args = parser.parse_args()

    init_db(DB_PATH)
    with get_connection(DB_PATH) as conn:
        ensure_short_term_tables(conn)
        n = refresh_short_review_prices(conn, args.trade_date)
    print(f"已更新 {n} 条短线记录的 T1/T2 开收盘价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
