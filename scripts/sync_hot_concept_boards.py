"""
同步东方财富热门概念板块名称 + 成份股 → data/hot_sectors.json、board_stocks.json、SQLite。

用法（项目根目录）:
  python scripts/sync_hot_concept_boards.py
  python scripts/sync_hot_concept_boards.py --force
  python scripts/sync_hot_concept_boards.py --top-n 20 --no-constituents
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.concept_board_sync import ensure_hot_sectors_for_trade_date
from src.database import init_db, sync_concept_boards_from_json
from src.config import DB_PATH


def main() -> None:
    p = argparse.ArgumentParser(description="同步东方财富热门概念与成份股")
    p.add_argument("--top-n", type=int, default=15, help="热门概念板块数量")
    p.add_argument("--force", action="store_true", help="强制刷新（忽略缓存日期）")
    p.add_argument(
        "--no-constituents",
        action="store_true",
        help="仅更新热门题材名称，不同步成份股",
    )
    args = p.parse_args()
    init_db(DB_PATH)
    stats = ensure_hot_sectors_for_trade_date(
        top_n=args.top_n,
        sync_constituents=not args.no_constituents,
        force_refresh=args.force,
        verbose=True,
    )
    n = sync_concept_boards_from_json()
    print(
        f"完成: 交易日={stats.get('trade_date')} "
        f"题材刷新={stats.get('tags_refreshed')} "
        f"板块={stats.get('boards_synced')} "
        f"成份股映射={stats.get('stock_mappings')} "
        f"SQLite行={n}",
        flush=True,
    )
    if stats.get("error"):
        print(f"警告: {stats['error']}", flush=True)
    if not stats.get("tags_refreshed") and not stats.get("boards_synced"):
        print(
            "提示: 未从东方财富拉取到新数据，已沿用本地缓存；"
            "请检查网络/代理或稍后重试。",
            flush=True,
        )


if __name__ == "__main__":
    main()
