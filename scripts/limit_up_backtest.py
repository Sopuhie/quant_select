"""
打板历史滚动回测（T+1 开盘价买入；T+2 起一字续骑，开板日开盘价卖出，不写库）。

用法（项目根目录）::

  python scripts/limit_up_backtest.py
  python scripts/limit_up_backtest.py --start-date 2025-01-01 --end-date 2025-06-30
  python scripts/limit_up_backtest.py --include-300 --max-scan-stocks 2000

结果写入 ``data/limit_up_backtest_trades.csv``、
``data/limit_up_backtest_daily.csv``、``data/limit_up_backtest_summary.json``。
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
from src.limit_up_hit.backtest import run_limit_up_rolling_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="打板历史滚动回测")
    parser.add_argument("--start-date", type=str, default=None, help="信号日起始 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="K 线截止 YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=None, help="每日 Top N")
    parser.add_argument(
        "--max-scan-stocks",
        type=int,
        default=None,
        help="单日扫描股票上限（加速用）",
    )
    parser.add_argument("--include-300", action="store_true", help="包含创业板 300/301")
    parser.add_argument("--include-688", action="store_true", help="包含科创板 688")
    parser.add_argument("--legacy", action="store_true", help="使用旧版回测规则（无优化过滤）")
    parser.add_argument("--quiet", action="store_true", help="减少进度输出")
    args = parser.parse_args()

    init_db(DB_PATH)
    with get_connection(DB_PATH) as conn:
        result = run_limit_up_rolling_backtest(
            conn,
            args.start_date,
            args.end_date,
            top_n=args.top_n,
            include_300=args.include_300,
            include_688=args.include_688,
            max_scan_stocks=args.max_scan_stocks,
            legacy=bool(args.legacy),
            verbose=not args.quiet,
        )

    s = result["summary"]
    print(
        "摘要:",
        {
            "区间": f"{s.get('start_date')} ~ {s.get('scan_end_date')}（K线至 {s.get('end_date')}）",
            "复利收益_pct": s.get("cum_return_pct"),
            "胜率": s.get("win_rate"),
            "单笔均盈_pct": s.get("avg_pnl_pct"),
            "信号日": s.get("signal_days"),
            "平仓笔数": s.get("closed_trades"),
            "最大回撤_pct": s.get("max_drawdown_pct"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
