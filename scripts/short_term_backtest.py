"""
短线规则历史滚动回测（纯日线模拟，不写 short_daily_selections）。

对每个信号日重跑 ``ShortTermRuleStrategy.scan``，再用 ``evaluate_daily_exit`` 模拟
T 收盘买 / T+1 止损 / T+N 收盘卖，汇总胜率与复利收益。

用法（项目根目录）::

  python scripts/short_term_backtest.py
  python scripts/short_term_backtest.py --start-date 2025-01-01 --end-date 2025-06-30
  python scripts/short_term_backtest.py --include-300 --max-scan-stocks 2000

结果写入 ``data/short_term_backtest_trades.csv``、
``data/short_term_backtest_daily.csv``、``data/short_term_backtest_summary.json``。
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
from src.short_term.backtest import run_short_term_rolling_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="短线规则历史滚动回测")
    parser.add_argument("--start-date", type=str, default=None, help="信号日起始 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="K 线截止 YYYY-MM-DD（默认可扫描至库内最新）")
    parser.add_argument("--top-n", type=int, default=None, help="每日 Top N（默认 QUANT_SHORT_TOP_N）")
    parser.add_argument(
        "--max-scan-stocks",
        type=int,
        default=None,
        help="单日扫描股票上限（按库内顺序截取，加速用）",
    )
    parser.add_argument("--include-300", action="store_true", help="包含创业板 300/301")
    parser.add_argument("--include-688", action="store_true", help="包含科创板 688")
    parser.add_argument(
        "--sell-offset",
        type=int,
        default=None,
        choices=(1, 2),
        help="未触发止损时的平仓日偏移（1=T+1 收盘，2=T+2 收盘）",
    )
    parser.add_argument("--quiet", action="store_true", help="减少进度输出")
    args = parser.parse_args()

    init_db(DB_PATH)
    with get_connection(DB_PATH) as conn:
        result = run_short_term_rolling_backtest(
            conn,
            args.start_date,
            args.end_date,
            top_n=args.top_n,
            include_300=args.include_300,
            include_688=args.include_688,
            max_scan_stocks=args.max_scan_stocks,
            sell_offset=args.sell_offset,
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
