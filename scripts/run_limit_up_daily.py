"""
打板选股（涨停接力）：独立流水线，不影响 run_daily / run_short_daily。

用法:
  python scripts/run_limit_up_daily.py
  python scripts/run_limit_up_daily.py --trade-date 2026-05-21
  python scripts/run_limit_up_daily.py --force
  python scripts/run_limit_up_daily.py --include-300 --include-688
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.limit_up_hit.runner import run_limit_up_daily_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="打板选股（涨停接力，纯日线模拟）")
    parser.add_argument("--trade-date", type=str, default=None, help="信号日 YYYY-MM-DD")
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖当日已有 luh_daily_selections 记录",
    )
    parser.add_argument("--top-n", type=int, default=None, help="输出 TopN，默认见 QUANT_LUH_TOP_N")
    parser.add_argument(
        "--max-scan-stocks",
        type=int,
        default=None,
        help="限制扫描截面行数（调试用）",
    )
    parser.add_argument("--no-json", action="store_true", help="不写入 limit_up_today.json")
    parser.add_argument(
        "--include-300",
        dest="include_300",
        action="store_true",
        help="包含创业板（300/301）",
    )
    parser.add_argument(
        "--include-688",
        dest="include_688",
        action="store_true",
        help="包含科创板（688）",
    )
    args = parser.parse_args()

    summary = run_limit_up_daily_pipeline(
        args.trade_date,
        force=bool(args.force),
        top_n=args.top_n,
        max_scan_stocks=args.max_scan_stocks,
        include_300=bool(args.include_300),
        include_688=bool(args.include_688),
        write_json=not args.no_json,
    )

    out = json.dumps(summary, ensure_ascii=False, indent=2)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(out)

    if summary.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
