"""
短线规则选股（持有 1 个交易日）：独立流水线，不影响 run_daily.py。

用法:
  python scripts/run_short_daily.py
  python scripts/run_short_daily.py --trade-date 2026-05-19
  python scripts/run_short_daily.py --force
  python scripts/run_short_daily.py --top-n 8
  python scripts/run_short_daily.py --max-scan-stocks 800
  python scripts/run_short_daily.py --skip-dingtalk
  python scripts/run_short_daily.py --include-300 --include-688
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.short_term import db as _short_db
from src.short_term.runner import run_short_daily_pipeline


def main() -> int:
    # 便于确认加载的是当前工程内的 db 模块（排查旧代码/缓存）
    print(f"[short_term.db] {_short_db.__file__}", flush=True)

    parser = argparse.ArgumentParser(description="短线规则选股（1 日持有）")
    parser.add_argument("--trade-date", type=str, default=None, help="信号日 YYYY-MM-DD")
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖当日已有 short_daily_selections 记录",
    )
    parser.add_argument("--top-n", type=int, default=None, help="输出 TopN，默认见 QUANT_SHORT_TOP_N")
    parser.add_argument(
        "--max-scan-stocks",
        type=int,
        default=None,
        help="限制扫描截面行数（调试用，默认全市场当日截面）",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="不写入项目根目录 short_today.json",
    )
    parser.add_argument(
        "--skip-dingtalk",
        action="store_true",
        help="不发送短线选股钉钉消息",
    )
    parser.add_argument(
        "--include-300",
        dest="include_300",
        action="store_true",
        help="包含创业板代码（300、301 开头）；不传则从扫描池中剔除",
    )
    parser.add_argument(
        "--include-688",
        dest="include_688",
        action="store_true",
        help="包含代码以 688 开头的股票（科创板）；不传则从扫描池中剔除",
    )
    args = parser.parse_args()

    summary = run_short_daily_pipeline(
        args.trade_date,
        force=bool(args.force),
        top_n=args.top_n,
        max_scan_stocks=args.max_scan_stocks,
        include_300=bool(args.include_300),
        include_688=bool(args.include_688),
        write_json=not args.no_json,
        skip_dingtalk=bool(args.skip_dingtalk),
    )

    out = json.dumps(summary, ensure_ascii=False, indent=2)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(out)

    if summary.get("error"):
        return 1
    if summary.get("skipped"):
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
