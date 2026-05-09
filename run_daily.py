"""
每日选股：拉取行情、打分、写入 daily_selections / daily_predictions；
并尝试回填历史选股的次日与 5 日收益。

用法:
  python run_daily.py
  python run_daily.py --force          # 当日已选仍重新写入
  python run_daily.py --max-stocks 300
定时: Windows 任务计划程序或 Linux cron 指向本脚本。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MAX_STOCKS_UNIVERSE, MODEL_PATH, PREDICT_FETCH_WORKERS
from src.database import init_db
from src.predictor import run_selection_for_latest
from src.return_updater import update_all_returns  # 与 scripts/update_returns.py 同源逻辑


def main() -> None:
    parser = argparse.ArgumentParser(description="每日自动选股")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=MAX_STOCKS_UNIVERSE,
        help="参与打分股票数量上限",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略「该交易日已存在记录」跳过逻辑",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=PREDICT_FETCH_WORKERS,
        help="并发拉取 K 线的线程数（默认读取配置 PREDICT_FETCH_WORKERS）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印拉取进度",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"未找到模型文件 {MODEL_PATH}，请先运行: python train_model.py --train-end-date YYYY-MM-DD"
        )

    init_db()
    result = run_selection_for_latest(
        max_stocks=args.max_stocks,
        skip_if_exists=not args.force,
        fetch_workers=args.workers,
        verbose=not args.quiet,
    )
    print("选股结果:", result)

    if not result.get("skipped"):
        try:
            # 钉钉推送正文仅含 rank / stock_code / stock_name（见 maybe_push_daily_selections）
            from src.dingtalk_notifier import maybe_push_daily_selections

            maybe_push_daily_selections(str(result["trade_date"]))
        except Exception as e:
            print(f"钉钉推送异常（不影响主流程）: {e}")

    ret = update_all_returns()
    print("回填收益:", ret)


if __name__ == "__main__":
    main()
