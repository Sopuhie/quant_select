"""
手动训练 LightGBM 模型并写入 models/lgb_model.pkl 与 model_versions 表。

用法（在 quant_select 目录下）:
  python train_model.py --train-end-date 2024-12-31
  python train_model.py --train-end-date 2024-12-31 --max-stocks 200 --version v1.0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MAX_STOCKS_UNIVERSE
from src.data_fetcher import get_stock_pool
from src.database import init_db
from src.model_trainer import train_and_register


def main() -> None:
    parser = argparse.ArgumentParser(description="训练选股 LightGBM 模型")
    parser.add_argument(
        "--train-end-date",
        type=str,
        required=True,
        help="训练样本截止日期（含）YYYY-MM-DD",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=MAX_STOCKS_UNIVERSE,
        help="参与训练的股票数量上限",
    )
    parser.add_argument("--version", type=str, default=None, help="模型版本号")
    args = parser.parse_args()

    init_db()
    pairs = get_stock_pool(as_of_date=args.train_end_date, max_count=args.max_stocks)
    if not pairs:
        raise SystemExit("无法获取股票列表，请检查 AkShare 与网络。")
    _model, info = train_and_register(
        stock_pairs=pairs,
        train_end_date=args.train_end_date,
        version=args.version,
    )
    print("训练完成:", info)


if __name__ == "__main__":
    main()
