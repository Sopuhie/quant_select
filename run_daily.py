"""
每日选股：拉行情、截面清洗、LightGBM 打分，写入 daily_predictions / daily_selections。

用法:
  python run_daily.py
  python run_daily.py --trade-date 2026-05-08
  python run_daily.py --force --max-stocks 300 --pool hs300 --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MAX_STOCKS_UNIVERSE,
    MODEL_PATH,
    PREDICT_FETCH_WORKERS,
    STOCK_POOL,
    TOP_N_SELECTION,
)
from src.data_fetcher import fetch_daily_hist, get_stock_pool
from src.database import (
    delete_daily_outputs_for_trade_date,
    init_db,
    insert_daily_predictions,
    insert_daily_selections,
    selection_exists_for_date,
)
from src.factor_calculator import clean_cross_sectional_features, compute_factors_for_history
from src.model_trainer import load_model
from src.predictor import filter_predictions
from src.utils import get_last_trading_date


def _fetch_one_predict_row(
    code: str,
    name: str,
    trade_date: str,
    end_compact: str,
) -> dict[str, float | str] | None:
    """单只股票：拉 K 线并生成当日因子行（供线程池并发调用）。"""
    df_hist = fetch_daily_hist(code, start_date="20230101", end_date=end_compact)
    if df_hist.empty or len(df_hist) < 60:
        return None

    df_today = df_hist[df_hist["date"] <= trade_date].reset_index(drop=True)
    if df_today.empty:
        return None

    factors = compute_factors_for_history(df_today)
    if factors.empty:
        return None

    last_idx = len(df_today) - 1
    last_row = factors.iloc[last_idx]

    if last_row[list(FEATURE_COLUMNS)].isna().any():
        return None

    row_dict: dict[str, float | str] = {c: float(last_row[c]) for c in FEATURE_COLUMNS}
    row_dict["trade_date"] = trade_date
    row_dict["stock_code"] = code
    row_dict["stock_name"] = name
    row_dict["close_price"] = float(df_today.iloc[last_idx]["close"])

    if len(df_today) >= 2:
        c_prev = float(df_today.iloc[-2]["close"])
        c_now = float(df_today.iloc[-1]["close"])
        row_dict["pct_prev_day"] = (c_now / c_prev - 1.0) if c_prev > 0 else 0.0
    else:
        row_dict["pct_prev_day"] = 0.0

    return row_dict


def predict_daily(
    trade_date: str,
    max_stocks: int,
    pool_type: str,
    force: bool = False,
    *,
    max_workers: int | None = None,
    verbose: bool = True,
) -> None:
    print(f"开始执行 {trade_date} 每日预测选股流程...")
    # 检查数据库是否已存在该日记录
    if selection_exists_for_date(trade_date) and not force:
        print(
            f"提示: {trade_date} 选股记录已存在，跳过。若要强制覆盖，请使用 --force 参数。"
        )
        return

    # 1. 载入模型
    if not MODEL_PATH.exists():
        print(f"错误: 找不到模型文件 {MODEL_PATH}，请先运行 train_model.py")
        sys.exit(1)
    model = load_model(MODEL_PATH)

    # 2. 获取股票池
    pairs = get_stock_pool(as_of_date=trade_date, pool_type=pool_type, max_count=max_stocks)
    if not pairs:
        print("错误: 股票池为空，请检查网络或参数设置。")
        sys.exit(1)

    workers = int(max_workers if max_workers is not None else PREDICT_FETCH_WORKERS)
    workers = max(1, min(workers, 32))
    print(
        f"正在获取 {len(pairs)} 只股票的历史 K 线并计算因子（并发 {workers}）..."
    )

    # 3. 并发拉取 K 线并计算因子切片（模型打分仍在主线程）
    rows: list[dict[str, float | str]] = []
    total = len(pairs)
    done = 0

    end_compact = trade_date.replace("-", "")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_fetch_one_predict_row, c, n, trade_date, end_compact): (c, n)
            for c, n in pairs
        }
        for fut in as_completed(futs):
            done += 1
            if verbose and (done % 50 == 0 or done == total):
                print(f"  进度: {done}/{total}", flush=True)
            try:
                r = fut.result()
            except Exception:
                continue
            if r:
                rows.append(r)

    if not rows:
        print("错误: 未能获取到任何有效的股票因子特征数据。")
        sys.exit(1)

    # 转化为 DataFrame
    feat_df = pd.DataFrame(rows)

    # 4. 【核心升级】执行多因子截面去极值与 Z-Score 标准化清洗
    # 构造临时的 'date' 列以适配 clean_cross_sectional_features 的按日期分组逻辑
    feat_df["date"] = feat_df["trade_date"]
    feat_df = clean_cross_sectional_features(feat_df)
    feat_df = feat_df.drop(columns=["date"])

    # 5. 过滤掉 ST 以及无法建仓的股票（如上一日已接近涨跌停）
    filtered_df = filter_predictions(feat_df)
    if filtered_df.empty:
        print("警告: 过滤后没有剩余的候选股票。")
        sys.exit(1)

    # 6. 使用 LightGBM 模型进行打分预测
    X = filtered_df[FEATURE_COLUMNS].astype(np.float64)
    scores = model.predict(X.values)

    filtered_df = filtered_df.copy()
    filtered_df["score"] = scores.astype(float)

    # 全市场排序并记录预测数据
    filtered_df = filtered_df.sort_values("score", ascending=False).reset_index(drop=True)
    filtered_df["rank_in_market"] = range(1, len(filtered_df) + 1)

    # 清理当天旧数据，防止重复写入
    delete_daily_outputs_for_trade_date(trade_date)

    # 7. 将全市场预测结果写入 daily_predictions 表
    predict_rows = filtered_df[
        ["trade_date", "stock_code", "stock_name", "score", "rank_in_market"]
    ].to_dict("records")
    insert_daily_predictions(predict_rows)
    print(f"已成功将全市场 {len(predict_rows)} 只股票打分数据写入 daily_predictions。")

    # 8. 筛选前 TOP_N_SELECTION（默认 3 只）作为今日推荐，写入 daily_selections 表
    selections = filtered_df.head(TOP_N_SELECTION).copy()
    selections["rank"] = range(1, len(selections) + 1)
    selections["next_day_return"] = None
    selections["hold_5d_return"] = None

    selection_rows = selections[
        [
            "trade_date",
            "stock_code",
            "stock_name",
            "rank",
            "score",
            "close_price",
            "next_day_return",
            "hold_5d_return",
        ]
    ].to_dict("records")
    insert_daily_selections(selection_rows)

    print(f"🎉 今日选股完成！{trade_date} 推荐 Top{TOP_N_SELECTION} 为:")
    for r in selection_rows:
        print(
            f"  Rank {r['rank']}: {r['stock_code']} {r['stock_name']} "
            f"(分数: {r['score']:.4f}, 收盘价: {r['close_price']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="量化每日选股打分预测")
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="预测日期 (YYYY-MM-DD)，留空则默认最近交易日",
    )
    parser.add_argument("--max-stocks", type=int, default=MAX_STOCKS_UNIVERSE)
    parser.add_argument("--pool", type=str, default=STOCK_POOL)
    parser.add_argument("--force", action="store_true", help="强制覆盖已有的选股数据")
    parser.add_argument(
        "--workers",
        type=int,
        default=PREDICT_FETCH_WORKERS,
        help="并发拉取 K 线的线程数（默认读取配置 PREDICT_FETCH_WORKERS，与 start.cmd 一致）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印拉取进度",
    )
    args = parser.parse_args()

    init_db()

    target_date = args.trade_date
    if not target_date:
        # 获取当前时间，如果在 15:00 之前，则使用前一交易日作为最新预测日
        now = datetime.now()
        if now.hour < 15:
            target_date = get_last_trading_date()
        else:
            target_date = now.strftime("%Y-%m-%d")
            # 简单校验当前是否为交易日（非周末），如果周末则退回上一个交易日
            if now.weekday() >= 5:
                target_date = get_last_trading_date()

    predict_daily(
        trade_date=target_date,
        max_stocks=args.max_stocks,
        pool_type=args.pool,
        force=args.force,
        max_workers=args.workers,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
