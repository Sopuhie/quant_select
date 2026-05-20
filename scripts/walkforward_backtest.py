"""
Walk-forward 滚动回测：每 N 个交易日重训模型，再对下一段区间回测并拼接净值。

与单次 ``backtest.py`` 的区别：
  - 每窗训练截止日 = 该窗首个信号日前一交易日（样本外）
  - 默认 ``--fast-train --no-catboost`` 以控制耗时
  - 各段净值首尾相接（下段 initial_cash = 上段末 NAV）

用法（项目根目录）:
  python scripts/walkforward_backtest.py --start-date 2025-01-01 --end-date 2025-12-31
  python scripts/walkforward_backtest.py --retrain-every 40 --fast-train
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_backtest_module() -> dict[str, Any]:
    import runpy

    return runpy.run_path(str(ROOT / "scripts" / "backtest.py"))


def _run_train(train_end: str, *, max_stocks: int, fast: bool, no_cat: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "train_model.py"),
        "--train-end-date",
        str(train_end)[:10],
        "--max-stocks",
        str(int(max_stocks)),
    ]
    if fast:
        cmd.append("--fast-train")
    if no_cat:
        cmd.append("--no-catboost")
    print(f"[WF] 训练: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    from src.backtest_universe import (
        build_point_in_time_universe_fn,
        split_trade_dates_windows,
        trading_day_before,
    )
    from src.config import (
        DB_PATH,
        LABEL_HORIZON_DAYS,
        MAX_STOCKS_UNIVERSE,
        MIN_HISTORY_BARS,
        TOP_N_SELECTION,
        get_quant_config_merged,
    )
    from src.database import get_active_model_version

    bt = _load_backtest_module()

    parser = argparse.ArgumentParser(description="Walk-forward 滚动重训 + 回测")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=60,
        help="每多少个**交易日**重训一次（默认 60）",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--holding-days", type=int, default=LABEL_HORIZON_DAYS)
    parser.add_argument("--max-stocks", type=int, default=MAX_STOCKS_UNIVERSE)
    parser.add_argument("--pool", type=str, default="hs300")
    parser.add_argument("--fast-train", action="store_true", default=True)
    parser.add_argument("--full-train", action="store_true", help="关闭 --fast-train")
    parser.add_argument("--no-catboost", action="store_true", default=True)
    parser.add_argument("--with-catboost", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data" / "walkforward_backtest_results.csv"),
    )
    parser.add_argument("--include-300", action="store_true")
    parser.add_argument("--include-688", action="store_true")
    args = parser.parse_args()

    fast_train = bool(args.fast_train) and not args.full_train
    no_cat = bool(args.no_catboost) and not args.with_catboost

    db_lo, db_hi = bt["get_stock_daily_kline_date_bounds"](DB_PATH)
    if db_lo is None or db_hi is None:
        raise SystemExit("本地无 K 线，请先 sync。")

    start_date = bt["_normalize_date"](args.start_date or db_lo)
    end_date = bt["_normalize_date"](args.end_date or db_hi)
    if start_date > end_date:
        raise SystemExit("start-date 不能晚于 end-date。")

    qbt = get_quant_config_merged().get("backtest", {})
    slip = float(qbt.get("slippage_rate", bt["BACKTEST_DEFAULT_SLIPPAGE_RATE"]))
    comm = float(qbt.get("commission_rate", 0.0003))
    stamp = float(qbt.get("stamp_duty_sell_rate", 0.001))

    pairs: list[tuple[str, str]] = []
    try:
        pairs = bt["get_stock_pool"](
            as_of_date=end_date, pool_type=args.pool, max_count=args.max_stocks
        )
    except Exception as exc:
        print(f"[WF] 在线股票池失败: {exc}", flush=True)

    cal_start = (pd.Timestamp(start_date) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    bars = bt["read_stock_daily_kline_range"](cal_start, end_date, DB_PATH)
    if not pairs:
        pairs = bt["_pairs_from_db_bars"](bars, args.max_stocks)
    pairs = [
        (c, n)
        for c, n in pairs
        if bt["_board_allowed"](c, include_300=args.include_300, include_688=args.include_688)
    ]
    if not pairs:
        raise SystemExit("股票池为空。")

    bars = bt["fetch_backtest_data"](
        start_date,
        end_date,
        pairs,
        db_path=DB_PATH,
        max_workers=4,
        online_fallback=False,
    )
    all_days = sorted(bars["date"].unique())
    trade_dates = [d for d in all_days if start_date <= d <= end_date]
    if len(trade_dates) < args.holding_days + 10:
        raise SystemExit("回测交易日过少。")

    windows = split_trade_dates_windows(trade_dates, retrain_every=args.retrain_every)
    print(f"[WF] 共 {len(windows)} 个窗口，retrain_every={args.retrain_every}", flush=True)

    universe_fn = build_point_in_time_universe_fn(
        pairs,
        bars,
        min_history_bars=MIN_HISTORY_BARS,
        max_stocks=args.max_stocks,
        include_300=args.include_300,
        include_688=args.include_688,
    )

    nav_all: list[dict[str, Any]] = []
    cash = float(args.initial_cash)
    segment_rows: list[dict[str, Any]] = []

    for wi, (w_start, w_end) in enumerate(windows):
        train_end = trading_day_before(trade_dates, w_start)
        if not train_end:
            print(f"[WF] 窗口 {wi+1} 跳过（无训练截止日）: {w_start}~{w_end}", flush=True)
            continue

        _run_train(
            train_end,
            max_stocks=args.max_stocks,
            fast=fast_train,
            no_cat=no_cat,
        )
        lgb_m, xgb_m, cat_m = bt["load_active_models"]()
        mv = get_active_model_version()
        print(
            f"[WF] 窗口 {wi+1}/{len(windows)} 回测 {w_start}~{w_end} "
            f"模型 {mv.get('version') if mv else '?'} train_end={train_end}",
            flush=True,
        )

        seg_dates = [d for d in trade_dates if w_start <= d <= w_end]
        engine = bt["BacktestEngine"](
            bars=bars,
            trade_dates=seg_dates,
            lgb_model=lgb_m,
            xgb_model=xgb_m,
            cat_model=cat_m,
            initial_cash=cash,
            holding_days=args.holding_days,
            buy_price="open",
            sell_price="open",
            commission_rate=comm,
            stamp_duty_sell_rate=stamp,
            slippage_rate=slip,
            limit_price_eps_yuan=float(bt["BACKTEST_LIMIT_PRICE_EPS_YUAN"]),
            max_calendar_gap_days=int(bt["BACKTEST_MAX_CALENDAR_GAP_DAYS"]),
            get_universe=universe_fn,
            include_300=args.include_300,
            include_688=args.include_688,
        )
        engine.run()
        if not engine.nav_records:
            continue
        seg_nav = pd.DataFrame(engine.nav_records)
        seg_nav["wf_window"] = wi + 1
        seg_nav["wf_train_end"] = train_end
        nav_all.extend(seg_nav.to_dict("records"))
        cash = float(seg_nav.iloc[-1]["nav"])
        segment_rows.append(
            {
                "window": wi + 1,
                "start": w_start,
                "end": w_end,
                "train_end": train_end,
                "end_nav": cash,
                "fuse_days": int(engine.regime_fuse_days),
            }
        )

    if not nav_all:
        raise SystemExit("Walk-forward 未产生任何净值记录。")

    out_df = pd.DataFrame(nav_all)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[WF] 净值已写入 {out_path.resolve()}", flush=True)

    seg_df = pd.DataFrame(segment_rows)
    print("[WF] 各段摘要:", flush=True)
    print(seg_df.to_string(index=False), flush=True)

    bench = bt["fetch_benchmark_close"](start_date, end_date, bench="000300", db_path=DB_PATH)
    metrics = bt["_compute_metrics"](out_df, bench, [], rf_annual=0.03)
    bt["_print_report"](metrics)


if __name__ == "__main__":
    main()
