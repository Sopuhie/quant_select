"""
每日选股：从本地 stock_daily_kline 读 K 线、截面清洗，LightGBM + XGBoost Ranker 融合打分（若存在 xgb_model.pkl），写入 daily_predictions / daily_selections。
默认股票池为「本地库中截至预测日有足够历史的代码」；可加 --online-pool 用 AkShare 成分后再与本地求交。

用法:
  python run_daily.py
  python run_daily.py --trade-date 2026-05-08
  python run_daily.py --workers 8   # 默认覆盖当日已有选股记录
  python run_daily.py --skip-if-exists   # 若当日已有记录则跳过
  python run_daily.py --max-stocks 500   # 可选：限制最多参与预测的股票数
  python run_daily.py --online-pool --pool hs300   # 成分在线拉取，K 线仍只读库
  python run_daily.py --only-data   # 探测本地可预测股票数量
  python run_daily.py --include-300 --include-688   # 默认不传则从池中剔除 300/301/688 开头股票

选股写入成功后，若 config 中启用钉钉，将自动调用推送（与 Streamlit「系统设置」一致）。
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
    MIN_HISTORY_BARS,
    MODEL_PATH,
    PREDICT_FETCH_WORKERS,
    STOCK_POOL,
    TOP_N_SELECTION,
)
from src.data_fetcher import get_stock_pool
from src.database import (
    delete_daily_outputs_for_trade_date,
    fetch_stock_daily_bars_until,
    init_db,
    insert_daily_predictions,
    insert_daily_selections,
    list_predict_universe_from_kline,
    selection_exists_for_date,
    stock_codes_with_local_bars,
)
from src.dingtalk_notifier import maybe_push_daily_selections
from src.factor_calculator import (
    clean_cross_sectional_features,
    compute_factors_for_history,
    normalize_industry_label,
)
from src.model_trainer import load_model, load_xgb_ranker_optional
from src.predictor import (
    analyze_stock_reasons,
    blend_ranker_scores,
    feature_importances_aligned,
    filter_predictions,
)
from src.utils import get_last_trading_date, is_a_share_trading_day


def _normalize_stock_code(code: str) -> str:
    return str(code).strip().zfill(6)


def _board_allowed(code: str, *, include_300: bool, include_688: bool) -> bool:
    """未勾选对应 include 时剔除创业板常见前缀 300/301 与科创板 688。"""
    c = _normalize_stock_code(code)
    if not include_300 and (c.startswith("300") or c.startswith("301")):
        return False
    if not include_688 and c.startswith("688"):
        return False
    return True


def _fetch_one_predict_row(
    code: str,
    name: str,
    trade_date: str,
) -> dict[str, float | str] | None:
    """单只股票：从本地 stock_daily_kline 读 K 线并生成当日因子行（供线程池并发调用）。"""
    df_hist = fetch_stock_daily_bars_until(code, trade_date)
    if df_hist.empty or len(df_hist) < MIN_HISTORY_BARS:
        return None

    df_today = df_hist[df_hist["date"] <= trade_date].reset_index(drop=True)
    if df_today.empty:
        return None

    factors = compute_factors_for_history(df_today)
    if factors.empty:
        return None

    last_idx = len(df_today) - 1
    actual_last = pd.to_datetime(
        df_today.iloc[last_idx]["date"], errors="coerce"
    )
    if pd.isna(actual_last):
        return None
    actual_last_str = actual_last.strftime("%Y-%m-%d")
    td = str(trade_date).strip()[:10]
    # 本地最新一根必须与预测日对齐，避免用停牌前旧 K 线参与截面排序
    if actual_last_str != td:
        return None

    last_row = factors.iloc[last_idx]

    if last_row[list(FEATURE_COLUMNS)].isna().any():
        return None

    row_dict: dict[str, float | str] = {c: float(last_row[c]) for c in FEATURE_COLUMNS}
    row_dict["trade_date"] = td
    row_dict["stock_code"] = code
    row_dict["stock_name"] = name
    if "industry" in df_today.columns:
        row_dict["industry"] = normalize_industry_label(df_today.iloc[last_idx]["industry"])
    else:
        row_dict["industry"] = normalize_industry_label(None)
    row_dict["close_price"] = float(df_today.iloc[last_idx]["close"])

    if len(df_today) >= 2:
        c_prev = float(df_today.iloc[-2]["close"])
        c_now = float(df_today.iloc[-1]["close"])
        row_dict["pct_prev_day"] = (c_now / c_prev - 1.0) if c_prev > 0 else 0.0
    else:
        row_dict["pct_prev_day"] = 0.0

    return row_dict


def run_only_data_probe(
    max_stocks: int | None,
    pool_type: str,
    *,
    max_workers: int | None = None,
    verbose: bool = True,
    use_online_pool: bool = False,
) -> None:
    """探测本地 K 线是否足够做预测（不写入选股表）。"""
    _ = (pool_type, max_workers)  # 保留 CLI 签名兼容；探测仅统计本地库
    print("📊 任务：本地行情探测（不写入选股表 / daily_predictions）...")
    td = get_last_trading_date()
    pairs = list_predict_universe_from_kline(
        td,
        min_bars=MIN_HISTORY_BARS,
        max_count=max_stocks,
    )
    if not pairs:
        print(
            f"❌ 截止 {td}，本地 stock_daily_kline 中没有满足 ≥{MIN_HISTORY_BARS} "
            "根 K 线的股票。请先运行 scripts/update_local_data.py。"
        )
        raise SystemExit(1)
    lim_note = "全库" if max_stocks is None else f"至多 {max_stocks} 只"
    print(f"✅ 截止 {td}，{lim_note} 约有 {len(pairs)} 只股票满足本地 K 线长度。")
    if verbose and len(pairs) <= 10:
        for c, n in pairs:
            nb = len(fetch_stock_daily_bars_until(c, td))
            print(f"   {c} {n[:16]} … 共 {nb} 根（截至 {td}）")
    if use_online_pool:
        print(
            "提示: 已配置 --online-pool 时真实选股会尝试在线成分池并与本地求交；"
            "本次探测仍仅统计本地可算因子股票数。"
        )


def predict_daily(
    trade_date: str,
    max_stocks: int | None,
    pool_type: str,
    force: bool = False,
    *,
    max_workers: int | None = None,
    verbose: bool = True,
    use_online_pool: bool = False,
    include_300: bool = False,
    include_688: bool = False,
    skip_dingtalk: bool = False,
) -> None:
    print(f"开始执行 {trade_date} 每日预测选股流程...")
    # 默认覆盖写入；仅当 force=False（命令行 --skip-if-exists）时跳过已存在记录
    if selection_exists_for_date(trade_date) and not force:
        print(f"提示: {trade_date} 选股记录已存在，跳过（已使用 --skip-if-exists）。")
        return

    # 1. 载入模型
    if not MODEL_PATH.exists():
        print(f"错误: 找不到模型文件 {MODEL_PATH}，请先运行 train_model.py")
        sys.exit(1)
    lgb_model = load_model(MODEL_PATH)
    xgb_model = load_xgb_ranker_optional()

    # 2. 股票池：默认完全来自本地库；--online-pool 时用在线成分与本地可算股票求交
    if use_online_pool:
        try:
            pool_cap = None if max_stocks is None else max(max_stocks * 3, max_stocks)
            pairs_online = get_stock_pool(
                as_of_date=trade_date,
                pool_type=pool_type,
                max_count=pool_cap,
            )
        except Exception as exc:
            print(f"警告: 在线股票池获取失败（{exc}），改用纯本地股票列表。")
            pairs_online = []
        eligible = stock_codes_with_local_bars(trade_date, MIN_HISTORY_BARS)
        pairs = [
            (str(c).zfill(6), str(n).strip())
            for c, n in pairs_online
            if str(c).strip().zfill(6) in eligible
        ]
        if max_stocks is not None:
            pairs = pairs[: int(max_stocks)]
        if not pairs:
            pairs = list_predict_universe_from_kline(
                trade_date,
                min_bars=MIN_HISTORY_BARS,
                max_count=max_stocks,
            )
            print("提示: 在线池与本地 K 线无交集，已退回本地股票列表。")
    else:
        pairs = list_predict_universe_from_kline(
            trade_date,
            min_bars=MIN_HISTORY_BARS,
            max_count=max_stocks,
        )

    if not pairs:
        print(
            "错误: 本地 stock_daily_kline 中没有足够历史的股票。"
            "请先运行 scripts/update_local_data.py，或使用 --online-pool（仍需本地有对应 K 线）。"
        )
        sys.exit(1)

    before_board = len(pairs)
    pairs = [
        (c, n)
        for c, n in pairs
        if _board_allowed(c, include_300=include_300, include_688=include_688)
    ]
    if verbose and before_board != len(pairs):
        print(
            f"板块过滤：剔除后剩余 {len(pairs)}/{before_board} 只 "
            f"（含创业板300/301={include_300}, 含科创板688={include_688}）。"
        )
    if not pairs:
        print("错误: 板块过滤后没有剩余候选股票，请调整 --include-300 / --include-688。")
        sys.exit(1)

    workers = int(max_workers if max_workers is not None else PREDICT_FETCH_WORKERS)
    workers = max(1, min(workers, 32))
    src = "在线成分∩本地" if use_online_pool else "本地库"
    print(
        f"从本地库读取 {len(pairs)} 只股票的 K 线并计算因子（股票池来源: {src}，并发 {workers}）..."
    )

    # 3. 并发读取本地 K 线并计算因子（模型打分仍在主线程）
    rows: list[dict[str, float | str]] = []
    total = len(pairs)
    done = 0

    rows_by_code: dict[str, dict[str, float | str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_fetch_one_predict_row, c, n, trade_date): (c, n)
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
                rows_by_code[_normalize_stock_code(str(r["stock_code"]))] = r

    # 按股票池顺序组装，避免仅用 as_completed 顺序导致 stable sort 下并列分时 TopN 抖动
    rows = [rows_by_code[c] for c, _n in pairs if c in rows_by_code]

    if not rows:
        print("错误: 未能获取到任何有效的股票因子特征数据。")
        sys.exit(1)

    # 转化为 DataFrame（已按 pairs 顺序，便于与训练/排查对齐）
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

    sc = filtered_df["stock_code"].map(_normalize_stock_code)
    mask = pd.Series(True, index=filtered_df.index)
    if not include_300:
        mask &= ~(sc.str.startswith("300") | sc.str.startswith("301"))
    if not include_688:
        mask &= ~sc.str.startswith("688")
    filtered_df = filtered_df.loc[mask].copy()
    if filtered_df.empty:
        print("警告: 板块过滤后没有剩余的候选股票。")
        sys.exit(1)

    # 6. LightGBM + XGBoost Ranker 打分并截面秩融合（无 xgb 文件时等同仅 LGB）
    X = filtered_df[FEATURE_COLUMNS].astype(np.float64)
    lgb_scores = lgb_model.predict(X.values)
    xgb_scores = xgb_model.predict(X.values) if xgb_model is not None else None

    filtered_df = filtered_df.copy()
    filtered_df["score"] = blend_ranker_scores(lgb_scores, xgb_scores)

    # 全市场排序并记录预测数据（分数相同时按代码稳定次序，避免重复运行 Top 边界跳动）
    filtered_df = filtered_df.sort_values(
        ["score", "stock_code"],
        ascending=[False, True],
    ).reset_index(drop=True)
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

    imp_vec = feature_importances_aligned(lgb_model)
    selection_rows: list[dict[str, object]] = []
    for _, row in selections.iterrows():
        selection_rows.append(
            {
                "trade_date": row["trade_date"],
                "stock_code": row["stock_code"],
                "stock_name": row["stock_name"],
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "close_price": float(row["close_price"]),
                "next_day_return": None,
                "hold_5d_return": None,
                "selection_reason": analyze_stock_reasons(row, imp_vec, FEATURE_COLUMNS),
            }
        )
    insert_daily_selections(selection_rows)

    print(f"今日选股完成！{trade_date} 推荐 Top{TOP_N_SELECTION} 为:")
    for r in selection_rows:
        print(
            f"  Rank {r['rank']}: {r['stock_code']} {r['stock_name']} "
            f"(分数: {r['score']:.4f}, 收盘价: {r['close_price']})"
        )

    if not skip_dingtalk:
        try:
            maybe_push_daily_selections(trade_date)
        except Exception as exc:
            print(f"钉钉推送环节异常（选股数据已写入）: {exc}")
    else:
        print("已跳过钉钉推送（--skip-dingtalk，由上游流水线统一推送）。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="量化每日选股打分预测")
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="预测日期 (YYYY-MM-DD)，留空则默认最近交易日",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="最多参与预测的股票数量；默认不限制，使用本地库内全部满足 K 线长度的股票",
    )
    parser.add_argument("--pool", type=str, default=STOCK_POOL)
    parser.add_argument(
        "--force",
        action="store_true",
        help=argparse.SUPPRESS,
    )  # 兼容旧脚本；默认已始终覆盖，无需再传
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="若当日选股记录已存在则跳过（默认每次均覆盖写入）",
    )
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
    parser.add_argument(
        "--only-data",
        action="store_true",
        help="仅探测本地 K 线是否足够预测，不执行选股写入",
    )
    parser.add_argument(
        "--online-pool",
        action="store_true",
        help="股票池使用 AkShare 成分（需网络），K 线仍仅从本地 stock_daily_kline 读取",
    )
    parser.add_argument(
        "--include-300",
        dest="include_300",
        action="store_true",
        help="包含创业板代码（300、301 开头）；不传则从选股池中剔除",
    )
    parser.add_argument(
        "--include-688",
        dest="include_688",
        action="store_true",
        help="包含代码以 688 开头的股票（科创板）；不传则从选股池中剔除",
    )
    parser.add_argument(
        "--skip-dingtalk",
        action="store_true",
        help="选股完成后不推送钉钉（供一键流水线等在上游步骤结束后再统一推送）",
    )
    args = parser.parse_args()

    init_db()

    if args.only_data:
        run_only_data_probe(
            max_stocks=args.max_stocks,
            pool_type=args.pool,
            max_workers=args.workers,
            verbose=not args.quiet,
            use_online_pool=args.online_pool,
        )
        return

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

    if not is_a_share_trading_day(target_date):
        print(
            f"警告: {target_date} 不是 A 股交易日（或交易日历不可用时的周一至周五近似），"
            "已跳过选股写入，未修改数据库。",
            flush=True,
        )
        # 供一键流水线识别：避免末尾按旧 MAX(trade_date) 误推钉钉
        print("QUANT_RUN_DAILY_SKIPPED=non_trading_day", flush=True)
        sys.exit(0)

    predict_daily(
        trade_date=target_date,
        max_stocks=args.max_stocks,
        pool_type=args.pool,
        force=not args.skip_if_exists,
        max_workers=args.workers,
        verbose=not args.quiet,
        use_online_pool=args.online_pool,
        include_300=args.include_300,
        include_688=args.include_688,
        skip_dingtalk=args.skip_dingtalk,
    )


if __name__ == "__main__":
    main()
