"""
使用本地 SQLite（stock_daily_kline）中的日线计算因子并重训 LightGBM LambdaRank。
不依赖在线行情拉取；预测侧仍使用 factor_calculator 中同一套因子定义。

用法（在 quant_select 目录下）:
  python train_model.py
  python train_model.py --train-end-date 2024-12-31
  python train_model.py --tune                    # Optuna 调参并写入 models/best_params.json
  python train_model.py --tune --tune-trials 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import (
    BEST_LGB_PARAMS_JSON,
    FEATURE_COLUMNS,
    LABEL_HORIZON_DAYS,
    MIN_HISTORY_BARS,
    MODEL_PATH,
)
from src.database import init_db, register_model_version
from src.factor_calculator import (
    DEFAULT_INDUSTRY_LABEL,
    clean_cross_sectional_features,
    compute_factors_for_history,
    label_forward_return,
    normalize_industry_column,
)
from src.model_trainer import (
    _json_safe,
    load_ranker_params_json,
    merge_ranker_params,
    optuna_tune_lgbm_ranker,
    save_model,
    save_ranker_params_json,
    train_lgbm_ranker,
)


def _load_local_kline_panel(
    *,
    train_end_date: str | None,
    max_stocks: int | None,
    verbose: bool,
) -> pd.DataFrame:
    """从 stock_daily_kline 构建训练面板（含 label_ret）。"""
    import sqlite3

    from src.config import DB_PATH

    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    sql = """
        SELECT date, stock_code, stock_name, open, high, low, close, volume, industry, market_cap
        FROM stock_daily_kline
        ORDER BY stock_code, date
    """
    raw_df = pd.read_sql_query(sql, conn)
    conn.close()

    if raw_df.empty:
        raise SystemExit(
            "本地数据库中没有 K 线数据。请先运行「本地行情同步」："
            "python scripts/update_local_data.py"
        )

    raw_df["date"] = raw_df["date"].astype(str).str[:10]
    if "industry" not in raw_df.columns:
        raw_df["industry"] = DEFAULT_INDUSTRY_LABEL
    else:
        raw_df["industry"] = normalize_industry_column(raw_df["industry"])
    db_date_min = str(raw_df["date"].min())
    db_date_max = str(raw_df["date"].max())

    if train_end_date:
        te = str(train_end_date)[:10]
        filtered = raw_df[raw_df["date"] <= te]
        if filtered.empty:
            if verbose:
                print(
                    f"[警告] --train-end-date={te} 筛干后无数据（库内日期范围 {db_date_min} ~ {db_date_max}），"
                    f"已改为使用库内全部日期训练。",
                    flush=True,
                )
        else:
            raw_df = filtered

    codes = sorted(raw_df["stock_code"].astype(str).str.zfill(6).unique())
    if max_stocks is not None and max_stocks > 0:
        codes = codes[: int(max_stocks)]
        raw_df = raw_df[
            raw_df["stock_code"].astype(str).str.zfill(6).isin(codes)
        ]

    min_bars = max(MIN_HISTORY_BARS + LABEL_HORIZON_DAYS, 65)

    total_g = len(codes)
    parts: list[pd.DataFrame] = []
    grouped = raw_df.groupby(raw_df["stock_code"].astype(str).str.zfill(6))
    skipped_short = 0

    for gi, (code, group) in enumerate(grouped):
        if verbose and total_g and (gi % 500 == 0 or gi == total_g - 1):
            print(f"[本地因子] {gi + 1}/{total_g} 只股票…", flush=True)
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) < min_bars:
            skipped_short += 1
            continue
        facts = compute_factors_for_history(g)
        meta = g[["date", "stock_code", "stock_name", "industry"]].copy()
        merged = pd.concat(
            [meta.reset_index(drop=True), facts.reset_index(drop=True)],
            axis=1,
        )
        merged["label_ret"] = label_forward_return(g["close"].astype(float)).values
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
            subset=FEATURE_COLUMNS + ["label_ret"]
        )
        if len(merged) > 0:
            parts.append(merged)

    if not parts:
        raise SystemExit(
            "有效训练样本为空。请检查：\n"
            f"  1) stock_daily_kline 是否有数据（当前约 {len(raw_df)} 行，日期 {db_date_min}~{db_date_max}）；\n"
            f"  2) --train-end-date 是否早于库内最早日期（导致误筛空）；\n"
            f"  3) 单股 K 线是否不少于 {min_bars} 根（当前因过短跳过的股票约 {skipped_short} 只）。"
        )

    out = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"[本地因子] 合并后面板行数: {len(out)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 SQLite K 线训练 LightGBM LambdaRank")
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=None,
        help="仅使用该日及之前的样本（YYYY-MM-DD）；默认使用库内全部日期",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="参与训练的股票数量上限（按代码排序截取）；默认不限制，使用库内全部股票",
    )
    parser.add_argument("--version", type=str, default=None, help="写入 model_versions 的版本号")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少进度输出",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="使用 Optuna 搜索超参（最大化验证集平均 Rank IC），结果写入 models/best_params.json",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=20,
        help="Optuna 试验次数（仅 --tune 时生效）",
    )
    args = parser.parse_args()

    verbose = not args.quiet
    panel = _load_local_kline_panel(
        train_end_date=args.train_end_date,
        max_stocks=args.max_stocks,
        verbose=verbose,
    )

    if verbose:
        print(
            f"[截面清洗] 行数 {len(panel)}，分位秩 + 行业内 MAD/Z + 市值残差中性化…",
            flush=True,
        )
    panel = clean_cross_sectional_features(panel)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["label_ret"]
    )

    unique_dates = sorted(panel["date"].astype(str).unique())
    if len(unique_dates) < 30:
        train_df = panel
        val_df = panel
        if verbose:
            print("[划分] 交易日不足 30，训练集与验证集相同（仅拟合，慎用）。", flush=True)
    else:
        split_date = unique_dates[-20]
        train_df = panel[panel["date"].astype(str) < split_date].copy()
        val_df = panel[panel["date"].astype(str) >= split_date].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit("训练或验证子集为空，请放宽 --train-end-date 或增大股票数量。")

    x_tr = train_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_tr = pd.to_numeric(train_df["label_ret"], errors="coerce").to_numpy()
    mask_tr = np.isfinite(x_tr).all(axis=1) & np.isfinite(y_tr)

    x_va = val_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_va = pd.to_numeric(val_df["label_ret"], errors="coerce").to_numpy()
    mask_va = np.isfinite(x_va).all(axis=1) & np.isfinite(y_va)

    train_df = train_df.loc[mask_tr].reset_index(drop=True)
    val_df = val_df.loc[mask_va].reset_index(drop=True)

    if len(train_df) < 500:
        raise SystemExit(f"训练样本过少（{len(train_df)}），请同步更多本地行情或放宽筛选。")

    if args.tune:
        if verbose:
            print(
                f"[Optuna] 开始调参 n_trials={args.tune_trials}（目标：验证集平均 Rank IC）…",
                flush=True,
            )
        best_full = optuna_tune_lgbm_ranker(
            train_df,
            val_df,
            n_trials=max(1, int(args.tune_trials)),
            feature_cols=list(FEATURE_COLUMNS),
            verbose=verbose,
        )
        save_ranker_params_json(best_full)
        if verbose:
            print("[Optuna] 最优参数:", best_full, flush=True)
            print(f"[Optuna] 已写入 {BEST_LGB_PARAMS_JSON}", flush=True)
    else:
        loaded = load_ranker_params_json()
        best_full = merge_ranker_params(loaded) if loaded else merge_ranker_params(None)
        if verbose and loaded:
            print(f"[训练] 使用 {BEST_LGB_PARAMS_JSON} 中的超参数。", flush=True)

    if verbose:
        print(
            f"[LambdaRank] 样本 训练 {len(train_df)} / 验证 {len(val_df)}；"
            f" objective=lambdarank，按 date 分组 …",
            flush=True,
        )

    model, rank_metrics = train_lgbm_ranker(
        train_df,
        val_df,
        params=best_full,
        feature_cols=list(FEATURE_COLUMNS),
        n_estimators=800,
        early_stopping_rounds=50,
        verbose=verbose,
    )

    train_end_register = (
        args.train_end_date
        if args.train_end_date
        else str(unique_dates[-1])
    )
    version = args.version or ("local_" + datetime.now().strftime("%Y%m%d.%H%M"))

    metrics_register = {
        "mean_rank_ic_val": rank_metrics.get("mean_rank_ic_val"),
        "n_train": rank_metrics.get("n_train"),
        "n_val": rank_metrics.get("n_val"),
        "source": "sqlite_stock_daily_kline_lambdarank",
        "last_panel_date": str(unique_dates[-1]),
    }
    for k, v in rank_metrics.items():
        if k not in metrics_register and str(k).startswith("val_ndcg"):
            metrics_register[k] = v

    save_model(model, MODEL_PATH)
    register_model_version(
        version=version,
        train_end_date=str(train_end_register)[:10],
        features=list(FEATURE_COLUMNS),
        metrics=_json_safe(metrics_register),
        set_active=True,
    )

    print(
        "训练完成:",
        {
            "version": version,
            "train_end_date": train_end_register,
            "mean_rank_ic_val": rank_metrics.get("mean_rank_ic_val"),
            "model_path": str(MODEL_PATH),
            "best_params_json": str(BEST_LGB_PARAMS_JSON) if args.tune else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
