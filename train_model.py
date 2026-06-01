"""
任务 B：重新训练选股模型 — 实现说明
====================================

本脚本是 Streamlit 控制台「🎯 任务 B：重新训练选股模型」的子进程入口
（``src.config.SCRIPT_TRAIN_MODEL``）。中长线每日选股（任务 C）依赖此处产出的
排序模型与元学习器；**不拉在线行情**，仅读本地 SQLite ``stock_daily_kline``。

触发入口
--------
1. **Web 控制台**（``app.py`` → 系统控制台 Tab → 任务 B）
   - 点击「开始模型训练」后，父进程执行::

         python train_model.py --fast-train --no-catboost [--train-end-date YYYY-MM-DD]

   - ``--fast-train`` / ``--no-catboost`` 为界面固定参数，缩短训练耗时。
   - 若环境变量 ``QUANT_TRAIN_END_DATE`` 已设置，会追加 ``--train-end-date``。
   - 子进程 stdout/stderr 经 ``run_command_interactive`` 实时回显；成功则覆盖
     ``models/`` 下权重并写入 ``model_versions`` 表。

2. **命令行 / 定时流水线**（``scripts/auto_pipeline.bat`` 步骤 3）
   - 默认全量训练：``python train_model.py``（含 CatBoost，迭代上限更高）。
   - 调参：``--tune --tune-trials N`` → Optuna 搜索 → ``models/best_params.json``。

整体流水线（``main()``）
-----------------------
::

  本地 K 线 → 逐股因子/标签面板 → 截面特征管道 → Purged 时序划分
       → [可选 Optuna] → LGBM LambdaRank + XGB Ranker + [可选 CatBoost]
       → 验证指标 / PSI / 置换重要性 → Ridge 元模型（OOF Stacking）
       → 落盘 pkl/cbm + register_model_version(set_active=True)

阶段 1：构建训练面板（``_load_local_kline_panel``）
--------------------------------------------------
- 从 ``stock_daily_kline`` 读取 OHLCV 及 industry / market_cap / turnover_rate / pe_ttm。
- ``--train-end-date``：只保留该日及之前的行；筛空则回退为库内全部日期。
- ``--max-stocks`` / ``--sh-main-board`` / ``--train-stock-prefixes``：限制参与训练的股票池。
- 单股最少 K 线根数：``max(MIN_HISTORY_BARS + LABEL_HORIZON_DAYS, 65)``（见 ``src.config``）。
- 对每只有足够历史的股票依次：
  1. ``panel_enrichment.enrich_ohlcv_history`` — 合并基本面/资金流等辅助列；
  2. ``factor_calculator.compute_factors_for_history`` — 技术因子（与预测侧 ``FEATURE_COLUMNS`` 一致）；
  3. ``factor_calculator.compute_morphology_metrics_for_history`` — 形态类因子；
  4. ``factor_calculator.label_forward_return`` — **监督标签** ``label_ret``：
     T 日收盘后信号 → T+1 可成交买入价 → 约 ``LABEL_HORIZON_DAYS``（默认 5）日后可成交卖出价，
     收益率；一字涨跌停顺延，避免不可成交的虚假标签。
- 丢弃 ``FEATURE_COLUMNS + label_ret`` 含 NaN/Inf 的行，合并为全市场面板 DataFrame。

阶段 2：截面特征管道（与任务 C 预测对齐）
-----------------------------------------
- ``factor_calculator.prepare_ranking_cross_section_pipeline(panel)``
  做北向交互、增量库合并、截面去极值/标准化等；**必须与 ``run_daily.py`` 使用同一管道**，
  否则训练特征分布与推理不一致。
- 再次 dropna 后得到可用于排序学习的干净面板。

阶段 3：Purged 时序划分（``model_trainer.split_panel_train_val_purged``）
-------------------------------------------------------------------------
- 按交易日排序；**验证集** = 最后 60 个交易日；**训练集** = 验证起点前再往前
  ``purge_days``（默认 = ``LABEL_HORIZON_DAYS``）个交易日之前的全部样本。
- Purging Gap 防止 forward return 标签跨越切分点造成**时序泄露**。
- 全市场交易日 < 90 天则拒绝训练，避免 train≈val 过拟合。

阶段 4：超参数（可选 ``--tune``）
---------------------------------
- ``--tune``：``model_trainer.optuna_tune_lgbm_ranker`` 在验证集上最大化 **平均 Rank IC**
  （逐日截面 Spearman(预测分, label_ret) 的均值），结果写入 ``models/best_params.json``。
- 未调参时：若存在 ``best_params.json`` 则加载，否则用 ``LGB_RANKER_DEFAULT_PARAMS``。

阶段 5：三级排序基学习器
------------------------
均以**同一训练/验证划分**、**按 date 分组的 query**（每个交易日一组）训练：

1. **LightGBM LambdaRank**（``train_lgbm_ranker``）
   - ``objective=lambdarank``；``label_ret`` 在组内线性映射为 0..30 的整型 relevance。
   - 样本权重：对大幅亏损尾部加权（``ranking_sample_weights_extreme_loss``），强化「避雷」。
   - 早停：验证集 NDCG；输出 ``mean_rank_ic_val`` 等指标。

2. **XGBoost XGBRanker**（``train_xgb_ranker``）
   - ``rank:pairwise``，相同分组与 relevance 定义，与 LGBM 形成多样性。

3. **CatBoost YetiRank**（``train_catboost_ranker_optional``，界面任务 B 默认跳过）
   - 落盘 ``models/cat_ranker.cbm``；``--no-catboost`` 时删除陈旧 cbm，避免预测侧误加载。

``--fast-train`` 降低三者的 ``n_estimators`` 与早停轮数（见 ``main()`` 内 ``n_lgb/n_xgb/n_cat``）。

阶段 6：训练后诊断
------------------
- **PSI**（``population_stability_index``）：训练集 vs 验证集 LGBM 预测分分布漂移；
  超 ``config.json → quant.psi.alert_threshold`` 写入 ``psi_alert``。
- **置换重要性**（``utils.permutation_importance_rank_ic_delta``）：验证集上逐特征打乱对 Rank IC 的影响。

阶段 7：Ridge 元学习器（Stacking）
----------------------------------
- ``fit_meta_stacker_ridge_oof``：在训练集上做 Walk-forward OOF，得到无泄露的一级模型 OOF 分；
  以 OOF 分为特征、``label_ret`` 为标签拟合 ``Ridge(alpha=10)``。
- 验证集用**已训好的全量** LGBM/XGB/[Cat] 打分评估 ``meta_mean_rank_ic_val``。
- ``QUANT_META_OOF_FOLDS=0`` 或 ``ranking.meta_oof_folds<=1`` 时退化为简单 hold-out Stacking。
- 落盘 ``models/meta_rank_stacker.pkl``（含 ``n_base``：2 或 3）。

阶段 8：落盘与版本登记
----------------------
- ``models/lgb_model.pkl``、``models/xgb_model.pkl``、``models/meta_rank_stacker.pkl``、
  [可选] ``models/cat_ranker.cbm``。
- ``database.register_model_version``：写入 ``model_versions``（features、metrics JSON），
  并将新版本 ``is_active=1``（旧版本置 0）。任务 C ``run_daily.py`` 加载的即为 active 版本。

与任务 A / C 的关系
-------------------
- **任务 A**（行情同步）→ 填充 ``stock_daily_kline``；无数据时本脚本 ``SystemExit`` 提示先同步。
- **任务 B**（本脚本）→ 产出 ``models/*`` + ``model_versions``。
- **任务 C**（``run_daily.py``）→ 用同一 ``FEATURE_COLUMNS`` 与 ``prepare_ranking_cross_section_pipeline``
  对最新交易日打分，LGBM+XGB+Cat 一级分经 Ridge 融合后排序，再经经验风控取 TopN 写入 ``daily_selections``。

线程与环境
----------
- 启动时设置 ``KMP_DUPLICATE_LIB_OK``、``OMP_NUM_THREADS``、``MKL_NUM_THREADS``（默认 4），
  减轻 Windows 上 LightGBM/XGB/CatBoost 嵌套 OpenMP 导致的假死。
- 未设置 ``CATBOOST_NUM_THREADS`` 时与 ``OMP_NUM_THREADS`` 对齐。

命令行用法（项目根目录）
------------------------
::

  python train_model.py
  python train_model.py --train-end-date 2024-12-31
  python train_model.py --sh-main-board --fast-train
  python train_model.py --tune --tune-trials 30

主要实现文件
------------
- 本文件 ``train_model.py`` — 入口、面板构建、流程编排
- ``src/model_trainer.py`` — 划分、Ranker 训练、OOF Meta、指标
- ``src/factor_calculator.py`` — 因子与 ``label_forward_return``
- ``src/panel_enrichment.py`` — OHLCV  enrichment
- ``src/config.py`` — ``FEATURE_COLUMNS``、路径、``LABEL_HORIZON_DAYS``
"""
from __future__ import annotations

import gc
import io
import os

# 子进程训练 LightGBM/XGBoost 时降低 OpenMP/MKL 与宿主（如 Streamlit）嵌套冲突概率（尤其 Windows）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
# CatBoost 默认会占满逻辑核；与 LGBM/XGB 串行衔接时易在 Windows 上超订导致极慢或长时间无新日志
if "CATBOOST_NUM_THREADS" not in os.environ:
    _omp_raw = (os.environ.get("OMP_NUM_THREADS") or "4").strip() or "4"
    try:
        _omp_n = max(1, int(_omp_raw))
    except ValueError:
        _omp_n = 4
    os.environ["CATBOOST_NUM_THREADS"] = str(_omp_n)

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
    CATBOOST_MODEL_PATH,
    FEATURE_COLUMNS,
    LABEL_HORIZON_DAYS,
    MIN_HISTORY_BARS,
    MODEL_PATH,
    XGB_MODEL_PATH,
    get_quant_config_merged,
)
from src.database import init_db, register_model_version
from src.factor_calculator import (
    DEFAULT_INDUSTRY_LABEL,
    compute_factors_for_history,
    compute_morphology_metrics_for_history,
    label_forward_return,
    normalize_industry_column,
    prepare_ranking_cross_section_pipeline,
)
from src.panel_enrichment import enrich_ohlcv_history, load_financial_panels_bulk
from src.model_trainer import (
    _json_safe,
    _prepare_rank_xy,
    fit_meta_stacker_ridge_oof,
    load_ranker_params_json,
    mean_cross_sectional_rank_ic,
    merge_ranker_params,
    optuna_tune_lgbm_ranker,
    population_stability_index,
    save_meta_stacker,
    save_model,
    save_ranker_params_json,
    split_panel_train_val_purged,
    train_catboost_ranker_optional,
    train_lgbm_ranker,
    train_xgb_ranker,
)
from src.utils import permutation_importance_rank_ic_delta


# 沪市主板 A 常见代码段（不含 300/301 创业板、688 科创板、8 北交所等）
_SH_MAIN_BOARD_PREFIXES: tuple[str, ...] = ("600", "601", "603", "605")


def _parse_train_stock_prefixes(arg: str) -> list[str]:
    """逗号分隔的前缀列表，用于 6 位 stock_code 左匹配。"""
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--train-stock-prefixes 不能为空。")
    out: list[str] = []
    for p in parts:
        if not p.isdigit() or not (1 <= len(p) <= 6):
            raise SystemExit(
                f"--train-stock-prefixes 每项须为 1～6 位数字，无效项: {p!r}"
            )
        out.append(p)
    return out


def _codes_allowed_by_prefixes(codes: list[str], prefixes: list[str]) -> list[str]:
    return [c for c in codes if any(c.startswith(pref) for pref in prefixes)]


def _line_buffer_stdio_if_supported() -> None:
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is None:
            continue
        reconf = getattr(stream, "reconfigure", None)
        if not callable(reconf):
            continue
        try:
            reconf(line_buffering=True)
        except (OSError, ValueError, io.UnsupportedOperation):
            pass


def _load_local_kline_panel(
    *,
    train_end_date: str | None,
    max_stocks: int | None,
    code_prefixes: list[str] | None,
    verbose: bool,
) -> pd.DataFrame:
    """从 stock_daily_kline 构建训练面板（含 label_ret）。"""
    import sqlite3

    from src.config import DB_PATH

    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    sql = """
        SELECT date, stock_code, stock_name, open, high, low, close, volume,
               industry, market_cap, turnover_rate, pe_ttm
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
    if code_prefixes:
        before_n = len(codes)
        codes = _codes_allowed_by_prefixes(codes, code_prefixes)
        if not codes:
            raise SystemExit(
                "按 --train-stock-prefixes / --sh-main-board 筛选后没有剩余股票代码，"
                "请检查本地库是否含对应板块数据。"
            )
        if verbose:
            print(
                f"[股票范围] 代码前缀 {list(code_prefixes)}："
                f"{before_n} → {len(codes)} 只",
                flush=True,
            )
        raw_df = raw_df[
            raw_df["stock_code"].astype(str).str.zfill(6).isin(set(codes))
        ]
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
    fin_cache = load_financial_panels_bulk(codes)

    for gi, (code, group) in enumerate(grouped):
        if verbose and total_g and (gi % 500 == 0 or gi == total_g - 1):
            print(f"[本地因子] {gi + 1}/{total_g} 只股票…", flush=True)
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) < min_bars:
            skipped_short += 1
            continue
        g = enrich_ohlcv_history(g, stock_code=str(code), financial_cache=fin_cache)
        facts = compute_factors_for_history(g)
        morph = compute_morphology_metrics_for_history(g)
        meta_cols = ["date", "stock_code", "stock_name", "industry"]
        if "market_cap" in g.columns:
            meta_cols.append("market_cap")
        meta = g[meta_cols].copy()
        merged = pd.concat(
            [
                meta.reset_index(drop=True),
                facts.reset_index(drop=True),
                morph.reset_index(drop=True),
            ],
            axis=1,
        )
        merged["label_ret"] = label_forward_return(
            g, horizon=LABEL_HORIZON_DAYS, stock_code=str(code)
        ).values
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
    # Rename market_cap to mcap for factor pipeline compatibility
    if "market_cap" in out.columns:
        out["mcap"] = pd.to_numeric(out["market_cap"], errors="coerce")
        out = out.drop(columns=["market_cap"], errors="ignore")
    if verbose:
        print(f"[本地因子] 合并后面板行数: {len(out)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 SQLite K 线训练 LightGBM LambdaRank + XGBoost Ranker")
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
        help="参与训练的股票数量上限（按代码排序截取）；默认不限制；建议排在 --sh-main-board 之后",
    )
    parser.add_argument(
        "--train-stock-prefixes",
        type=str,
        default=None,
        metavar="PREFIXES",
        help="仅保留代码 6 位左匹配此前缀的股票，逗号分隔，如 600,601,603,605 或仅 600",
    )
    parser.add_argument(
        "--sh-main-board",
        action="store_true",
        help="仅沪市主板常见段（600/601/603/605），不含 300/301 创业板、688 科创板、北交所等",
    )
    parser.add_argument(
        "--fast-train",
        action="store_true",
        help="降低 LGBM/XGB/CatBoost 迭代上限以缩短耗时（效果可能下降）",
    )
    parser.add_argument(
        "--no-catboost",
        action="store_true",
        help="跳过 CatBoost 训练与落盘，仅 LightGBM+XGBoost+Ridge 元模型",
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

    code_prefixes: list[str] | None = None
    if args.train_stock_prefixes:
        code_prefixes = _parse_train_stock_prefixes(args.train_stock_prefixes)
    elif args.sh_main_board:
        code_prefixes = list(_SH_MAIN_BOARD_PREFIXES)

    _line_buffer_stdio_if_supported()
    verbose = not args.quiet
    if verbose and sys.platform == "win32":
        print(
            "[提示] 若在终端里拖选文字，Windows 可能暂停前台训练；卡住时按 Enter 或 Esc 取消选择。",
            flush=True,
        )
    panel = _load_local_kline_panel(
        train_end_date=args.train_end_date,
        max_stocks=args.max_stocks,
        code_prefixes=code_prefixes,
        verbose=verbose,
    )

    if verbose:
        print(
            "[特征管道] 北向交互 + 增量库 + 截面清洗（与预测侧 prepare_ranking_cross_section_pipeline 一致）…",
            flush=True,
        )
    panel = prepare_ranking_cross_section_pipeline(panel, date_col="date")
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["label_ret"]
    )

    unique_dates = sorted(panel["date"].astype(str).str[:10].unique())
    try:
        train_df, val_df, split_meta = split_panel_train_val_purged(panel, date_col="date")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if verbose:
        print(
            "[划分] Purged 时序切分："
            f"训练截止 {split_meta['train_cutoff_date']}，"
            f"隔离带 {split_meta['purge_days']} 日，"
            f"验证自 {split_meta['split_date']} 起共 {split_meta['val_days']} 个交易日。",
            flush=True,
        )

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

    if args.fast_train:
        n_lgb, es_lgb = 360, 40
        n_xgb, es_xgb = 120, 35
        n_cat, es_cat = 90, 30
    else:
        n_lgb, es_lgb = 800, 50
        n_xgb, es_xgb = 200, 50
        n_cat, es_cat = 180, 40

    model, rank_metrics = train_lgbm_ranker(
        train_df,
        val_df,
        params=best_full,
        feature_cols=list(FEATURE_COLUMNS),
        n_estimators=n_lgb,
        early_stopping_rounds=es_lgb,
        verbose=verbose,
    )
    gc.collect()

    if verbose:
        print("[XGBRanker] rank:pairwise，与 LightGBM 相同分组与 relevance …", flush=True)
    xgb_model, xgb_metrics = train_xgb_ranker(
        train_df,
        val_df,
        best_params=best_full,
        feature_cols=list(FEATURE_COLUMNS),
        n_estimators=n_xgb,
        early_stopping_rounds=es_xgb,
        verbose=verbose,
    )
    gc.collect()

    if args.no_catboost:
        cat_model, cat_metrics = None, {"catboost_skip": "disabled_by_cli"}
        if verbose:
            print("[CatBoost] 已跳过（--no-catboost）", flush=True)
    else:
        cat_model, cat_metrics = train_catboost_ranker_optional(
            train_df,
            val_df,
            best_params=best_full,
            feature_cols=list(FEATURE_COLUMNS),
            n_estimators=n_cat,
            early_stopping_rounds=es_cat,
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
    if code_prefixes is not None:
        metrics_register["train_stock_prefixes"] = list(code_prefixes)
    if args.fast_train:
        metrics_register["fast_train"] = True
    if args.no_catboost:
        metrics_register["no_catboost"] = True
    for k, v in rank_metrics.items():
        if k not in metrics_register and str(k).startswith("val_ndcg"):
            metrics_register[k] = v
    metrics_register.update({k: v for k, v in xgb_metrics.items() if v is not None})
    metrics_register.update({k: v for k, v in cat_metrics.items() if v is not None})

    tr_r, X_tr_m, y_tr_m = _prepare_rank_xy(train_df, date_col="date")
    va_r, X_va_m, y_va_m = _prepare_rank_xy(val_df, date_col="date")
    pred_tr_lgb = np.asarray(model.predict(X_tr_m), dtype=float).ravel()
    pred_va_lgb = np.asarray(model.predict(X_va_m), dtype=float).ravel()
    qconf = get_quant_config_merged()
    psi_bins = int(qconf.get("psi", {}).get("bins", 12))
    psi_val = population_stability_index(
        pred_tr_lgb, pred_va_lgb, n_bins=psi_bins
    )
    metrics_register["psi_lgb_pred_train_vs_val"] = psi_val
    psi_thr = float(qconf.get("psi", {}).get("alert_threshold", 0.25))
    if np.isfinite(psi_val) and psi_val > psi_thr:
        metrics_register["psi_alert"] = f"PSI={psi_val:.4f} 超过阈值 {psi_thr}"

    yv_arr = y_va_m.to_numpy(dtype=float)
    dt_va = va_r["date"].to_numpy()

    def _baseline_ic_lgb_tm() -> float:
        p0 = np.asarray(model.predict(X_va_m), dtype=float).ravel()
        return float(mean_cross_sectional_rank_ic(p0, dt_va, yv_arr))

    try:
        pi_lgb = permutation_importance_rank_ic_delta(
            lambda xdf: np.asarray(model.predict(xdf), dtype=float).ravel(),
            X_va_m,
            yv_arr,
            dt_va,
            list(FEATURE_COLUMNS),
            baseline_ic_fn=_baseline_ic_lgb_tm,
            n_repeats=1,
            seed=42,
        )
        metrics_register["perm_importance_lgb_mean_ic_drop"] = _json_safe(pi_lgb)
    except Exception as exc:
        metrics_register["perm_importance_lgb_error"] = str(exc)

    meta_model, meta_m = fit_meta_stacker_ridge_oof(
        model,
        xgb_model,
        cat_model,
        train_df,
        val_df,
        date_col="date",
        cols=list(FEATURE_COLUMNS),
        rank_params=best_full,
    )
    metrics_register.update({k: v for k, v in meta_m.items() if v is not None})
    n_base = int(float(meta_m.get("meta_n_base", 2))) if meta_m else 2
    save_meta_stacker(meta_model, n_base=n_base)

    save_model(model, MODEL_PATH)
    save_model(xgb_model, XGB_MODEL_PATH)
    if cat_model is not None:
        try:
            cat_model.save_model(str(CATBOOST_MODEL_PATH))
            metrics_register["cat_model_path"] = str(CATBOOST_MODEL_PATH)
        except Exception as exc:
            metrics_register["cat_model_save_error"] = str(exc)
    elif args.no_catboost and CATBOOST_MODEL_PATH.exists():
        try:
            CATBOOST_MODEL_PATH.unlink()
            metrics_register["cat_model_removed_stale"] = str(CATBOOST_MODEL_PATH)
        except OSError as exc:
            metrics_register["cat_model_remove_error"] = str(exc)
        if verbose:
            print(
                f"[CatBoost] 已删除旧 {CATBOOST_MODEL_PATH.name}，避免预测侧仍加载过期模型。",
                flush=True,
            )
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
            "xgb_mean_rank_ic_val": xgb_metrics.get("xgb_mean_rank_ic_val"),
            "psi_lgb_pred_train_vs_val": metrics_register.get("psi_lgb_pred_train_vs_val"),
            "meta_mean_rank_ic_val": metrics_register.get("meta_mean_rank_ic_val"),
            "model_path": str(MODEL_PATH),
            "xgb_model_path": str(XGB_MODEL_PATH),
            "best_params_json": str(BEST_LGB_PARAMS_JSON) if args.tune else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    main()
