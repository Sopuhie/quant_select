"""加载模型、全市场打分、写库；更新历史选股收益。"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    AKSHARE_REQUEST_TIMEOUT,
    EXCLUDE_NEAR_LIMIT_LAST_BAR,
    EXCLUDE_ST,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NEAR_LIMIT_PCT_THRESHOLD,
    PREDICT_FETCH_WORKERS,
    TOP_N_SELECTION,
)
from .data_fetcher import (
    compact_start_calendar_days_ago,
    fetch_daily_hist,
    get_risk_st_stock_codes,
    get_stock_pool,
    has_enough_history,
)
from .database import (
    fetch_latest_industry_by_codes,
    fetch_latest_market_cap_by_codes,
    get_active_model_version,
    init_db,
    insert_daily_predictions,
    delete_daily_outputs_for_trade_date,
    insert_daily_selections,
    selection_exists_for_date,
)
from .factor_calculator import (
    clean_cross_sectional_features,
    compute_factors_for_history,
    normalize_industry_label,
)
from .model_trainer import load_model, load_xgb_ranker_optional
from .utils import is_kline_too_stale_vs_prediction


def feature_importances_aligned(model: Any) -> list[float]:
    """与 ``FEATURE_COLUMNS`` 对齐的特征重要性；缺失或非 sklearn 模型时退回均等权重。"""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return [1.0] * len(FEATURE_COLUMNS)
    arr = np.asarray(imp, dtype=float).ravel()
    if arr.size != len(FEATURE_COLUMNS):
        return [1.0] * len(FEATURE_COLUMNS)
    return arr.tolist()


def analyze_stock_reasons(
    row: Any,
    importances: list[float],
    feature_cols: list[str],
) -> str:
    """按因子贡献 Top3 生成可读「入选原因」文案。"""
    templates = {
        "factor_bias_5": "短期均线偏离度处于合理洗盘或强势突破的上攻区间",
        "factor_bias_60": "稳稳站上60日牛熊分界线，具备中长期大底部扎实筹码特征",
        "factor_ratio_5_20": "5日均线与20日生命线多头间距完美拉开，呈现加速拉升之势",
        "factor_return_1d": "昨日放量收大阳线（或涨停），日内多头动量惯性极强",
        "factor_momentum_10d": "10日动量效应共振爆发，资金多头追涨意愿高涨",
        "factor_volume_ratio": "今日成交量较5日均量显著放量，主力资金正深度建仓突破",
        "factor_volatility_5d": "短期历史波动率处于爆发临界点，向上变盘向上弹性极大",
        "factor_macd_diff": "MACD中线金叉发散，趋势红柱持续高企增长",
        "factor_close_position": "收盘价几乎砸在全天最高点，主力抢筹极其坚决，多头承接完美",
        "factor_pe_ratio": "估值市盈率分位数极具性价比，具备安全边际防御特征",
        "factor_turnover_rate": "换手率处于高度活跃区间，筹码交换频繁，主力博弈资金关注度极高",
    }

    contributions: list[tuple[str, float]] = []
    for i, col in enumerate(feature_cols):
        imp = importances[i] if i < len(importances) else 1.0
        try:
            raw = row.get(col, 0) if hasattr(row, "get") else row[col]
            val = float(raw)
        except (TypeError, ValueError, KeyError):
            val = 0.0
        score = val * float(imp)
        contributions.append((col, score))

    contributions.sort(key=lambda x: x[1], reverse=True)
    top_feats = [c[0] for c in contributions[:3]]

    reasons: list[str] = []
    for idx, f in enumerate(top_feats):
        desc = templates.get(f, f"核心因子 [{f}] 处于截面优势地位")
        reasons.append(f"{idx + 1}. {desc}")

    return "；".join(reasons) + "。"


def blend_ranker_scores(
    lgb_scores: np.ndarray | pd.Series,
    xgb_scores: np.ndarray | pd.Series | None,
    *,
    lgb_weight: float = 0.6,
    xgb_weight: float = 0.4,
) -> np.ndarray:
    """
    双排序器融合：将各自原始得分在截面转为分位秩 ``[0,1]`` 后加权；
    若无 XGBoost 得分则退回 LightGBM 原始 predict（与旧管线兼容）。
    """
    ls = np.asarray(lgb_scores, dtype=float).ravel()
    if xgb_scores is None:
        return ls.astype(float)
    xs = np.asarray(xgb_scores, dtype=float).ravel()
    if xs.shape[0] != ls.shape[0]:
        raise ValueError("LightGBM 与 XGBoost 预测长度不一致")
    lgb_r = pd.Series(ls).rank(pct=True).to_numpy(dtype=float)
    xgb_r = pd.Series(xs).rank(pct=True).to_numpy(dtype=float)
    w = float(lgb_weight) + float(xgb_weight)
    if w <= 0:
        return lgb_r.astype(float)
    return (
        (float(lgb_weight) / w) * lgb_r + (float(xgb_weight) / w) * xgb_r
    ).astype(float)


def _normalize_stock_display_name(stock_name: Optional[str]) -> str:
    if not stock_name:
        return ""
    s = str(stock_name).strip()
    s = s.replace("\u3000", "").replace("\xa0", "")
    # 全角星号等统一为 ASCII *（部分数据源用 ＊ 而非 *）
    for ch in ("＊", "\u2217", "\u22c6"):
        s = s.replace(ch, "*")
    return s


def is_st_stock_name(stock_name: Optional[str]) -> bool:
    """名称是否属于 ST / *ST 等（含全角＊、大小写混排）。"""
    n = _normalize_stock_display_name(stock_name)
    if not n:
        return False
    nu = n.upper()
    if any(k in nu for k in ("*ST", "S*ST", "SST")):
        return True
    if nu.startswith("ST"):
        return True
    if re.search(r"\*\s*S\s*T", nu):
        return True
    return False


def is_st_stock_row(stock_code: Any, stock_name: Optional[str]) -> bool:
    """名称规则 + 东方财富风险警示板代码。"""
    if is_st_stock_name(stock_name):
        return True
    code = str(stock_code).strip().zfill(6)
    if len(code) == 6 and code.isdigit() and code in get_risk_st_stock_codes():
        return True
    return False


def filter_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤名称中含 ST 风险标识的股票。
    使用字面子串匹配（regex=False），避免 '*ST' 在正则里有特殊含义。
    """
    if df.empty or "stock_name" not in df.columns:
        return df.copy() if len(df) else df
    st_keywords = ["*ST", "S*ST", "SST", "ST"]
    s = df["stock_name"].map(_normalize_stock_display_name).astype(str)
    hit = pd.Series(False, index=df.index)
    for kw in st_keywords:
        hit |= s.str.contains(kw, na=False, regex=False)
    return df[~hit].reset_index(drop=True)


def filter_predictions(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤打分结果：剔除 ST；可选剔除上一交易日接近涨跌停。
    scores_df 需含 stock_code、stock_name；可选列 pct_prev_day。
    """
    out = scores_df.copy()
    if EXCLUDE_ST:
        if "stock_code" in out.columns:
            mask = ~out.apply(
                lambda r: is_st_stock_row(r.get("stock_code"), r.get("stock_name")),
                axis=1,
            )
            out = out[mask]
        elif "stock_name" in out.columns:
            out = out[~out["stock_name"].map(is_st_stock_name)]
    if (
        EXCLUDE_NEAR_LIMIT_LAST_BAR
        and "pct_prev_day" in out.columns
        and len(out) > 0
    ):
        # 剔除「上一交易日」涨幅已达涨停附近标的（避免追高难以成交）；不限跌停侧
        p = pd.to_numeric(out["pct_prev_day"], errors="coerce")
        out = out[p < NEAR_LIMIT_PCT_THRESHOLD].copy()
    return out.reset_index(drop=True)


def _industry_from_hist_last_bar(hist: pd.DataFrame) -> str:
    if hist.empty or "industry" not in hist.columns:
        return ""
    raw = hist.iloc[-1].get("industry")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    return s if s else ""


def latest_feature_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty or len(df) < 30:
        return None
    fac = compute_factors_for_history(df)
    if fac.empty:
        return None
    last_idx = len(df) - 1
    row = fac.iloc[last_idx]
    if row[FEATURE_COLUMNS].isna().any():
        return None
    out = row[FEATURE_COLUMNS].copy()
    out["date"] = df.iloc[last_idx]["date"]
    out["close"] = float(df.iloc[last_idx]["close"])
    return out


def _fetch_one_stock_features(
    code: str,
    name: str,
    start_compact: str,
    timeout: float,
    industry_by_code: dict[str, str] | None,
    mcap_by_code: dict[str, float] | None,
) -> Optional[dict[str, Any]]:
    """仅拉数据算因子；在主线程里统一 predict，避免多线程共用模型。"""
    hist = fetch_daily_hist(code, start_date=start_compact, timeout=timeout)
    if not has_enough_history(hist):
        return None
    code6 = str(code).strip().zfill(6)
    mc = (mcap_by_code or {}).get(code6)
    if mc is not None and np.isfinite(mc) and mc > 0:
        hist = hist.copy()
        hist["market_cap"] = float(mc)
    feat = latest_feature_row(hist)
    if feat is None:
        return None
    row: dict[str, Any] = {
        "trade_date": str(feat["date"]),
        "stock_code": code,
        "stock_name": name,
        "close_price": float(feat["close"]),
    }
    for c in FEATURE_COLUMNS:
        row[c] = float(feat[c])
    hist_ind = _industry_from_hist_last_bar(hist)
    if hist_ind:
        row["industry"] = normalize_industry_label(hist_ind)
    else:
        row["industry"] = normalize_industry_label(
            (industry_by_code or {}).get(code6)
        )
    if len(hist) >= 2:
        c0 = float(hist.iloc[-2]["close"])
        c1 = float(hist.iloc[-1]["close"])
        row["pct_prev_day"] = (c1 / c0 - 1.0) if c0 > 1e-12 else float("nan")
    else:
        row["pct_prev_day"] = float("nan")
    return row


def predict_universe_scores(
    stock_pairs: list[tuple[str, str]],
    model_path: Path | None = None,
    max_workers: int | None = None,
    verbose: bool = True,
    *,
    target_trade_date: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """
    对股票池用最新一根 K 线计算因子并预测分数。
    并发拉取 K 线（缩短历史窗口），主线程一次性 predict，避免 sklearn 特征名告警与长时间无反馈。
    返回 (as_of_trade_date, DataFrame columns: stock_code, stock_name, score, close_price)。
    ``score``：若存在 ``models/xgb_model.pkl``，为 LightGBM / XGBoost 截面分位秩的 0.6/0.4 融合；否则为 LightGBM 原始得分。

    Args:
        target_trade_date: 期望预测所属交易日（YYYY-MM-DD）。传入时以此为锚剔除
            「最新 K 线日晚于该日」或相对该日滞后超过 5 个交易日的标的；不传则以样本中最新交易日为锚。
    """
    workers = int(max_workers if max_workers is not None else PREDICT_FETCH_WORKERS)
    workers = max(1, min(workers, 32))
    start_compact = compact_start_calendar_days_ago()
    timeout = AKSHARE_REQUEST_TIMEOUT
    lgb_model = load_model(model_path)
    xgb_model = load_xgb_ranker_optional()

    init_db()
    pool_codes = [str(c).strip().zfill(6) for c, _ in stock_pairs]
    industry_by_code = fetch_latest_industry_by_codes(pool_codes)
    mcap_by_code = fetch_latest_market_cap_by_codes(pool_codes)

    raw_rows: list[dict[str, Any]] = []
    total = len(stock_pairs)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _fetch_one_stock_features,
                c,
                n,
                start_compact,
                timeout,
                industry_by_code,
                mcap_by_code,
            ): (c, n)
            for c, n in stock_pairs
        }
        for fut in as_completed(futs):
            done += 1
            if verbose and done % 40 == 0:
                print(f"拉取进度: {done}/{total}", flush=True)
            try:
                r = fut.result()
            except Exception:
                continue
            if r:
                raw_rows.append(r)
    if not raw_rows:
        raise RuntimeError(
            f"未能拉取到任何有效 K 线（本次股票池共 {total} 只，全部未得到可用因子行）。"
            "已在 AkShare 失败后尝试 Baostock 兜底（默认开启，需 pip install baostock）。\n"
            "仍失败时请检查：① 网络 / 代理 ② "
            "set QUANT_AK_TIMEOUT=45 QUANT_FETCH_RETRIES=5 ③ "
            "set QUANT_FETCH_WORKERS=2 ④ "
            "set QUANT_MAX_STOCKS=100 ⑤ "
            "关闭兜底排查：set QUANT_BAOSTOCK_FALLBACK=0"
        )
    feat_df = pd.DataFrame(raw_rows)
    tb = feat_df["trade_date"].map(lambda x: str(x).strip()[:10])

    if target_trade_date is not None:
        anchor = str(target_trade_date).strip()[:10]
        feat_df = feat_df.loc[tb <= anchor].copy()
        tb = feat_df["trade_date"].map(lambda x: str(x).strip()[:10])
    else:
        anchor = str(tb.max())

    stale_mask = tb.map(
        lambda lb: is_kline_too_stale_vs_prediction(str(lb), anchor)
    )
    feat_df = feat_df.loc[~stale_mask].copy()
    tb = feat_df["trade_date"].map(lambda x: str(x).strip()[:10])
    if feat_df.empty:
        raise RuntimeError(
            "剔除「最新 K 线相对锚定交易日滞后超过 5 个交易日」的标的后样本为空。"
        )

    as_of = str(tb.max())
    feat_df = feat_df.loc[tb == as_of].reset_index(drop=True)
    if feat_df.empty:
        raise RuntimeError("过滤到统一交易日后样本为空。")

    # 与 train_model / run_daily 一致：按「date」分组做截面清洗（业内 MAD+Z + 分位秩）
    feat_df["date"] = feat_df["trade_date"]
    feat_df = clean_cross_sectional_features(feat_df)
    feat_df = feat_df.drop(columns=["date"], errors="ignore")

    feat_df = filter_predictions(feat_df)
    if EXCLUDE_ST:
        feat_df = filter_st_stocks(feat_df)
    if feat_df.empty:
        raise RuntimeError("经 ST/涨跌停过滤后无可用股票，请检查股票池或关闭部分过滤开关。")

    X = feat_df[FEATURE_COLUMNS].astype(np.float64)
    lgb_scores = lgb_model.predict(X)
    xgb_scores = xgb_model.predict(X) if xgb_model is not None else None
    feat_df["score"] = blend_ranker_scores(lgb_scores, xgb_scores)
    feat_df = feat_df.sort_values(
        ["score", "stock_code"],
        ascending=[False, True],
    ).reset_index(drop=True)
    cols = ["trade_date", "stock_code", "stock_name", "score", "close_price"]
    out = feat_df[cols + FEATURE_COLUMNS].copy()
    return as_of, out


def get_full_market_predictions(
    stock_pairs: list[tuple[str, str]],
    model_path: Path | None = None,
    max_workers: int | None = None,
    verbose: bool = True,
    *,
    target_trade_date: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """全市场（给定股票池）打分别名，逻辑同 predict_universe_scores。"""
    return predict_universe_scores(
        stock_pairs,
        model_path=model_path,
        max_workers=max_workers,
        verbose=verbose,
        target_trade_date=target_trade_date,
    )


def persist_predictions(
    trade_date: str,
    scores_df: pd.DataFrame,
    top_n: int = TOP_N_SELECTION,
) -> None:
    delete_daily_outputs_for_trade_date(trade_date)
    scores_df = filter_predictions(scores_df.copy())
    if EXCLUDE_ST:
        scores_df = filter_st_stocks(scores_df)
    if scores_df.empty:
        raise RuntimeError("写入前过滤结果为空，无法保存预测与选股。")
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
    scores_df["rank_in_market"] = np.arange(1, len(scores_df) + 1)
    lgb_model = load_model()
    imp_vec = feature_importances_aligned(lgb_model)
    pred_rows = scores_df.to_dict("records")
    insert_daily_predictions(
        [
            {
                "trade_date": trade_date,
                "stock_code": r["stock_code"],
                "stock_name": r["stock_name"],
                "score": float(r["score"]),
                "rank_in_market": int(r["rank_in_market"]),
            }
            for r in pred_rows
        ]
    )
    top = scores_df.head(top_n).reset_index(drop=True)
    sel_rows = []
    for i, r in top.iterrows():
        reason = analyze_stock_reasons(r, imp_vec, FEATURE_COLUMNS)
        sel_rows.append(
            {
                "trade_date": trade_date,
                "stock_code": r["stock_code"],
                "stock_name": r["stock_name"],
                "rank": int(i) + 1,
                "score": float(r["score"]),
                "close_price": float(r["close_price"]),
                "next_day_return": None,
                "hold_5d_return": None,
                "selection_reason": reason,
            }
        )
    insert_daily_selections(sel_rows)


def run_selection_for_latest(
    max_stocks: int,
    skip_if_exists: bool = True,
    model_path: Path | None = None,
    fetch_workers: int | None = None,
    verbose: bool = True,
    pool_as_of_date: str | None = None,
) -> dict[str, Any]:
    from datetime import datetime

    as_of = pool_as_of_date or datetime.now().strftime("%Y-%m-%d")
    pairs = get_stock_pool(as_of_date=as_of, max_count=max_stocks)
    if not pairs:
        raise RuntimeError("股票列表为空（请检查指数成份接口或网络）。")
    trade_date, df = predict_universe_scores(
        pairs,
        model_path=model_path,
        max_workers=fetch_workers,
        verbose=verbose,
        target_trade_date=as_of,
    )
    if len(df) < TOP_N_SELECTION:
        raise RuntimeError(
            f"过滤后仅 {len(df)} 只股票，不足 Top{TOP_N_SELECTION}，可调大股票池或放宽过滤。"
        )
    if skip_if_exists and selection_exists_for_date(trade_date):
        return {"skipped": True, "trade_date": trade_date, "reason": "already_exists"}
    persist_predictions(trade_date, df)
    mv = get_active_model_version()
    return {
        "skipped": False,
        "trade_date": trade_date,
        "n_scored": len(df),
        "model_version": mv["version"] if mv else None,
    }


def backfill_selection_returns(
    max_stocks: int = 800,
) -> int:
    """兼容旧入口：转调统一回填逻辑。"""
    from .return_updater import update_all_returns

    _ = max_stocks
    out = update_all_returns()
    return int(out.get("rows_updated", 0))
