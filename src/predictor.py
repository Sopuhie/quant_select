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
    ENABLE_PREV_GAIN_SUPPRESSION,
    EXCLUDE_NEAR_LIMIT_LAST_BAR,
    EXCLUDE_ST,
    FEATURE_COLUMNS,
    MAX_ALLOWED_20D_RETURN,
    MAX_ALLOWED_5D_RETURN,
    MODEL_PATH,
    NEAR_LIMIT_PCT_THRESHOLD,
    PREDICT_FETCH_WORKERS,
    TOP_N_SELECTION,
    get_experience_thresholds,
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


def suppress_high_recent_gains(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    在 ``clean_cross_sectional_features`` 之前执行：按**原始** ``factor_return_5d``、
    ``factor_momentum_10d`` 剔除短期涨幅透支标的（清洗后该列已非收益率，不可再比阈值）。
    """
    if feat_df is None or len(feat_df) == 0:
        return feat_df
    try:
        if not ENABLE_PREV_GAIN_SUPPRESSION:
            return feat_df
        out = feat_df.copy()
        n0 = len(out)
        if "factor_return_5d" in out.columns:
            r5 = pd.to_numeric(out["factor_return_5d"], errors="coerce")
            out = out[r5 <= float(MAX_ALLOWED_5D_RETURN)]
        if "factor_momentum_10d" in out.columns and len(out) > 0:
            m10 = pd.to_numeric(out["factor_momentum_10d"], errors="coerce")
            out = out[m10 <= float(MAX_ALLOWED_20D_RETURN)]
        if len(out) < n0:
            print(
                "[风控增强] 已强行剔除过去5日涨幅过大或中线动量过高的高位超买股 "
                f"（5日上限 {MAX_ALLOWED_5D_RETURN * 100:.1f}%，动量上限 {MAX_ALLOWED_20D_RETURN * 100:.1f}%），"
                f"共剔除 {n0 - len(out)} 只。",
                flush=True,
            )
        return out.reset_index(drop=True)
    except Exception as exc:
        print(f"[警告] 前期涨幅压制阀门执行异常: {exc}", flush=True)
        return feat_df


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
        "factor_bias_5": "短期5日均线乖离温和，低位安全蓄势区，价格处于健康反弹或突破通道",
        "factor_bias_10": "10日均线支撑扎实，低位换手充分，多头震荡上行形态保持良好",
        "factor_bias_20": "运行在20日生命线之上，低位安全垫较厚，中线上行防守及承接结构优异",
        "factor_bias_60": "站稳60日牛熊分界线，具备中长期大底部扎实换手筑底特征",
        "factor_ratio_5_20": "短期与中期均线距离拉开，呈现健康的经典多头形态形态",
        "factor_ratio_10_60": "中长期均线系统发散向上，呈现典型的黄金多头排列形态",
        "factor_return_1d": "昨日放量收出大阳线，日内多头动量和赚钱效应较强",
        "factor_return_5d": "近5个交易日表现温和、蓄势充分，并未透支暴涨，属于典型的安全低位起步拐点",
        "factor_momentum_10d": "10日截面动量效应爆发，多头追涨及资金吸筹意愿高涨",
        "factor_volume_ratio": "今日成交量较5日均量显著放量，增量资金正深度建仓突破",
        "factor_volume_position": "5日均量超越20日均量，量能温和交织放大，买盘换手充分",
        "factor_volatility_5d": "短期历史波动率处于变盘临界点，个股向上拉升弹性极大",
        "factor_volatility_20d": "中期波幅收敛后重新发散，有望打开全新一轮上升主升浪",
        "factor_close_position": "收盘价几乎死死封在全天最高点，日内主力资金控盘和买入抢筹极其坚决",
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


def apply_experience_trading_filters(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    交易员经验硬过滤：在取 Top N 前按价格、总市值（亿元）、换手（或量比代理）裁剪。
    各阈值为 ``None`` 时不启用该项；``mcap`` 为 NaN 时不因市值上下限丢行。
    阈值优先来自 ``config.json`` → ``experience_filters``，见 ``get_experience_thresholds``。
    """
    if scores_df is None or scores_df.empty:
        return scores_df
    (
        min_price,
        max_price,
        min_mcap,
        max_mcap,
        min_turnover,
        max_turnover,
    ) = get_experience_thresholds()
    if all(
        v is None
        for v in (
            min_price,
            max_price,
            min_mcap,
            max_mcap,
            min_turnover,
            max_turnover,
        )
    ):
        return scores_df
    filtered_df = scores_df.copy()
    n0 = len(filtered_df)
    try:
        if "close_price" in filtered_df.columns:
            if min_price is not None:
                cp = pd.to_numeric(filtered_df["close_price"], errors="coerce")
                filtered_df = filtered_df[cp >= float(min_price)]
            if max_price is not None and len(filtered_df) > 0:
                cp = pd.to_numeric(filtered_df["close_price"], errors="coerce")
                filtered_df = filtered_df[cp <= float(max_price)]

        if "mcap" in filtered_df.columns and (
            min_mcap is not None or max_mcap is not None
        ):
            m_bn = pd.to_numeric(filtered_df["mcap"], errors="coerce") / 1e8
            if min_mcap is not None:
                filtered_df = filtered_df[m_bn.isna() | (m_bn >= float(min_mcap))]
                m_bn = pd.to_numeric(filtered_df["mcap"], errors="coerce") / 1e8
            if max_mcap is not None and len(filtered_df) > 0:
                filtered_df = filtered_df[m_bn.isna() | (m_bn <= float(max_mcap))]

        if (min_turnover is not None or max_turnover is not None) and len(
            filtered_df
        ) > 0:
            if "turnover_rate" in filtered_df.columns:
                tr = pd.to_numeric(filtered_df["turnover_rate"], errors="coerce")
                if min_turnover is not None:
                    filtered_df = filtered_df[tr >= float(min_turnover)]
                    tr = pd.to_numeric(filtered_df["turnover_rate"], errors="coerce")
                if max_turnover is not None and len(filtered_df) > 0:
                    filtered_df = filtered_df[tr <= float(max_turnover)]
            elif "volume_ratio_raw" in filtered_df.columns:
                vr = pd.to_numeric(filtered_df["volume_ratio_raw"], errors="coerce")
                if min_turnover is not None:
                    filtered_df = filtered_df[vr >= (float(min_turnover) / 2.0)]
                    vr = pd.to_numeric(
                        filtered_df["volume_ratio_raw"], errors="coerce"
                    )
                if max_turnover is not None and len(filtered_df) > 0:
                    filtered_df = filtered_df[vr <= (float(max_turnover) / 2.0)]
            elif "factor_volume_ratio" in filtered_df.columns:
                vr = pd.to_numeric(filtered_df["factor_volume_ratio"], errors="coerce")
                if min_turnover is not None:
                    filtered_df = filtered_df[vr >= (float(min_turnover) / 2.0)]
                    vr = pd.to_numeric(
                        filtered_df["factor_volume_ratio"], errors="coerce"
                    )
                if max_turnover is not None and len(filtered_df) > 0:
                    filtered_df = filtered_df[vr <= (float(max_turnover) / 2.0)]

        n1 = len(filtered_df)
        print(
            "[经验风控] 已按阈值裁剪："
            f"价格[{min_price}-{max_price}] 元，"
            f"市值[{min_mcap}-{max_mcap}] 亿元，"
            f"换手/量比代理[{min_turnover}-{max_turnover}]% → {n0} → {n1} 只",
            flush=True,
        )
        return filtered_df.reset_index(drop=True)
    except Exception as exc:
        print(f"[警告] 执行交易员经验条件过滤失败: {exc}", flush=True)
        return scores_df


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


def latest_feature_row(
    df: pd.DataFrame, *, stock_code: str | None = None
) -> Optional[pd.Series]:
    if df.empty or len(df) < 30:
        return None
    _ = stock_code
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
    feat = latest_feature_row(hist, stock_code=code6)
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

    if mc is not None and np.isfinite(mc) and mc > 0:
        row["mcap"] = float(mc)
    else:
        row["mcap"] = float("nan")
    vol_s = hist["volume"].astype(float)
    vma5 = vol_s.rolling(5).mean()
    li = len(hist) - 1
    row["volume_ratio_raw"] = float(vol_s.iloc[li] / (float(vma5.iloc[li]) + 1e-12))
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
    feat_df = suppress_high_recent_gains(feat_df)
    if feat_df.empty:
        raise RuntimeError(
            "前期涨幅压制后无剩余候选股票，可设置 QUANT_PREV_GAIN_SUPPRESSION=0 关闭，"
            "或放宽 QUANT_MAX_5D_RETURN / QUANT_MAX_20D_MOMENTUM。"
        )
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
    meta_keep = [c for c in ("mcap", "volume_ratio_raw") if c in feat_df.columns]
    cols = ["trade_date", "stock_code", "stock_name", "score", "close_price"] + meta_keep
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
    scores_df = apply_experience_trading_filters(scores_df)
    if scores_df.empty:
        raise RuntimeError(
            "经验风控过滤后无剩余股票，请在 config.json 的 experience_filters 放宽阈值，"
            "或在 Streamlit「任务 C」展开面板中调整。"
        )
    scores_df = scores_df.reset_index(drop=True)
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
