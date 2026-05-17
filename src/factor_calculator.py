"""基于 OHLCV 计算因子（仅使用截至当日的历史，避免未来函数）。

截面清洗支持「行业内 MAD + Z-score」、指定因子分位秩、以及行业内对
``factor_size_mcap``（总市值对数）做最小二乘残差的市值中性化；行业字段由
``stock_daily_kline.industry``（或上游 DataFrame）提供。

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ENABLE_PREV_GAIN_SUPPRESSION,
    FACTOR_EWM_BLEND,
    FACTOR_EWM_HALFLIFE_DAYS,
    FEATURE_COLUMNS,
    LABEL_HORIZON_DAYS,
    MAX_ALLOWED_20D_RETURN,
    MAX_ALLOWED_5D_RETURN,
)
from .utils import rank_gauss_cross_section, zca_whiten_columns_cross_section

# 行业内样本过少时退回当日全截面 MAD+Z，避免行业组统计不稳
MIN_INDUSTRY_GROUP_SIZE = 5

# 行业内市值回归样本过少则跳过残差中性化（保留原值）
MIN_SIZE_NEUTRAL_GROUP = 5

SIZE_FACTOR_COL = "factor_size_mcap"
HSGT_FLOW_INTERACT_COL = "factor_hsgt_flow_interact"

# 规模因子 log10(市值) 全为缺失时的兜底（约 log10(100 亿元) 量级，仅作回归自变量中轴）
SIZE_MCAP_LOG_FALLBACK = 10.0
# 北向交互因子全缺失时的安全中轴（与 attach 失败置 0 一致）
HSGT_FLOW_INTERACT_SAFE_CENTER = 0.0

# 训练 / 推理共用：缺失或非字符串行业统一为该标签，便于行业内截面标准化成组
DEFAULT_INDUSTRY_LABEL = "未知行业"


def normalize_industry_label(val: object) -> str:
    """将单行行业字段规范为字符串；空值与空白视为默认标签。"""
    if val is None:
        return DEFAULT_INDUSTRY_LABEL
    if isinstance(val, float) and np.isnan(val):
        return DEFAULT_INDUSTRY_LABEL
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return DEFAULT_INDUSTRY_LABEL
    return s


def normalize_industry_column(series: pd.Series) -> pd.Series:
    """整列行业字段批量规范化（SQLite / DataFrame 读入后使用）。"""
    t = series.fillna("").astype(str).str.strip()
    t = t.replace("", DEFAULT_INDUSTRY_LABEL).replace("nan", DEFAULT_INDUSTRY_LABEL)
    return t


# 高波动 / 重尾因子：先做截面分位秩 [-0.5, 0.5]，再参与行业内标准化
PERCENTILE_RANK_FEATURES: tuple[str, ...] = (
    "factor_return_1d",
    "factor_return_5d",
    "factor_momentum_10d",
    "factor_amihud_20d",
    "factor_bb_width_20d",
    "factor_drawdown_60d",
    "factor_hsgt_flow_interact",
    "factor_big_order_net_ratio",
    "factor_north_hold_ratio_chg",
    "factor_chip_profit_ratio",
    "factor_chip_concentration_width",
    "factor_rsi_14",
    "factor_kdj_j",
    "factor_macd_hist",
)

# 波动 / 换手类：先截面相对尺度，再 RankGauss（在 ``clean_cross_sectional_features`` 内顺序执行）
RANK_GAUSS_AFTER_RANK_FEATURES: tuple[str, ...] = (
    "factor_volatility_5d",
    "factor_volatility_20d",
    "factor_volume_ratio",
    "factor_volume_position",
)

# 截面相对波动率：当日个股时序波动 / 当日全市场截面中位数（RankGauss 之前）
CROSS_SECTION_RELATIVE_VOL_FEATURES: tuple[str, ...] = RANK_GAUSS_AFTER_RANK_FEATURES

# 技术指标：截面分位前做 ZCA 对称正交（无列顺序偏好）
TECHNICAL_ORTHOGONAL_FEATURE_COLS: tuple[str, ...] = (
    "factor_rsi_14",
    "factor_kdj_j",
    "factor_macd_hist",
)

# 顺势强度因子：截面清洗时保留原尺度，不参与分位秩 / MAD-Z / 市值中性化
PRESERVE_RAW_CROSS_SECTION_FEATURES: tuple[str, ...] = (
    "factor_ma_trend_score",
    "factor_price_pos_250d",
)


def _mad_clip_zscore(series: pd.Series) -> pd.Series:
    """单截面序列：MAD 截断后 Z-score，索引与原序列对齐。"""
    s = series.astype(float)
    idx = s.index
    if not s.notna().any():
        return pd.Series(np.nan, index=idx)
    if float(s.std(ddof=0)) < 1e-8 or float(s.max() - s.min()) < 1e-15:
        return pd.Series(0.0, index=idx)
    median = float(s.median())
    mad = float((s - median).abs().median())
    threshold = 3.0 * 1.4826 * mad
    if threshold > 1e-6:
        lo = median - threshold
        hi = median + threshold
        s = s.clip(lower=lo, upper=hi)
    mean = float(s.mean())
    std = float(s.std())
    if std > 1e-6:
        return (s - mean) / std
    return pd.Series(0.0, index=idx)


def assign_factor_size_mcap_from_mcap(feat_df: pd.DataFrame) -> pd.DataFrame:
    """由 ``mcap``（总市值，元）推导 ``factor_size_mcap`` = log10(mcap)；非正或缺失为 NaN。"""
    if feat_df.empty:
        return feat_df
    out = feat_df.copy()
    if "mcap" in out.columns:
        mcap_s = pd.to_numeric(out["mcap"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        out[SIZE_FACTOR_COL] = np.where(
            mcap_s.notna() & (mcap_s > 0),
            np.log10(mcap_s.astype(float)),
            np.nan,
        )
    elif SIZE_FACTOR_COL not in out.columns:
        out[SIZE_FACTOR_COL] = np.nan
    return out


def coerce_hsgt_flow_interact_finite(
    feat_df: pd.DataFrame,
    *,
    col: str = HSGT_FLOW_INTERACT_COL,
) -> pd.DataFrame:
    """
    北向交互 ``zv * r5`` 后的数值有限性强转：NaN/Inf 收敛为 0，防止诊股/截面打分爆破。
    """
    if feat_df.empty or col not in feat_df.columns:
        return feat_df
    out = feat_df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    vals = out[col].to_numpy(dtype=float)
    out[col] = np.where(np.isfinite(vals), vals, 0.0)
    return out


# 兼容旧调用名
sanitize_hsgt_flow_interact_column = coerce_hsgt_flow_interact_finite


def sanitize_factor_size_mcap_column(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    全市场/截面 ``factor_size_mcap`` 中位数兜底安全锁（送入行业截面清洗器前调用）。

    等价于::

        feat_df['factor_size_mcap'] = pd.to_numeric(..., errors='coerce').fillna(
            median if any valid else 10.0
        )

    中位数在 ``errors='coerce'`` 之后计算，避免脏字符串拉高统计量。
    """
    if feat_df.empty or SIZE_FACTOR_COL not in feat_df.columns:
        return feat_df
    out = feat_df.copy()
    sz = pd.to_numeric(out[SIZE_FACTOR_COL], errors="coerce")
    out[SIZE_FACTOR_COL] = sz.fillna(
        float(sz.median()) if not sz.isna().all() else float(SIZE_MCAP_LOG_FALLBACK)
    )
    return out


def _apply_size_residual_regression(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    行业内：对每个因子（除 SIZE_FACTOR_COL）做 y ~ 1 + factor_size_mcap 的 OLS，
    用残差替换 y（市值对数中性化）。
    """
    sub_df = sub_df.copy()
    if SIZE_FACTOR_COL not in sub_df.columns:
        return sub_df
    sub_df = sanitize_factor_size_mcap_column(sub_df)
    sz = pd.to_numeric(sub_df[SIZE_FACTOR_COL], errors="coerce")
    if int(sz.notna().sum()) < MIN_SIZE_NEUTRAL_GROUP:
        return sub_df
    if float(sz.std(ddof=0)) < 1e-10:
        return sub_df
    sz_vals = sz.to_numpy(dtype=float)
    for col in FEATURE_COLUMNS:
        if col == SIZE_FACTOR_COL or col in PRESERVE_RAW_CROSS_SECTION_FEATURES:
            continue
        if col not in sub_df.columns:
            continue
        y = pd.to_numeric(sub_df[col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(y) & np.isfinite(sz_vals)
        k = int(m.sum())
        if k < MIN_SIZE_NEUTRAL_GROUP:
            continue
        X = np.column_stack([np.ones(k), sz_vals[m]])
        yy = y[m]
        coef, _, rank, _ = np.linalg.lstsq(X, yy, rcond=None)
        if rank < 2:
            continue
        beta0 = float(coef[0])
        beta1 = float(coef[1])
        pred = beta0 + beta1 * sz_vals
        resid = y - pred
        sub_df[col] = np.where(
            np.isfinite(sz_vals) & np.isfinite(resid),
            resid,
            y,
        )
    return sub_df


def _apply_cross_section_relative_volatility(day_df: pd.DataFrame) -> pd.DataFrame:
    """波动/换手类：当日个股值 / 当日全市场截面中位数，体现相对波动 regime。"""
    out = day_df.copy()
    for col in CROSS_SECTION_RELATIVE_VOL_FEATURES:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        med = float(s.median()) if s.notna().any() else float("nan")
        if np.isfinite(med) and abs(med) > 1e-12:
            out[col] = s / med
    return out


def _limit_move_ratio(stock_code: str | None) -> float:
    """涨跌停幅度代理（主板 10%，创业板/科创板 20%）。"""
    c = str(stock_code or "").strip().zfill(6)
    if c.startswith(("300", "301", "688")):
        return 0.20
    if c.startswith(("8", "4")):
        return 0.30
    return 0.10


def _bar_one_word_limit_locked(
    row: pd.Series,
    prev_close: float,
    *,
    limit_ratio: float,
    direction: str,
) -> bool:
    """一字涨/跌停：开高低收近似相等且触及涨跌停价。"""
    try:
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        vol = float(row.get("volume", 0) or 0)
    except (TypeError, ValueError):
        return False
    if not all(np.isfinite(x) for x in (o, h, l, c, prev_close)) or prev_close <= 0:
        return False
    if vol <= 0:
        return False
    span = max(abs(h - l), 1e-9)
    one_word = span <= max(1e-6, 1e-4 * abs(c))
    if not one_word:
        return False
    tol = prev_close * 1e-3
    if direction == "down":
        limit_px = prev_close * (1.0 - limit_ratio)
        return c <= limit_px + tol
    limit_px = prev_close * (1.0 + limit_ratio)
    return c >= limit_px - tol


def _roll_ewm_blend(s: pd.Series, window: int) -> pd.Series:
    """rolling 与 EWM 线性混合，近期权重更高（因子衰减）。"""
    w = max(2, int(window))
    roll = s.rolling(w, min_periods=min(3, w)).mean()
    hl = float(FACTOR_EWM_HALFLIFE_DAYS)
    ewm = s.ewm(halflife=hl, adjust=False).mean()
    b = float(FACTOR_EWM_BLEND)
    b = max(0.0, min(1.0, b))
    return (1.0 - b) * roll + b * ewm


def build_hsgt_net_zscore_by_trade_date(
    dates: list[str],
) -> dict[str, float]:
    """
    北向资金净流入（全市场日频）按日期映射为 Z-score；用于 ``factor_hsgt_flow_interact``。
    仅从本地 SQLite ``market_hsgt_flow_daily`` 读取（网络同步见 ``market_hsgt_sync`` / 数据脚本）。
    对每个交易日 T，仅用 T 及此前历史做累积均值/标准差，避免混入未来北向数据。
    无本地数据时返回空字典（调用方填 0）。
    """
    if not dates:
        return {}
    ds = sorted({str(d).strip()[:10] for d in dates if d})
    if not ds:
        return {}
    ds_set = set(ds)
    try:
        from .database import fetch_market_hsgt_net_flow_up_to

        flow = fetch_market_hsgt_net_flow_up_to(ds[-1])
    except Exception:
        return {}
    if not flow:
        return {}
    _std_eps = 1e-6
    flow_df = pd.DataFrame(
        sorted((str(k)[:10], float(v)) for k, v in flow.items()),
        columns=["date", "net_inflow"],
    )
    mu = flow_df["net_inflow"].expanding(min_periods=1).mean()
    sig = flow_df["net_inflow"].expanding(min_periods=2).std(ddof=1)
    sig_safe = sig.clip(lower=_std_eps)
    z = (flow_df["net_inflow"] - mu) / sig_safe
    z = z.where(sig.notna(), 0.0)
    z = z.where(np.isfinite(z), np.nan)
    flow_df["z"] = z
    sub = flow_df.loc[flow_df["date"].isin(ds_set), ["date", "z"]].dropna(subset=["z"])
    if sub.empty:
        return {}
    return dict(zip(sub["date"].tolist(), sub["z"].astype(float).tolist()))


def attach_hsgt_flow_interact(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    ret_col: str = "factor_return_5d",
    out_col: str = "factor_hsgt_flow_interact",
) -> pd.DataFrame:
    """按 ``date_col`` 将北向 Z 与 ``ret_col`` 相乘写入 ``out_col``；失败则置 0。"""
    if panel.empty or date_col not in panel.columns:
        return panel
    work = panel.copy()
    if ret_col not in work.columns:
        work[out_col] = 0.0
        return work
    dates = work[date_col].astype(str).str[:10].tolist()
    zmap = build_hsgt_net_zscore_by_trade_date(dates)
    if not zmap:
        work[out_col] = 0.0
        return work
    zv = work[date_col].astype(str).str[:10].map(zmap).astype(float)
    r = pd.to_numeric(work[ret_col], errors="coerce").astype(float)
    work[out_col] = zv * r
    return coerce_hsgt_flow_interact_finite(work, col=out_col)


def suppress_high_recent_gains(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    在 ``clean_cross_sectional_features`` 之前：按原始 ``factor_return_5d``、
    ``factor_momentum_10d`` 剔除短期透支标的（与 ``ENABLE_PREV_GAIN_SUPPRESSION`` 一致）。
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


def merge_incremental_db_features(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    将 SQLite 增量表中的资金流 / 北向持股变化左连接进面板；无键则保留原列（通常为 OHLCV 代理或 0）。
    """
    if panel.empty or date_col not in panel.columns or "stock_code" not in panel.columns:
        return panel
    try:
        from .database import fetch_auxiliary_feature_frames
    except Exception:
        return panel
    work = panel.copy()
    dates_u = sorted({str(d)[:10] for d in work[date_col].astype(str).tolist() if d})
    codes_u = sorted({str(c).strip().zfill(6) for c in work["stock_code"].tolist() if c})
    try:
        from .auxiliary_ak_sync import maybe_autofill_auxiliary_from_env

        maybe_autofill_auxiliary_from_env(dates_u, codes_u, verbose=True)
    except Exception as exc:
        print(f"[merge_incremental_db_features] 自动补全跳过: {exc}", flush=True)
    mf, nh = fetch_auxiliary_feature_frames(dates_u, codes_u)
    work["_k_date"] = work[date_col].astype(str).str[:10]
    work["_k_code"] = work["stock_code"].astype(str).str.zfill(6)
    if mf is not None and not mf.empty and "big_net_ratio" in mf.columns:
        work = work.merge(
            mf,
            on=["_k_date", "_k_code"],
            how="left",
        )
        bn = pd.to_numeric(work["big_net_ratio"], errors="coerce")
        if "factor_big_order_net_ratio" in work.columns:
            base = pd.to_numeric(work["factor_big_order_net_ratio"], errors="coerce")
            work["factor_big_order_net_ratio"] = bn.where(bn.notna(), base).fillna(0.0)
        else:
            work["factor_big_order_net_ratio"] = bn.fillna(0.0)
        work = work.drop(columns=["big_net_ratio"], errors="ignore")
    if nh is not None and not nh.empty and "hold_pct_chg" in nh.columns:
        work = work.merge(
            nh,
            on=["_k_date", "_k_code"],
            how="left",
        )
        hc = pd.to_numeric(work["hold_pct_chg"], errors="coerce")
        if "factor_north_hold_ratio_chg" in work.columns:
            base = pd.to_numeric(work["factor_north_hold_ratio_chg"], errors="coerce")
            work["factor_north_hold_ratio_chg"] = hc.where(hc.notna(), base).fillna(0.0)
        else:
            work["factor_north_hold_ratio_chg"] = hc.fillna(0.0)
        drop_nh = [c for c in ("hold_pct_chg", "hold_pct") if c in work.columns]
        if drop_nh:
            work = work.drop(columns=drop_nh, errors="ignore")
    return work.drop(columns=["_k_date", "_k_code"], errors="ignore")


def enrich_factors_with_incremental_db(
    factors: pd.DataFrame,
    hist: pd.DataFrame,
    *,
    stock_code: str | None = None,
) -> pd.DataFrame:
    """
    单股/小样本因子面板：将 SQLite 增量表中的大单净占比、北向持股变化左连接进因子列，
    避免 ``compute_factors_for_history`` 中 OHLCV 代理或 0 占位与训练截面脱节。
    """
    if factors is None or factors.empty or hist is None or hist.empty:
        return factors
    code = str(stock_code or "").strip().zfill(6)
    if len(code) != 6 and "stock_code" in hist.columns:
        code = str(hist.iloc[-1]["stock_code"]).strip().zfill(6)
    if len(code) != 6:
        return factors
    panel = factors.copy()
    panel["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    panel["stock_code"] = code
    panel = merge_incremental_db_features(panel, date_col="date")
    cols = [c for c in FEATURE_COLUMNS if c in panel.columns]
    if len(cols) != len(FEATURE_COLUMNS):
        for c in FEATURE_COLUMNS:
            if c not in panel.columns:
                panel[c] = factors[c] if c in factors.columns else 0.0
    return panel[FEATURE_COLUMNS]


def prepare_ranking_cross_section_pipeline(
    feat_df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    训练 / 预测 / 回测共用的截面排序特征管道：
    北向交互 → 增量库特征合并 → 截面分组用 ``date`` → 前期涨幅抑制 →
    对数市值 ``factor_size_mcap`` → **截面中位数兜底** → 行业截面清洗。

    当 ``date_col="trade_date"`` 时，会临时构造 ``date`` 供 ``clean_cross_sectional_features`` 分组，
    结束后删除 ``date``，保留 ``trade_date``。当 ``date_col="date"``（训练面板）时**保留** ``date``。
    """
    if feat_df.empty:
        return feat_df
    w = feat_df.copy()
    w = attach_hsgt_flow_interact(w, date_col=date_col)
    w = merge_incremental_db_features(w, date_col=date_col)
    if "date" not in w.columns:
        w["date"] = w[date_col].astype(str).str[:10]
    else:
        w["date"] = w["date"].astype(str).str[:10]
    w = suppress_high_recent_gains(w)
    w = assign_factor_size_mcap_from_mcap(w)
    # 全市场截面中位数兜底安全锁（行业/市值截面归一化之前；严格按量化柜台规范）
    if "factor_size_mcap" in w.columns:
        w["factor_size_mcap"] = pd.to_numeric(
            w["factor_size_mcap"], errors="coerce"
        ).fillna(
            w["factor_size_mcap"].median()
            if not w["factor_size_mcap"].isna().all()
            else 10.0
        )
    w = clean_cross_sectional_features(w)
    if str(date_col) != "date":
        w = w.drop(columns=["date"], errors="ignore")
    return w


def compute_factors_for_history(df: pd.DataFrame) -> pd.DataFrame:
    """与 ``train_model`` / ``run_daily`` 一致：按 ``FEATURE_COLUMNS`` 输出因子列。"""
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    work = df.sort_values("date").reset_index(drop=True)

    close = work["close"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    vol = work["volume"].astype(float)

    eps = 1e-12
    ma5 = _roll_ewm_blend(close, 5)
    ma10 = _roll_ewm_blend(close, 10)
    ma20 = _roll_ewm_blend(close, 20)
    ma60 = _roll_ewm_blend(close, 60)

    out = pd.DataFrame(index=work.index)
    out["factor_bias_5"] = (close - ma5) / (ma5 + eps)
    out["factor_bias_10"] = (close - ma10) / (ma10 + eps)
    out["factor_bias_20"] = (close - ma20) / (ma20 + eps)
    out["factor_bias_60"] = (close - ma60) / (ma60 + eps)
    out["factor_ratio_5_20"] = ma5 / (ma20 + eps) - 1.0
    out["factor_ratio_10_60"] = ma10 / (ma60 + eps) - 1.0

    out["factor_return_1d"] = close.pct_change(1)
    out["factor_return_5d"] = close.pct_change(5)
    out["factor_momentum_10d"] = close / (close.shift(10) + eps) - 1.0

    raw_trend_score = (
        (ma5 > ma10).astype(float)
        + (ma10 > ma20).astype(float)
        + (ma20 > ma60).astype(float)
        + (close > ma20).astype(float)
    )
    out["factor_ma_trend_score"] = raw_trend_score

    roll_min250 = close.rolling(250, min_periods=30).min()
    roll_max250 = close.rolling(250, min_periods=30).max()
    raw_price_pos_250d = (close - roll_min250) / (roll_max250 - roll_min250 + eps)
    out["factor_price_pos_250d"] = raw_price_pos_250d

    vol_ma5 = _roll_ewm_blend(vol, 5)
    vol_ma20 = _roll_ewm_blend(vol, 20)
    out["factor_volume_ratio"] = vol / (vol_ma5 + eps)
    out["factor_volume_position"] = vol_ma5 / (vol_ma20 + eps) - 1.0

    high_low_ratio = (high - low) / (close + eps)
    out["factor_volatility_5d"] = high_low_ratio.rolling(5).mean()
    out["factor_volatility_20d"] = high_low_ratio.rolling(20).mean()

    denom = high.astype(float) - low.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    d_np = denom.to_numpy(dtype=float)
    safe_d = np.where(d_np < eps, eps, d_np)
    out["factor_close_position"] = pd.Series(
        (c.to_numpy(dtype=float) - l.to_numpy(dtype=float)) / safe_d,
        index=work.index,
        dtype=float,
    ).clip(0.0, 1.0)

    # --- 扩展因子（OHLCV；北向交互在面板层 ``attach_hsgt_flow_interact`` 填充）---
    ret1 = close.pct_change(1)
    dollar_vol = (close * vol).replace(0, np.nan)
    out["factor_amihud_20d"] = (ret1.abs() / (dollar_vol + eps)).rolling(20, min_periods=5).mean()

    dv = vol.pct_change(1).replace([np.inf, -np.inf], np.nan)
    out["factor_pv_corr_10d"] = ret1.rolling(10, min_periods=5).corr(dv)

    typical = (high + low + close) / 3.0
    vwap_num = (typical * vol).rolling(20, min_periods=5).sum()
    vwap_den = vol.rolling(20, min_periods=5).sum()
    vwap20 = vwap_num / (vwap_den + eps)
    out["factor_vwap_bias_20d"] = close / (vwap20 + eps) - 1.0

    std20 = close.rolling(20, min_periods=5).std()
    out["factor_bb_width_20d"] = (4.0 * std20) / (ma20 + eps)

    roll_max60 = close.rolling(60, min_periods=20).max()
    out["factor_drawdown_60d"] = close / (roll_max60 + eps) - 1.0

    d1 = close.pct_change(1)
    shrink_day = (d1 < 0) & (vol < vol_ma5)
    out["factor_shrink_pullback_5d"] = shrink_day.astype(float).rolling(5, min_periods=1).mean()

    # --- RSI / KDJ / MACD（时间序列；截面正交在 clean 内完成）---
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = ag / (al + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    out["factor_rsi_14"] = (rsi - 50.0) / 50.0

    low9 = low.rolling(9, min_periods=5).min()
    high9 = high.rolling(9, min_periods=5).max()
    rsv = (close - low9) / (high9 - low9 + eps) * 100.0
    rsv = rsv.clip(0.0, 100.0)
    K = rsv.rolling(3, min_periods=1).mean()
    D = K.rolling(3, min_periods=1).mean()
    J = 3.0 * K - 2.0 * D
    out["factor_kdj_j"] = (J - 50.0) / 50.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    out["factor_macd_hist"] = (dif - dea) / (close + eps)

    vwap60_num = (typical * vol).rolling(60, min_periods=20).sum()
    vwap60_den = vol.rolling(60, min_periods=20).sum()
    vwap60 = vwap60_num / (vwap60_den + eps)
    out["factor_chip_profit_ratio"] = (close - vwap60) / (close + eps)

    w90 = close.rolling(60, min_periods=20).quantile(0.95) - close.rolling(
        60, min_periods=20
    ).quantile(0.05)
    w70 = close.rolling(60, min_periods=20).quantile(0.85) - close.rolling(
        60, min_periods=20
    ).quantile(0.15)
    out["factor_chip_concentration_width"] = (w90 - w70) / (close + eps)

    r1 = close.pct_change(1).replace([np.inf, -np.inf], np.nan)
    signed_amt = (close * vol) * np.sign(r1.fillna(0.0))
    out["factor_big_order_net_ratio"] = signed_amt.rolling(5, min_periods=2).sum() / (
        dollar_vol.rolling(5, min_periods=2).sum() + eps
    )
    out["factor_north_hold_ratio_chg"] = 0.0

    out["factor_hsgt_flow_interact"] = 0.0

    out = out.replace([np.inf, -np.inf], np.nan)
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].ffill().bfill().fillna(0.0)

    return out[FEATURE_COLUMNS]


def clean_cross_sectional_features(
    df: pd.DataFrame,
    *,
    use_industry_neutralization: bool = True,
    use_percentile_rank_volatility: bool = True,
    use_size_neutralization: bool = True,
) -> pd.DataFrame:
    """
    截面清洗：

    - 按 ``date`` / ``trade_date`` 分组；
    - 对 ``TECHNICAL_ORTHOGONAL_FEATURE_COLS`` 先做 ZCA 对称正交；
    - 对波动/换手类先做截面相对波动率（/ 当日中位数），再 RankGauss；
    - 对 ``PERCENTILE_RANK_FEATURES`` 做 ``rank(pct=True) - 0.5``；
    - 对 ``RANK_GAUSS_AFTER_RANK_FEATURES`` 做截面 RankGauss；
    - 可选：在每个行业组内对因子（除 ``factor_size_mcap``）相对市值对数做 OLS 残差提取；
    - 可选：行业内 MAD+Z；组内过少则退回当日全截面 MAD+Z。
    """
    if df.empty:
        return df

    group_key = "date" if "date" in df.columns else "trade_date"
    if group_key not in df.columns:
        return df

    cleaned_parts: list[pd.DataFrame] = []

    for _date, day_df in df.groupby(group_key, sort=False):
        day_df = day_df.copy()
        day_df = assign_factor_size_mcap_from_mcap(day_df)
        day_df = sanitize_factor_size_mcap_column(day_df)

        og_ok = all(c in day_df.columns for c in TECHNICAL_ORTHOGONAL_FEATURE_COLS)
        if og_ok:
            M = (
                day_df.loc[:, list(TECHNICAL_ORTHOGONAL_FEATURE_COLS)]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            Q = zca_whiten_columns_cross_section(M)
            for j, c in enumerate(TECHNICAL_ORTHOGONAL_FEATURE_COLS):
                day_df[c] = Q[:, j]

        if use_percentile_rank_volatility:
            day_df = _apply_cross_section_relative_volatility(day_df)
            for col in PERCENTILE_RANK_FEATURES:
                if col not in day_df.columns or col in PRESERVE_RAW_CROSS_SECTION_FEATURES:
                    continue
                s = day_df[col].astype(float)
                day_df[col] = s.rank(pct=True, method="average") - 0.5
            for col in RANK_GAUSS_AFTER_RANK_FEATURES:
                if col not in day_df.columns or col in PRESERVE_RAW_CROSS_SECTION_FEATURES:
                    continue
                s = day_df[col].astype(float)
                day_df[col] = rank_gauss_cross_section(s)

        ind_series_name = "_cs_industry_key"
        if use_industry_neutralization and "industry" in day_df.columns:
            day_df[ind_series_name] = normalize_industry_column(day_df["industry"])
        else:
            day_df[ind_series_name] = DEFAULT_INDUSTRY_LABEL

        out_day = day_df.copy()

        for _ind_val, idx in day_df.groupby(ind_series_name, sort=False).groups.items():
            idx = pd.Index(idx)
            if use_size_neutralization:
                sub_sz = _apply_size_residual_regression(out_day.loc[idx].copy())
                for col in FEATURE_COLUMNS:
                    if col in sub_sz.columns:
                        out_day.loc[idx, col] = sub_sz[col].to_numpy()

        global_z = pd.DataFrame(index=out_day.index)
        for col in FEATURE_COLUMNS:
            if col not in out_day.columns or col in PRESERVE_RAW_CROSS_SECTION_FEATURES:
                continue
            global_z[col] = _mad_clip_zscore(out_day[col])

        for _ind_val, idx in day_df.groupby(ind_series_name, sort=False).groups.items():
            idx = pd.Index(idx)
            if use_industry_neutralization:
                if len(idx) >= MIN_INDUSTRY_GROUP_SIZE:
                    sub2 = out_day.loc[idx]
                    for col in FEATURE_COLUMNS:
                        if col not in sub2.columns or col in PRESERVE_RAW_CROSS_SECTION_FEATURES:
                            continue
                        out_day.loc[idx, col] = _mad_clip_zscore(sub2[col]).values
                else:
                    for col in FEATURE_COLUMNS:
                        if col not in global_z.columns or col in PRESERVE_RAW_CROSS_SECTION_FEATURES:
                            continue
                        out_day.loc[idx, col] = global_z.loc[idx, col].values
            else:
                sub2 = out_day.loc[idx]
                for col in FEATURE_COLUMNS:
                    if col not in sub2.columns or col in PRESERVE_RAW_CROSS_SECTION_FEATURES:
                        continue
                    out_day.loc[idx, col] = _mad_clip_zscore(sub2[col]).values

        out_day = out_day.drop(columns=[ind_series_name])
        cleaned_parts.append(out_day)

    return pd.concat(cleaned_parts, ignore_index=True)


def anchor_ma_levels_from_history(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    锚定日（最后一根 K 线）的收盘价与 20/60 日混合均线。
    与 ``compute_factors_for_history`` 使用相同的 ``_roll_ewm_blend`` 定义。
    """
    if df is None or df.empty:
        return float("nan"), float("nan"), float("nan")
    work = df.sort_values("date").reset_index(drop=True)
    close = work["close"].astype(float)
    if close.empty:
        return float("nan"), float("nan"), float("nan")
    ma20 = _roll_ewm_blend(close, 20)
    ma60 = _roll_ewm_blend(close, 60)
    i = len(close) - 1
    c = float(close.iloc[i])
    m20 = float(ma20.iloc[i]) if np.isfinite(ma20.iloc[i]) else float("nan")
    m60 = float(ma60.iloc[i]) if np.isfinite(ma60.iloc[i]) else float("nan")
    return c, m20, m60


def label_forward_return(
    ohlcv: pd.DataFrame | pd.Series,
    horizon: int = LABEL_HORIZON_DAYS,
    *,
    stock_code: str | None = None,
) -> pd.Series:
    """
    T 日因子标签：T 收盘 → 可成交买入价 → 可成交卖出价，收益率考虑涨跌停一字板流动性陷阱。

    ``ohlcv`` 为含 open/high/low/close/volume 的 DataFrame；若仅传入 ``close`` Series 则退回简单 shift 标签。
    """
    if isinstance(ohlcv, pd.Series):
        c = ohlcv.astype(float)
        future = c.shift(-horizon)
        valid = (c > 1e-8) & c.notna() & future.notna() & (future > 0)
        ret = (future / c - 1.0).where(valid)
        return ret.replace([np.inf, -np.inf], np.nan)

    work = ohlcv.sort_values("date").reset_index(drop=True) if "date" in ohlcv.columns else ohlcv.reset_index(drop=True)
    n = len(work)
    out = np.full(n, np.nan, dtype=float)
    if n < horizon + 2:
        return pd.Series(out, index=work.index, dtype=float)

    limit_r = _limit_move_ratio(stock_code)
    close = pd.to_numeric(work["close"], errors="coerce").astype(float)
    max_scan = min(n, horizon + 25)

    for i in range(n - 1):
        c0 = float(close.iloc[i])
        if not np.isfinite(c0) or c0 <= 1e-8:
            continue
        buy_i = i + 1
        if buy_i >= n:
            break
        prev_buy = c0
        # 一字涨停无法买入，顺延至可成交开放日；一字跌停在 sell_i 循环处理
        while buy_i < min(n, i + max_scan):
            row_b = work.iloc[buy_i]
            if _bar_one_word_limit_locked(
                row_b, prev_buy, limit_ratio=limit_r, direction="up"
            ):
                prev_buy = float(row_b["close"])
                buy_i += 1
                continue
            break
        if buy_i >= n:
            continue
        entry = float(work.iloc[buy_i]["close"])
        if not np.isfinite(entry) or entry <= 1e-8:
            continue

        sell_i = min(i + horizon, n - 1)
        if sell_i <= buy_i:
            continue
        prev_sell = float(close.iloc[sell_i - 1]) if sell_i > 0 else entry
        while sell_i < min(n - 1, i + max_scan):
            row_s = work.iloc[sell_i]
            if _bar_one_word_limit_locked(
                row_s, prev_sell, limit_ratio=limit_r, direction="down"
            ):
                prev_sell = float(row_s["close"])
                sell_i += 1
                continue
            break
        if sell_i >= n:
            continue
        exit_px = float(work.iloc[sell_i]["close"])
        if not np.isfinite(exit_px) or exit_px <= 0:
            continue
        r = exit_px / entry - 1.0
        if np.isfinite(r):
            out[i] = r

    return pd.Series(out, index=work.index, dtype=float).replace([np.inf, -np.inf], np.nan)


def build_stock_panel_features(
    df: pd.DataFrame,
    stock_code: str,
    stock_name: str,
) -> pd.DataFrame:
    _ = stock_name
    factors = compute_factors_for_history(df)
    lab = label_forward_return(df, horizon=LABEL_HORIZON_DAYS, stock_code=stock_code)
    meta_cols = ["date"]
    if "industry" in df.columns:
        meta_cols.append("industry")
    merged = pd.concat(
        [
            df[meta_cols].reset_index(drop=True),
            factors.reset_index(drop=True),
        ],
        axis=1,
    )
    merged["stock_code"] = stock_code
    merged["stock_name"] = stock_name
    merged["label_ret"] = lab.values
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=FEATURE_COLUMNS + ["label_ret"])
    _f32_cols = [c for c in FEATURE_COLUMNS + ["label_ret"] if c in merged.columns]
    if _f32_cols:
        merged[_f32_cols] = merged[_f32_cols].astype(np.float32)
    return merged
