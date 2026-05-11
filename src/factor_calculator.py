"""基于 OHLCV 计算因子（仅使用截至当日的历史，避免未来函数）。

截面清洗支持「行业内 MAD + Z-score」、指定因子分位秩、以及行业内对
``factor_size_mcap``（总市值对数）做最小二乘残差的市值中性化；行业字段由
``stock_daily_kline.industry``（或上游 DataFrame）提供。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_HORIZON_DAYS

# 行业内样本过少时退回当日全截面 MAD+Z，避免行业组统计不稳
MIN_INDUSTRY_GROUP_SIZE = 5

# 行业内市值回归样本过少则跳过残差中性化（保留原值）
MIN_SIZE_NEUTRAL_GROUP = 5

# RSI / W%R / ATR 窗口（与因子列名一致）
OSCILLATOR_PERIOD = 14

SIZE_FACTOR_COL = "factor_size_mcap"

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


# 高波动因子：先做截面分位秩 [-0.5, 0.5]，再参与行业内标准化
PERCENTILE_RANK_FEATURES: tuple[str, ...] = (
    "factor_return_1d",
    "factor_return_5d",
    "factor_momentum_10d",
    "factor_volume_ratio",
    "factor_volatility_5d",
    "factor_volatility_20d",
)


def _rsi(close: pd.Series, period: int = OSCILLATOR_PERIOD) -> pd.Series:
    """Wilder RSI，区间约 [0,100]。"""
    eps = 1e-12
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    ag = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ag / (al + eps)
    return 100.0 - (100.0 / (1.0 + rs))


def _williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = OSCILLATOR_PERIOD,
) -> pd.Series:
    """Williams %R，典型区间 [-100, 0]。"""
    eps = 1e-12
    hh = high.astype(float).rolling(period).max()
    ll = low.astype(float).rolling(period).min()
    return -100.0 * (hh - close.astype(float)) / (hh - ll + eps)


def _atr_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = OSCILLATOR_PERIOD,
) -> pd.Series:
    """Wilder ATR / 收盘价，无量纲波动尺度。"""
    eps = 1e-12
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(
        axis=1
    )
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr / (c + eps)


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


def _apply_size_residual_regression(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    行业内：对每个因子（除 SIZE_FACTOR_COL）做 y ~ 1 + factor_size_mcap 的 OLS，
    用残差替换 y（市值对数中性化）。
    """
    sub_df = sub_df.copy()
    if SIZE_FACTOR_COL not in sub_df.columns:
        return sub_df
    sz = pd.to_numeric(sub_df[SIZE_FACTOR_COL], errors="coerce")
    if int(sz.notna().sum()) < MIN_SIZE_NEUTRAL_GROUP:
        return sub_df
    if float(sz.std(ddof=0)) < 1e-10:
        return sub_df
    sz_vals = sz.to_numpy(dtype=float)
    for col in FEATURE_COLUMNS:
        if col == SIZE_FACTOR_COL:
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


def compute_factors_for_history(df: pd.DataFrame) -> pd.DataFrame:
    """与 ``train_model`` 本地 SQLite 管线一致的技术因子（FEATURE_COLUMNS 维）。"""
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    eps = 1e-12
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    out = pd.DataFrame(index=df.index)
    out["factor_bias_5"] = (close - ma5) / (ma5 + eps)
    out["factor_bias_10"] = (close - ma10) / (ma10 + eps)
    out["factor_bias_20"] = (close - ma20) / (ma20 + eps)
    out["factor_bias_60"] = (close - ma60) / (ma60 + eps)

    out["factor_ratio_5_20"] = ma5 / (ma20 + eps) - 1.0
    out["factor_ratio_10_60"] = ma10 / (ma60 + eps) - 1.0

    out["factor_return_1d"] = close.pct_change(1)
    out["factor_return_5d"] = close.pct_change(5)
    out["factor_momentum_10d"] = close / (close.shift(10) + eps) - 1.0

    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    out["factor_volume_ratio"] = vol / (vol_ma5 + eps)
    out["factor_volume_position"] = vol_ma5 / (vol_ma20 + eps) - 1.0

    high_low_ratio = (high - low) / (close + eps)
    out["factor_volatility_5d"] = high_low_ratio.rolling(5).mean()
    out["factor_volatility_20d"] = high_low_ratio.rolling(20).mean()

    out["factor_rsi_14"] = _rsi(close, OSCILLATOR_PERIOD)
    out["factor_wr_14"] = _williams_r(high, low, close, OSCILLATOR_PERIOD)
    out["factor_atr_14"] = _atr_ratio(high, low, close, OSCILLATOR_PERIOD)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["factor_macd_diff"] = (ema12 - ema26) / (close + eps)

    denom = high.astype(float) - low.astype(float)
    safe_den = denom.where(denom >= eps, eps)
    out["factor_close_position"] = (
        (close.astype(float) - low.astype(float)) / safe_den
    ).clip(0.0, 1.0)

    if "market_cap" in df.columns:
        mcap = pd.to_numeric(df["market_cap"], errors="coerce").ffill().bfill()
    else:
        mcap = pd.Series(np.nan, index=df.index)
    mcap = mcap.clip(lower=0.0).fillna(0.0)
    out[SIZE_FACTOR_COL] = np.log(np.maximum(mcap.to_numpy(dtype=float), 1.0))

    out = out.replace([np.inf, -np.inf], np.nan)

    tech_cols = [c for c in FEATURE_COLUMNS if c != SIZE_FACTOR_COL]
    out[tech_cols] = out[tech_cols].ffill().bfill().fillna(0.0)
    out[SIZE_FACTOR_COL] = (
        pd.to_numeric(out[SIZE_FACTOR_COL], errors="coerce").ffill().bfill().fillna(0.0)
    )

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
    - 可选：对 ``PERCENTILE_RANK_FEATURES`` 做 ``rank(pct=True) - 0.5``；
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

        if use_percentile_rank_volatility:
            for col in PERCENTILE_RANK_FEATURES:
                if col not in day_df.columns:
                    continue
                s = day_df[col].astype(float)
                day_df[col] = s.rank(pct=True, method="average") - 0.5

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
            if col not in out_day.columns:
                continue
            global_z[col] = _mad_clip_zscore(out_day[col])

        for _ind_val, idx in day_df.groupby(ind_series_name, sort=False).groups.items():
            idx = pd.Index(idx)
            if use_industry_neutralization:
                if len(idx) >= MIN_INDUSTRY_GROUP_SIZE:
                    sub2 = out_day.loc[idx]
                    for col in FEATURE_COLUMNS:
                        if col not in sub2.columns:
                            continue
                        out_day.loc[idx, col] = _mad_clip_zscore(sub2[col]).values
                else:
                    for col in FEATURE_COLUMNS:
                        if col not in global_z.columns:
                            continue
                        out_day.loc[idx, col] = global_z.loc[idx, col].values
            else:
                sub2 = out_day.loc[idx]
                for col in FEATURE_COLUMNS:
                    if col not in sub2.columns:
                        continue
                    out_day.loc[idx, col] = _mad_clip_zscore(sub2[col]).values

        out_day = out_day.drop(columns=[ind_series_name])
        cleaned_parts.append(out_day)

    return pd.concat(cleaned_parts, ignore_index=True)


def label_forward_return(close: pd.Series, horizon: int = LABEL_HORIZON_DAYS) -> pd.Series:
    """T 日因子对应的标签：从 T 收盘到 T+h 收盘的收益率（最后 horizon 行为 NaN）。"""
    c = close.astype(float)
    future = c.shift(-horizon)
    valid = (c > 1e-8) & c.notna() & future.notna() & (future > 0)
    ret = future / c - 1.0
    ret = ret.where(valid)
    return ret.replace([np.inf, -np.inf], np.nan)


def build_stock_panel_features(
    df: pd.DataFrame,
    stock_code: str,
    stock_name: str,
) -> pd.DataFrame:
    factors = compute_factors_for_history(df)
    lab = label_forward_return(df["close"].astype(float))
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
    return merged
