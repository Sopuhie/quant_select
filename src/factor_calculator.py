"""基于 OHLCV 计算因子（仅使用截至当日的历史，避免未来函数）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_HORIZON_DAYS

# 行业内样本过少时退回当日全截面 MAD+Z，避免行业组统计不稳
MIN_INDUSTRY_GROUP_SIZE = 5

# 高波动因子：先做截面分位秩 [-0.5, 0.5]，再参与行业内标准化
PERCENTILE_RANK_FEATURES: tuple[str, ...] = (
    "factor_return_1d",
    "factor_return_5d",
    "factor_momentum_10d",
    "factor_volume_ratio",
    "factor_volatility_5d",
    "factor_volatility_20d",
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


def compute_factors_for_history(df: pd.DataFrame) -> pd.DataFrame:
    """与 ``train_model`` 本地 SQLite 管线一致的技术因子（13 维）。"""
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

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().bfill()
    out = out.fillna(0.0)
    return out[FEATURE_COLUMNS]


def clean_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    截面清洗：按交易日分组；高波动因子先做分位秩中心化；
    优先按（交易日 × 行业）做 MAD 截断 + Z-score，行业内少于 5 只时对该行业标的退回当日全截面标准化。
    """
    if df.empty:
        return df

    group_key = "date" if "date" in df.columns else "trade_date"
    if group_key not in df.columns:
        return df

    industry_col = "industry" if "industry" in df.columns else None
    ind_series_name = "_cs_industry_key"

    cleaned_parts: list[pd.DataFrame] = []

    for _date, day_df in df.groupby(group_key, sort=False):
        day_df = day_df.copy()

        for col in PERCENTILE_RANK_FEATURES:
            if col not in day_df.columns:
                continue
            s = day_df[col].astype(float)
            day_df[col] = s.rank(pct=True, method="average") - 0.5

        if industry_col is not None:
            ik = (
                day_df[industry_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", "__UNKNOWN__")
            )
        else:
            ik = pd.Series("__UNKNOWN__", index=day_df.index)
        day_df[ind_series_name] = ik

        global_z = pd.DataFrame(index=day_df.index)
        for col in FEATURE_COLUMNS:
            if col not in day_df.columns:
                continue
            global_z[col] = _mad_clip_zscore(day_df[col])

        out_day = day_df.copy()

        for _ind_val, idx in day_df.groupby(ind_series_name, sort=False).groups.items():
            idx = pd.Index(idx)
            if len(idx) >= MIN_INDUSTRY_GROUP_SIZE:
                sub = day_df.loc[idx]
                for col in FEATURE_COLUMNS:
                    if col not in sub.columns:
                        continue
                    out_day.loc[idx, col] = _mad_clip_zscore(sub[col]).values
            else:
                for col in FEATURE_COLUMNS:
                    if col not in global_z.columns:
                        continue
                    out_day.loc[idx, col] = global_z.loc[idx, col].values

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
    merged = pd.concat(
        [
            df[["date"]].reset_index(drop=True),
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
