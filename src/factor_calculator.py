"""基于 OHLCV 计算因子（仅使用截至当日的历史，避免未来函数）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_HORIZON_DAYS


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
    return out[FEATURE_COLUMNS]


def clean_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """按交易日截面 MAD 去极值 + Z-score（训练用 date；推理可用 trade_date）。"""
    if df.empty:
        return df

    group_key = "date" if "date" in df.columns else "trade_date"
    if group_key not in df.columns:
        return df

    cleaned_parts: list[pd.DataFrame] = []

    for _date, group in df.groupby(group_key):
        group_cleaned = group.copy()

        for col in FEATURE_COLUMNS:
            if col not in group_cleaned.columns:
                continue

            series = group_cleaned[col].astype(float)
            if series.isna().all() or series.std() < 1e-8:
                continue

            median = series.median()
            mad = (series - median).abs().median()
            threshold = 3 * 1.4826 * mad

            if threshold > 1e-6:
                lower_limit = median - threshold
                upper_limit = median + threshold
                series = series.clip(lower=lower_limit, upper=upper_limit)

            mean = series.mean()
            std = series.std()
            if std > 1e-6:
                group_cleaned[col] = (series - mean) / std
            else:
                group_cleaned[col] = 0.0

        cleaned_parts.append(group_cleaned)

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
