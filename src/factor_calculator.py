"""基于 OHLCV 计算因子（仅使用截至当日的历史，避免未来函数）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_HORIZON_DAYS


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif - dea


def compute_factors_for_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入单只股票按日期升序的 OHLCV，为每一行计算当日因子（该行因子仅依赖当日及以前数据）。
    返回与 df 等长的 DataFrame，含 FEATURE_COLUMNS；首若干行为 NaN（窗口不足）。
    """
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["volatility_20d"] = out["ret_1d"].rolling(20).std()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    out["vol_ratio_5_20"] = vol_ma5 / (vol_ma20 + 1e-12)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    out["ma5_bias"] = close / (ma5 + 1e-12) - 1
    out["ma20_bias"] = close / (ma20 + 1e-12) - 1
    out["rsi_14"] = _rsi(close, 14)
    out["macd_hist"] = _macd_hist(close)
    return out[FEATURE_COLUMNS]


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
    """
    单只股票：合并 date、代码、名称、因子、标签。
    """
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
