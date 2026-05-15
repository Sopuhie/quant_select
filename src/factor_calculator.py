"""基于 OHLCV 计算因子（仅使用截至当日的历史，避免未来函数）。

截面清洗支持「行业内 MAD + Z-score」、指定因子分位秩、以及行业内对
``factor_size_mcap``（总市值对数）做最小二乘残差的市值中性化；行业字段由
``stock_daily_kline.industry``（或上游 DataFrame）提供。

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FACTOR_EWM_BLEND,
    FACTOR_EWM_HALFLIFE_DAYS,
    FEATURE_COLUMNS,
    LABEL_HORIZON_DAYS,
)

# 行业内样本过少时退回当日全截面 MAD+Z，避免行业组统计不稳
MIN_INDUSTRY_GROUP_SIZE = 5

# 行业内市值回归样本过少则跳过残差中性化（保留原值）
MIN_SIZE_NEUTRAL_GROUP = 5

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


# 高波动 / 重尾因子：先做截面分位秩 [-0.5, 0.5]，再参与行业内标准化
PERCENTILE_RANK_FEATURES: tuple[str, ...] = (
    "factor_return_1d",
    "factor_return_5d",
    "factor_momentum_10d",
    "factor_volume_ratio",
    "factor_volatility_5d",
    "factor_volatility_20d",
    "factor_amihud_20d",
    "factor_bb_width_20d",
    "factor_drawdown_60d",
    "factor_hsgt_flow_interact",
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
    接口失败或列结构变化时返回空字典（调用方填 0）。
    """
    if not dates:
        return {}
    ds = sorted({str(d).strip()[:10] for d in dates if d})
    if not ds:
        return {}
    start = ds[0].replace("-", "")
    end = ds[-1].replace("-", "")
    try:
        import akshare as ak  # type: ignore[import-untyped]
    except Exception:
        return {}
    try:
        raw = ak.stock_hsgt_hist_em(symbol="北向资金", start_date=start, end_date=end)
    except Exception:
        try:
            raw = ak.stock_hsgt_hist_em(symbol="沪股通", start_date=start, end_date=end)
        except Exception:
            return {}
    if raw is None or getattr(raw, "empty", True):
        return {}
    df = raw.copy()
    date_col = next(
        (c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"),
        df.columns[0],
    )
    val_col = next(
        (
            c
            for c in df.columns
            if "净" in str(c)
            or "流入" in str(c)
            or "当日" in str(c)
            or str(c).lower() in ("value", "net", "amount")
        ),
        None,
    )
    if val_col is None:
        num_cols = [c for c in df.columns if c != date_col and str(df[c].dtype).startswith(("float", "int"))]
        if not num_cols:
            return {}
        val_col = num_cols[-1]
    ser = pd.to_numeric(df[val_col], errors="coerce")
    dt = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    g = pd.DataFrame({"d": dt, "v": ser}).dropna(subset=["d", "v"])
    if g.empty:
        return {}
    mu = float(g["v"].mean())
    sig = float(g["v"].std(ddof=0))
    if not np.isfinite(mu) or not np.isfinite(sig) or sig < 1e-12:
        return {}
    out: dict[str, float] = {}
    for _, row in g.iterrows():
        z = (float(row["v"]) - mu) / sig
        if np.isfinite(z):
            out[str(row["d"])[:10]] = float(z)
    return out


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
    work[out_col] = (zv.fillna(0.0) * r.fillna(0.0)).replace([np.inf, -np.inf], 0.0)
    return work


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
    _ = (stock_code, stock_name)
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
