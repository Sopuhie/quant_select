"""基于 OHLCV 计算因子（仅使用截至当日的历史，避免未来函数）。

截面清洗支持「行业内 MAD + Z-score」、指定因子分位秩、以及行业内对
``factor_size_mcap``（总市值对数）做最小二乘残差的市值中性化；行业字段由
``stock_daily_kline.industry``（或上游 DataFrame）提供。

"""
from __future__ import annotations

from collections.abc import Iterable

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
from .utils import rank_gauss_cross_section

# 行业内样本过少时退回当日全截面 MAD+Z，避免行业组统计不稳
MIN_INDUSTRY_GROUP_SIZE = 5

# 行业内市值回归样本过少则跳过残差中性化（保留原值）
MIN_SIZE_NEUTRAL_GROUP = 5

SIZE_FACTOR_COL = "factor_size_mcap"

# 规模因子 log10(市值) 全为缺失时的兜底（约 log10(100 亿元) 量级，仅作回归自变量中轴）
SIZE_MCAP_LOG_FALLBACK = 10.0

# 训练 / 推理共用：缺失或非字符串行业统一为该标签，便于行业内截面标准化成组
DEFAULT_INDUSTRY_LABEL = "未知行业"

# 数据库与 K 线均无有效行业时的兜底（与截面清洗同源，避免 NaN 丢行）
CANONICAL_INDUSTRY_FALLBACK_LABEL = "综合"


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


def fetch_canonical_industry_map(
    stock_codes: Iterable[str],
) -> dict[str, str]:
    """
    训练 / 预测共用：从 ``stock_daily_kline`` 取各股最新非空行业（与 ``fetch_latest_industry_by_codes`` 同源）。
    """
    from .database import fetch_latest_industry_by_codes

    uniq = sorted(
        {
            str(c).strip().zfill(6)
            for c in stock_codes
            if len(str(c).strip().zfill(6)) == 6
        }
    )
    if not uniq:
        return {}
    raw = fetch_latest_industry_by_codes(uniq)
    out: dict[str, str] = {}
    for code in uniq:
        label = normalize_industry_label(raw.get(code))
        if label == DEFAULT_INDUSTRY_LABEL:
            out[code] = CANONICAL_INDUSTRY_FALLBACK_LABEL
        else:
            out[code] = label
    return out


def attach_canonical_industry_labels(
    feat_df: pd.DataFrame,
    *,
    stock_code_col: str = "stock_code",
    industry_col: str = "industry",
) -> pd.DataFrame:
    """
    在截面清洗前统一行业列：优先 SQLite 最新行业快照，其次 K 线行内 industry，最后「综合」。
  训练与 ``run_daily`` / 回测均经 ``prepare_ranking_cross_section_pipeline`` 调用，保证同源。
    """
    if feat_df.empty or stock_code_col not in feat_df.columns:
        return feat_df
    w = feat_df.copy()
    codes = w[stock_code_col].astype(str).str.zfill(6)
    imap = fetch_canonical_industry_map(codes.unique().tolist())
    resolved: list[str] = []
    for _, row in w.iterrows():
        code = str(row[stock_code_col]).strip().zfill(6)
        db_label = imap.get(code, CANONICAL_INDUSTRY_FALLBACK_LABEL)
        if db_label not in (DEFAULT_INDUSTRY_LABEL, CANONICAL_INDUSTRY_FALLBACK_LABEL):
            resolved.append(db_label)
            continue
        if industry_col in w.columns:
            kline_label = normalize_industry_label(row.get(industry_col))
            if kline_label != DEFAULT_INDUSTRY_LABEL:
                resolved.append(kline_label)
                continue
        resolved.append(CANONICAL_INDUSTRY_FALLBACK_LABEL)
    w[industry_col] = normalize_industry_column(pd.Series(resolved, index=w.index))
    return w


# 高波动 / 重尾因子：先做截面分位秩 [-0.5, 0.5]，再参与行业内标准化
PERCENTILE_RANK_FEATURES: tuple[str, ...] = (
    "factor_return_1d",
    "factor_return_5d",
    "factor_momentum_10d",
)

# 量能类：先截面相对尺度，再 RankGauss（在 ``clean_cross_sectional_features`` 内顺序执行）
RANK_GAUSS_AFTER_RANK_FEATURES: tuple[str, ...] = (
    "factor_volume_ratio",
    "factor_volume_position",
)

# 截面相对波动率：当日个股值 / 当日全市场截面中位数（RankGauss 之前）
CROSS_SECTION_RELATIVE_VOL_FEATURES: tuple[str, ...] = RANK_GAUSS_AFTER_RANK_FEATURES

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


def is_bar_suspended(row: pd.Series) -> bool:
    """成交量为 0 或无效 → 停牌/无量，不可成交。"""
    try:
        v = float(row.get("volume", 0) or 0)
    except (TypeError, ValueError):
        return True
    return not (np.isfinite(v) and v > 0)


def is_bar_one_word_limit_up(
    row: pd.Series,
    prev_close: float,
    stock_code: str,
) -> bool:
    """一字涨停：开高低收贴涨停价且无量价波动，开盘价无法买入。"""
    if is_bar_suspended(row):
        return False
    return _bar_one_word_limit_locked(
        row,
        prev_close,
        limit_ratio=_limit_move_ratio(stock_code),
        direction="up",
    )


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


def prepare_ranking_cross_section_pipeline(
    feat_df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    训练 / 预测 / 回测共用的截面排序特征管道（轻量化纯净版）：
    仅处理 13 个纯技术面量价因子，彻底移除大单、北向、资金流等外部增量库合并与冗余查库动作。
    """
    if feat_df.empty:
        return feat_df
    w = feat_df.copy()

    if "date" not in w.columns:
        w["date"] = w[date_col].astype(str).str[:10]
    else:
        w["date"] = w["date"].astype(str).str[:10]

    w = attach_canonical_industry_labels(w)
    w = suppress_high_recent_gains(w)
    w = assign_factor_size_mcap_from_mcap(w)

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
    截面清洗（13 维纯技术因子，无 MACD/RSI/KDJ ZCA 正交）：

    - 按 ``date`` / ``trade_date`` 分组；
    - 对量能类先做截面相对波动率（/ 当日中位数），再 RankGauss；
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
