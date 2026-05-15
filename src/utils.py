"""日期与交易日工具；截面 RankGauss、施密特正交与排列重要性等纯数学工具。"""
from __future__ import annotations

import os
import time as time_module
from datetime import date, datetime, time, timedelta
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

_nd = NormalDist()


def rank_gauss_cross_section(s: pd.Series, *, eps: float = 1e-6) -> pd.Series:
    """
    截面 RankGauss：将分位秩映射为近似标准正态，缓解重尾。
    使用平均秩分位 ``(rank - 0.5) / n`` 再 ``NormalDist.inv_cdf``。
    """
    if s is None or len(s) == 0:
        return s
    v = pd.to_numeric(s, errors="coerce").astype(float)
    n = int(v.notna().sum())
    if n == 0:
        return pd.Series(np.nan, index=v.index)
    r = v.rank(method="average", pct=False)
    u = (r - 0.5) / max(float(n), 1.0)
    u = u.clip(float(eps), 1.0 - float(eps))
    out = np.full(len(v), np.nan, dtype=float)
    m = v.notna().to_numpy()
    inv = np.fromiter((_nd.inv_cdf(float(x)) for x in u[m]), dtype=float, count=int(m.sum()))
    out[np.where(m)[0]] = inv
    return pd.Series(out, index=v.index, dtype=float)


def gram_schmidt_columns_cross_section(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    列向施密特正交化（单位列）：``X`` 形状 ``(n_samples, n_features)``，逐列相对前面列去相关并单位化。
    用于同一交易日内多只股票在技术指标因子张成的子空间上去共线。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("gram_schmidt_columns_cross_section expects 2d array")
    n, k = X.shape
    Q = np.zeros((n, k), dtype=float)
    for j in range(k):
        v = X[:, j].copy()
        for i in range(j):
            qi = Q[:, i]
            ip = float(np.dot(qi, v))
            v = v - ip * qi
        norm = float(np.linalg.norm(v))
        if norm > float(eps):
            Q[:, j] = v / norm
        else:
            Q[:, j] = 0.0
    return Q


def permutation_importance_rank_ic_delta(
    predict_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    dates: np.ndarray,
    feature_cols: list[str],
    *,
    baseline_ic_fn,
    n_repeats: int = 1,
    seed: int = 42,
) -> dict[str, float]:
    """
    对排序模型做简易排列重要性：逐列打乱后 Rank IC 相对基线的下降量（越大越重要）。
    ``predict_fn(X_df) -> 1d array``；``baseline_ic_fn`` 无参，返回当前未扰动基线 IC。
    """
    rng = np.random.default_rng(int(seed))
    base = float(baseline_ic_fn())
    out: dict[str, float] = {}
    for c in feature_cols:
        if c not in X.columns:
            continue
        drops: list[float] = []
        for _ in range(max(1, int(n_repeats))):
            Xp = X.copy()
            col = Xp[c].to_numpy(copy=True)
            rng.shuffle(col)
            Xp[c] = col
            pred = np.asarray(predict_fn(Xp), dtype=float).ravel()
            frame = pd.DataFrame(
                {"d": pd.Series(dates).astype(str).str[:10], "p": pred, "y": y}
            )
            ics: list[float] = []
            for _, sub in frame.groupby("d", sort=False):
                if len(sub) < 10:
                    continue
                ic = sub["p"].corr(sub["y"], method="spearman")
                if pd.notna(ic):
                    ics.append(float(ic))
            ic_m = float(np.mean(ics)) if ics else float("nan")
            if np.isfinite(base) and np.isfinite(ic_m):
                drops.append(base - ic_m)
        out[c] = float(np.mean(drops)) if drops else 0.0
    return out

# 新浪交易日历缓存（进程内）：成功拉取后按 TTL 刷新，避免「首日失败永久空列表」或日历滞后不含当日
_SORTED_TRADE_DATES: list[str] | None = None
_SORTED_TRADE_DATES_AT: float = 0.0
_TRADE_CAL_TTL_SEC = float(os.environ.get("QUANT_TRADE_CAL_TTL_SEC", str(4 * 3600)))


def _ensure_eastmoney_no_proxy() -> None:
    """AkShare 访问东方财富前调用，避免走失效的系统 HTTP(S) 代理。"""
    from .config import ensure_eastmoney_no_proxy_if_configured

    ensure_eastmoney_no_proxy_if_configured()


def to_date_str(d: date | datetime | str) -> str:
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    return d.isoformat()


def latest_trade_date_from_series(dates: pd.Series) -> Optional[str]:
    if dates is None or len(dates) == 0:
        return None
    s = pd.to_datetime(dates).dt.strftime("%Y-%m-%d")
    return str(s.max())


def add_calendar_days(d: str, n: int) -> str:
    base = datetime.strptime(d[:10], "%Y-%m-%d").date()
    return (base + timedelta(days=n)).isoformat()


def get_sorted_a_share_trade_dates() -> list[str]:
    """返回升序 A 股交易日字符串列表 YYYY-MM-DD；失败时为空列表。"""
    global _SORTED_TRADE_DATES, _SORTED_TRADE_DATES_AT
    now = time_module.time()
    if (
        _SORTED_TRADE_DATES is not None
        and (now - _SORTED_TRADE_DATES_AT) < _TRADE_CAL_TTL_SEC
    ):
        return _SORTED_TRADE_DATES
    try:
        _ensure_eastmoney_no_proxy()
        from akshare.tool.trade_date_hist import tool_trade_date_hist_sina

        df = tool_trade_date_hist_sina()
        lst = sorted(
            pd.to_datetime(df["trade_date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        _SORTED_TRADE_DATES = lst
        _SORTED_TRADE_DATES_AT = now
        return lst
    except Exception:
        if _SORTED_TRADE_DATES is not None:
            return _SORTED_TRADE_DATES
        return []


def get_a_share_trade_date_set() -> set[str]:
    """交易日集合，便于 O(1) 判断（与日历列表同步，不单独永久缓存空集）。"""
    return set(get_sorted_a_share_trade_dates())


def is_a_share_intraday_session(now: datetime | None = None) -> bool:
    """
    是否处于 A 股连续竞价时段（9:30–11:30、13:00–15:00，本地时间）。
    非交易日返回 False。
    """
    n = now or datetime.now()
    if not is_a_share_trading_day(n.strftime("%Y-%m-%d")):
        return False
    t = n.time()
    morning = time(9, 30) <= t <= time(11, 30)
    afternoon = time(13, 0) <= t <= time(15, 0)
    return morning or afternoon


def is_a_share_trading_day(d: str) -> bool:
    """
    判断 ``d``（YYYY-MM-DD）是否为 A 股交易日。
    优先新浪日历；日历不可用时退回「周一至周五」近似（节假日可能被误判）。
    """
    ds = str(d).strip()[:10]
    cal = get_sorted_a_share_trade_dates()
    if cal:
        return ds in get_a_share_trade_date_set()
    try:
        dt = datetime.strptime(ds, "%Y-%m-%d").date()
    except ValueError:
        return False
    return dt.weekday() < 5


def count_trading_days_strictly_after_until(
    last_bar_date: str, prediction_date: str
) -> int:
    """
    统计满足 ``last_bar_date < d <= prediction_date`` 的交易日个数。
    用于度量「最新 K 线日期」相对「目标预测日」的滞后（不含 last_bar 当日）。
    """
    L = str(last_bar_date).strip()[:10]
    P = str(prediction_date).strip()[:10]
    if L >= P:
        return 0
    cal = get_sorted_a_share_trade_dates()
    if cal:
        return sum(1 for x in cal if L < x <= P)
    try:
        a = datetime.strptime(L, "%Y-%m-%d").date()
        b = datetime.strptime(P, "%Y-%m-%d").date()
    except ValueError:
        return 0
    n = 0
    cur = a + timedelta(days=1)
    while cur <= b:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


def is_kline_too_stale_vs_prediction(
    last_bar_date: str,
    prediction_date: str,
    max_trading_day_lag: int = 5,
) -> bool:
    """
    若最新 K 线日期早于目标预测日，且二者之间的交易日间隔 **大于**
    ``max_trading_day_lag``，视为停牌/数据过旧，应从截面池中剔除。
    """
    return count_trading_days_strictly_after_until(
        last_bar_date, prediction_date
    ) > max_trading_day_lag


def get_last_trading_date(as_of: date | datetime | str | None = None) -> str:
    """
    返回 as_of 当日或之前的最近一个 A 股交易日（YYYY-MM-DD）。
    优先使用新浪交易日历；失败时退回「跳过周六日」的近似工作日。
    """
    if as_of is None:
        base = datetime.now().date()
    elif isinstance(as_of, datetime):
        base = as_of.date()
    elif isinstance(as_of, date):
        base = as_of
    else:
        base = datetime.strptime(str(as_of)[:10], "%Y-%m-%d").date()

    try:
        _ensure_eastmoney_no_proxy()
        from akshare.tool.trade_date_hist import tool_trade_date_hist_sina

        df = tool_trade_date_hist_sina()
        trade_dates = sorted(
            pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
        )
        cut = base.strftime("%Y-%m-%d")
        past = [d for d in trade_dates if d <= cut]
        if past:
            return past[-1]
    except Exception:
        pass

    d = base
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def get_kline_incremental_end_trade_date(now: datetime | None = None) -> str:
    """
    增量日线应对齐的「最新交易日」（YYYY-MM-DD）。

    - 非交易日：与 ``get_last_trading_date(as_of=now)`` 一致。
    - 交易日收盘前：不把当日未收盘 K 当作应对齐目标，对齐到截至「昨自然日」的最近交易日。
    - 交易日收盘起（默认本地 15:00，可用 ``QUANT_MARKET_CLOSE_HOUR`` / ``QUANT_MARKET_CLOSE_MINUTE`` 覆盖）：
      将当日纳入增量；若新浪日历滞后，仍与自然日今天取较大者，便于盘后立刻拉当日收盘线。
    """
    n = now or datetime.now()
    today_str = n.strftime("%Y-%m-%d")
    if not is_a_share_trading_day(today_str):
        return get_last_trading_date(as_of=n)

    try:
        close_h = int(str(os.environ.get("QUANT_MARKET_CLOSE_HOUR", "15")).strip())
    except ValueError:
        close_h = 15
    try:
        close_m = int(str(os.environ.get("QUANT_MARKET_CLOSE_MINUTE", "0")).strip())
    except ValueError:
        close_m = 0
    close_h = max(0, min(23, close_h))
    close_m = max(0, min(59, close_m))

    if n.time() >= time(close_h, close_m):
        cal_last = get_last_trading_date(as_of=n)
        return max(today_str, cal_last)

    prev_cal = (n.date() - timedelta(days=1)).isoformat()
    return get_last_trading_date(as_of=prev_cal)


def next_trade_day_after(d: str) -> str | None:
    """
    给定YYYYMMDD或YYYY-MM-DD的「当前选股所属交易日」d，返回其后的下一个A股交易日（YYYY-MM-DD）。

    优先使用 akshare 新浪交易日历（含休市）；网络异常或日历无更晚日期时，退回为「跳过周六日」的近似工作日。
    d 无法解析为日期时返回 None。
    """
    try:
        base = datetime.strptime(d[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

    try:
        _ensure_eastmoney_no_proxy()
        from akshare.tool.trade_date_hist import tool_trade_date_hist_sina

        df = tool_trade_date_hist_sina()
        norm = (
            pd.to_datetime(df["trade_date"], errors="coerce")
            .dropna()
            .sort_values()
        )
        for ts in norm:
            td = pd.Timestamp(ts).date()
            if td > base:
                return td.isoformat()
    except Exception:
        pass

    nxt = base + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.isoformat()
