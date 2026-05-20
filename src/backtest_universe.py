"""回测股票池：时点可得（PIT）universe 与样本外校验。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .config import MIN_HISTORY_BARS


def _normalize_stock_code(code: str) -> str:
    return str(code).strip().zfill(6)


def _board_allowed(
    code: str,
    *,
    include_300: bool,
    include_688: bool,
) -> bool:
    c = _normalize_stock_code(code)
    if not include_300 and (c.startswith("300") or c.startswith("301")):
        return False
    if not include_688 and c.startswith("688"):
        return False
    return True


def build_point_in_time_universe_fn(
    pairs: list[tuple[str, str]],
    bars: pd.DataFrame,
    *,
    min_history_bars: int = MIN_HISTORY_BARS,
    max_stocks: int | None = None,
    include_300: bool = False,
    include_688: bool = False,
) -> Callable[[str], list[tuple[str, str]]]:
    """
    信号日 ``d`` 仅纳入：本地 K 线在 ``d`` 有 bar、且此前至少 ``min_history_bars`` 根 K 线的标的。
    顺序与 ``pairs`` 一致，并受 ``max_stocks`` 截断。
    """
    name_map = {_normalize_stock_code(c): str(n) for c, n in pairs}
    ordered_codes = [_normalize_stock_code(c) for c, _ in pairs]
    avail_by_date: dict[str, list[str]] = {}

    if bars is None or bars.empty:
        def _empty(_d: str) -> list[tuple[str, str]]:
            return []

        return _empty

    work = bars.copy()
    work["stock_code"] = work["stock_code"].astype(str).str.zfill(6)
    work["date"] = work["date"].astype(str).str[:10]
    need = max(int(min_history_bars), 1)

    for code, grp in work.groupby("stock_code", sort=False):
        if code not in name_map:
            continue
        if not _board_allowed(code, include_300=include_300, include_688=include_688):
            continue
        g = grp.sort_values("date").reset_index(drop=True)
        if len(g) < need:
            continue
        for i in range(need - 1, len(g)):
            d = str(g.iloc[i]["date"])
            vol = float(pd.to_numeric(g.iloc[i].get("volume"), errors="coerce") or 0.0)
            if vol <= 0:
                continue
            avail_by_date.setdefault(d, []).append(code)

    def universe_fn(signal_date: str) -> list[tuple[str, str]]:
        d = str(signal_date).strip()[:10]
        avail = set(avail_by_date.get(d, ()))
        out: list[tuple[str, str]] = []
        for code in ordered_codes:
            if code in avail:
                out.append((code, name_map[code]))
            if max_stocks is not None and len(out) >= int(max_stocks):
                break
        return out

    return universe_fn


def build_fixed_snapshot_universe_fn(
    pairs: list[tuple[str, str]],
) -> Callable[[str], list[tuple[str, str]]]:
    """旧行为：全回测期固定使用同一股票列表（易产生成分前视）。"""

    def universe_fn(_signal_date: str) -> list[tuple[str, str]]:
        return list(pairs)

    return universe_fn


def assert_sample_out_backtest(
    backtest_start: str,
    *,
    train_end_date: str | None,
    active_version: str | None = None,
) -> None:
    """
    若激活模型的 ``train_end_date`` >= 回测 ``start_date``，视为样本内回测并拒绝运行。
    """
    if not train_end_date:
        return
    bs = str(backtest_start).strip()[:10]
    te = str(train_end_date).strip()[:10]
    if te >= bs:
        ver = active_version or "unknown"
        raise SystemExit(
            f"样本外校验失败：激活模型版本 {ver!r} 的 train_end_date={te} "
            f"不早于回测开始日 {bs}。\n"
            "请使用更早的 --train-end-date 重训，或缩短回测起点，"
            "或使用 scripts/walkforward_backtest.py 做滚动重训回测。"
        )


def trading_day_before(sorted_dates: list[str], d: str) -> str | None:
    """``sorted_dates`` 升序；返回严格早于 ``d`` 的最后一个交易日。"""
    td = str(d).strip()[:10]
    prev: str | None = None
    for x in sorted_dates:
        xs = str(x).strip()[:10]
        if xs >= td:
            break
        prev = xs
    return prev


def split_trade_dates_windows(
    trade_dates: list[str],
    *,
    retrain_every: int,
) -> list[tuple[str, str]]:
    """将交易日序列切为若干 [start, end] 闭区间窗口（每窗至少 1 日）。"""
    if not trade_dates:
        return []
    step = max(1, int(retrain_every))
    uds = sorted({str(d).strip()[:10] for d in trade_dates})
    windows: list[tuple[str, str]] = []
    i = 0
    while i < len(uds):
        j = min(i + step, len(uds)) - 1
        windows.append((uds[i], uds[j]))
        i = j + 1
    return windows
