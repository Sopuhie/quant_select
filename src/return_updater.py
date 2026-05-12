"""根据日线回填 daily_selections 的次日与第 5 个交易日收盘收益。

优先使用本地 ``stock_daily_kline``；若本地缺交易日或后续 K 线不足，再请求 AkShare。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import AKSHARE_REQUEST_TIMEOUT
from .data_fetcher import fetch_daily_hist
from .database import fetch_stock_daily_bars_until, get_connection, update_selection_returns
from .utils import get_kline_incremental_end_trade_date


def _norm_code(code: str) -> str:
    return str(code).strip().zfill(6)


def _returns_from_hist(
    hist: Any,
    trade_date: str,
    close_hint: float | None,
) -> tuple[float | None, float | None]:
    """返回 (次日收益, 第5个交易日收益)。K 线为交易日序列。"""
    if hist is None or getattr(hist, "empty", True):
        return None, None
    dates = hist["date"].astype(str).str[:10].tolist()
    tid = str(trade_date).strip()[:10]
    if tid not in dates:
        return None, None
    idx = dates.index(tid)
    if close_hint is not None and close_hint > 0:
        close0 = float(close_hint)
    else:
        close0 = float(hist.iloc[idx]["close"])
    if close0 <= 0:
        return None, None
    nd = h5 = None
    if idx + 1 < len(hist):
        nd = float(hist.iloc[idx + 1]["close"]) / close0 - 1.0
    if idx + 5 < len(hist):
        h5 = float(hist.iloc[idx + 5]["close"]) / close0 - 1.0
    return nd, h5


def _pending_rows(
    db_path: Path | None,
    only_next: bool,
    only_h5: bool,
) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        if only_next and not only_h5:
            where = "next_day_return IS NULL"
        elif only_h5 and not only_next:
            where = "hold_5d_return IS NULL"
        else:
            where = "(next_day_return IS NULL OR hold_5d_return IS NULL)"
        cur = conn.execute(
            f"""
            SELECT trade_date, stock_code, close_price, next_day_return, hold_5d_return
            FROM daily_selections
            WHERE {where}
            ORDER BY trade_date
            """
        )
        return [dict(r) for r in cur.fetchall()]


def _row_still_needs_remote(
    r: dict[str, Any],
    nd: float | None,
    h5: float | None,
    *,
    update_next: bool,
    update_h5: bool,
) -> bool:
    """本地已算出的收益若仍缺库内待填字段，则需要在线补 K 线。"""
    if update_next and r.get("next_day_return") is None and nd is None:
        return True
    if update_h5 and r.get("hold_5d_return") is None and h5 is None:
        return True
    return False


def _refresh_with_grouped_fetch(
    pending: list[dict[str, Any]],
    db_path: Path | None,
    start_date: str,
    update_next: bool,
    update_h5: bool,
) -> int:
    if not pending:
        return 0
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pending:
        by_code[_norm_code(r["stock_code"])].append(r)
    timeout = AKSHARE_REQUEST_TIMEOUT
    end_anchor = get_kline_incremental_end_trade_date()
    updated = 0
    for code, rows in by_code.items():
        hist_local = fetch_stock_daily_bars_until(code, end_anchor, db_path=db_path)
        hist_remote = None

        for r in rows:
            close_hint = r.get("close_price")
            td = r["trade_date"]

            nd, h5 = _returns_from_hist(
                hist_local if not hist_local.empty else None,
                td,
                close_hint,
            )

            if _row_still_needs_remote(r, nd, h5, update_next=update_next, update_h5=update_h5):
                if hist_remote is None:
                    hist_remote = fetch_daily_hist(
                        code, start_date=start_date, timeout=timeout
                    )
                nd_r, h5_r = _returns_from_hist(hist_remote, td, close_hint)
                if nd is None:
                    nd = nd_r
                if h5 is None:
                    h5 = h5_r

            kwargs: dict[str, float] = {}
            if update_next and r.get("next_day_return") is None and nd is not None:
                kwargs["next_day_return"] = nd
            if update_h5 and r.get("hold_5d_return") is None and h5 is not None:
                kwargs["hold_5d_return"] = h5
            if not kwargs:
                continue
            rc = update_selection_returns(
                r["trade_date"],
                _norm_code(r["stock_code"]),
                next_day_return=kwargs.get("next_day_return"),
                hold_5d_return=kwargs.get("hold_5d_return"),
                db_path=db_path,
            )
            updated += max(rc, 0)
    return updated


def update_next_day_returns(
    db_path: Path | None = None,
    start_date: str = "20150101",
    max_workers: int | None = None,
) -> int:
    _ = max_workers
    pending = _pending_rows(db_path, only_next=True, only_h5=False)
    return _refresh_with_grouped_fetch(pending, db_path, start_date, True, False)


def update_hold_5d_returns(
    db_path: Path | None = None,
    start_date: str = "20150101",
    max_workers: int | None = None,
) -> int:
    _ = max_workers
    pending = _pending_rows(db_path, only_next=False, only_h5=True)
    return _refresh_with_grouped_fetch(pending, db_path, start_date, False, True)


def update_all_returns(
    db_path: Path | None = None,
    start_date: str = "20150101",
    max_workers: int | None = None,
) -> dict[str, int]:
    _ = max_workers
    pending = _pending_rows(db_path, only_next=False, only_h5=False)
    n = _refresh_with_grouped_fetch(pending, db_path, start_date, True, True)
    return {"rows_updated": n}
