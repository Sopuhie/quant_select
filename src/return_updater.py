"""根据日线回填 daily_selections 的次日与各持有期收盘收益。

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

# (库字段名, 选股日后第 N 个交易日偏移)
RETURN_OFFSETS: tuple[tuple[str, int], ...] = (
    ("next_day_return", 1),
    ("hold_5d_return", 5),
    ("hold_10d_return", 10),
    ("hold_60d_return", 60),
)

_RETURN_COLS = tuple(c for c, _ in RETURN_OFFSETS)


def _norm_code(code: str) -> str:
    return str(code).strip().zfill(6)


def _returns_from_hist(
    hist: Any,
    trade_date: str,
    close_hint: float | None,
) -> dict[str, float | None]:
    """按交易日偏移计算各持有期收益率（收盘价 / 选股日收盘 - 1）。"""
    out: dict[str, float | None] = {col: None for col in _RETURN_COLS}
    if hist is None or getattr(hist, "empty", True):
        return out
    dates = hist["date"].astype(str).str[:10].tolist()
    tid = str(trade_date).strip()[:10]
    if tid not in dates:
        return out
    idx = dates.index(tid)
    if close_hint is not None and close_hint > 0:
        close0 = float(close_hint)
    else:
        close0 = float(hist.iloc[idx]["close"])
    if close0 <= 0:
        return out
    for col, offset in RETURN_OFFSETS:
        if idx + offset < len(hist):
            out[col] = float(hist.iloc[idx + offset]["close"]) / close0 - 1.0
    return out


def _pending_rows(
    db_path: Path | None,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not fields:
        fields = _RETURN_COLS
    where = " OR ".join(f"{col} IS NULL" for col in fields)
    cols_sql = ", ".join(("trade_date", "stock_code", "close_price", *_RETURN_COLS))
    with get_connection(db_path) as conn:
        cur = conn.execute(
            f"""
            SELECT {cols_sql}
            FROM daily_selections
            WHERE {where}
            ORDER BY trade_date
            """
        )
        return [dict(r) for r in cur.fetchall()]


def _row_still_needs_remote(
    r: dict[str, Any],
    computed: dict[str, float | None],
    update_cols: tuple[str, ...],
) -> bool:
    for col in update_cols:
        if r.get(col) is None and computed.get(col) is None:
            return True
    return False


def _refresh_with_grouped_fetch(
    pending: list[dict[str, Any]],
    db_path: Path | None,
    start_date: str,
    update_cols: tuple[str, ...],
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

            computed = _returns_from_hist(
                hist_local if not hist_local.empty else None,
                td,
                close_hint,
            )

            if _row_still_needs_remote(r, computed, update_cols):
                if hist_remote is None:
                    hist_remote = fetch_daily_hist(
                        code, start_date=start_date, timeout=timeout
                    )
                computed_remote = _returns_from_hist(hist_remote, td, close_hint)
                for col in update_cols:
                    if computed.get(col) is None:
                        computed[col] = computed_remote.get(col)

            kwargs: dict[str, float] = {}
            for col in update_cols:
                if r.get(col) is None and computed.get(col) is not None:
                    kwargs[col] = float(computed[col])
            if not kwargs:
                continue
            rc = update_selection_returns(
                r["trade_date"],
                _norm_code(r["stock_code"]),
                next_day_return=kwargs.get("next_day_return"),
                hold_5d_return=kwargs.get("hold_5d_return"),
                hold_10d_return=kwargs.get("hold_10d_return"),
                hold_60d_return=kwargs.get("hold_60d_return"),
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
    pending = _pending_rows(db_path, ("next_day_return",))
    return _refresh_with_grouped_fetch(pending, db_path, start_date, ("next_day_return",))


def update_hold_5d_returns(
    db_path: Path | None = None,
    start_date: str = "20150101",
    max_workers: int | None = None,
) -> int:
    _ = max_workers
    pending = _pending_rows(db_path, ("hold_5d_return",))
    return _refresh_with_grouped_fetch(pending, db_path, start_date, ("hold_5d_return",))


def update_hold_10d_returns(
    db_path: Path | None = None,
    start_date: str = "20150101",
    max_workers: int | None = None,
) -> int:
    _ = max_workers
    pending = _pending_rows(db_path, ("hold_10d_return",))
    return _refresh_with_grouped_fetch(pending, db_path, start_date, ("hold_10d_return",))


def update_hold_60d_returns(
    db_path: Path | None = None,
    start_date: str = "20150101",
    max_workers: int | None = None,
) -> int:
    _ = max_workers
    pending = _pending_rows(db_path, ("hold_60d_return",))
    return _refresh_with_grouped_fetch(pending, db_path, start_date, ("hold_60d_return",))


def update_all_returns(
    db_path: Path | None = None,
    start_date: str = "20150101",
    max_workers: int | None = None,
) -> dict[str, int]:
    _ = max_workers
    pending = _pending_rows(db_path, _RETURN_COLS)
    n = _refresh_with_grouped_fetch(pending, db_path, start_date, _RETURN_COLS)
    return {"rows_updated": n}
