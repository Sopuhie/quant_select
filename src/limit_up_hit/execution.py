# -*- coding: utf-8 -*-
"""打板模块：理论涨跌停价、T+1 打板买入与 T+2 卖出模拟（纯日线）。"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from src.factor_calculator import is_bar_limit_up, is_bar_suspended

from .config import (
    LUH_BUY_REQUIRE_CLOSE_LIMIT,
    LUH_HOLD_PLAN,
    LUH_SELL_OFFSET,
    LUH_T2_RIDE_MIN_PCT,
    LUH_T2_STOP_LOSS,
    LUH_T2_TAKE_PROFIT_TIER1,
    LUH_T2_TAKE_PROFIT_TIER2,
    LUH_T2_TAKE_PROFIT_TIER2_LOCK,
)
from .review_prices import resolve_t1_t2_dates

ORDER_STATUS_HOLDING = "HOLDING"
ORDER_STATUS_CLOSED = "CLOSED"
ORDER_STATUS_SKIPPED = "SKIPPED"

_MIN_SELL_LAG_TRADING_DAYS = 2
_LIMIT_PRICE_EPS = 0.015


def _limit_move_ratio(stock_code: str) -> float:
    c = str(stock_code).strip().zfill(6)
    if c.startswith(("300", "301", "688")):
        return 0.20
    if c.startswith(("8", "4")):
        return 0.30
    return 0.10


def theoretical_limit_up_price(prev_close: float, stock_code: str) -> float:
    """理论涨停价（四舍五入到分）。"""
    pct = _limit_move_ratio(stock_code)
    return float(round(float(prev_close) * (1.0 + pct), 2))


def theoretical_limit_down_price(prev_close: float, stock_code: str) -> float:
    pct = _limit_move_ratio(stock_code)
    return float(round(float(prev_close) * (1.0 - pct), 2))


def _price_at_limit(px: float, limit_px: float) -> bool:
    return np.isfinite(px) and np.isfinite(limit_px) and abs(px - limit_px) <= _LIMIT_PRICE_EPS


def is_one_word_limit_up(
    bar: dict[str, float | None],
    prev_close: float,
    stock_code: str,
) -> bool:
    """一字涨停：开高低收均贴涨停价，开盘价无法买入。"""
    lim_up = theoretical_limit_up_price(prev_close, stock_code)
    o = bar.get("open")
    h = bar.get("high")
    lo = bar.get("low")
    if o is None or h is None or lo is None:
        return True
    try:
        o_f, h_f, l_f = float(o), float(h), float(lo)
    except (TypeError, ValueError):
        return True
    if not all(np.isfinite(x) for x in (o_f, h_f, l_f)):
        return True
    return (
        _price_at_limit(h_f, lim_up)
        and _price_at_limit(l_f, lim_up)
        and _price_at_limit(o_f, lim_up)
    )


def is_one_word_limit_down(
    bar: dict[str, float | None],
    prev_close: float,
    stock_code: str,
) -> bool:
    lim_dn = theoretical_limit_down_price(prev_close, stock_code)
    o = bar.get("open")
    h = bar.get("high")
    lo = bar.get("low")
    if o is None or h is None or lo is None:
        return False
    try:
        o_f, h_f, l_f = float(o), float(h), float(lo)
    except (TypeError, ValueError):
        return False
    if not all(np.isfinite(x) for x in (o_f, h_f, l_f)):
        return False
    return (
        _price_at_limit(h_f, lim_dn)
        and _price_at_limit(l_f, lim_dn)
        and _price_at_limit(o_f, lim_dn)
    )


def resolve_t1_board_entry(
    signal_close: float,
    t1_bar: dict[str, float | None],
    stock_code: str,
    *,
    require_close_limit: bool | None = None,
) -> tuple[float | None, str | None]:
    """
    T+1 打板买入：非一字且 T+1 收盘涨停时，以理论涨停价成交。

    Returns:
        (买入价, None) 或 (None, skip_reason)
    """
    sig = float(signal_close)
    if sig <= 0:
        return None, "invalid_signal_close"

    t1_close = t1_bar.get("close")
    if t1_close is None or not np.isfinite(float(t1_close)):
        return None, "await_t1_kline"

    close_f = float(t1_close)
    if is_one_word_limit_up(t1_bar, sig, stock_code):
        return None, "t1_one_word_limit_up"

    need_close_limit = (
        LUH_BUY_REQUIRE_CLOSE_LIMIT
        if require_close_limit is None
        else require_close_limit
    )
    if need_close_limit:
        if not is_bar_limit_up(close_f, sig, stock_code):
            return None, "t1_not_limit_up_close"
    else:
        t1_high = t1_bar.get("high")
        if t1_high is None or not np.isfinite(float(t1_high)):
            return None, "await_t1_high"
        lim_up = theoretical_limit_up_price(sig, stock_code)
        if float(t1_high) < lim_up - _LIMIT_PRICE_EPS:
            return None, "t1_high_miss_limit"

    buy_px = theoretical_limit_up_price(sig, stock_code)
    return buy_px, None


def fetch_post_signal_ohlc(
    conn: sqlite3.Connection,
    stock_code: str,
    signal_trade_date: str,
) -> dict[str, dict[str, float | None]]:
    code = str(stock_code).strip().zfill(6)
    t1, t2 = resolve_t1_t2_dates(signal_trade_date, conn)
    out: dict[str, dict[str, float | None]] = {
        "t1": {"open": None, "high": None, "low": None, "close": None},
        "t2": {"open": None, "high": None, "low": None, "close": None},
    }
    if not t1:
        return out
    dates = [t1]
    if t2:
        dates.append(t2)
    placeholders = ",".join("?" * len(dates))
    cur = conn.execute(
        f"""
        SELECT date, open, high, low, close
        FROM stock_daily_kline
        WHERE stock_code = ? AND date IN ({placeholders})
        """,
        [code, *dates],
    )
    by_date: dict[str, dict[str, float]] = {}
    for row in cur.fetchall():
        d = str(row[0]).strip()[:10]
        try:
            o, hi, lo, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        except (TypeError, ValueError):
            continue
        if all(np.isfinite(x) for x in (o, hi, lo, c)):
            by_date[d] = {"open": o, "high": hi, "low": lo, "close": c}
    if t1 in by_date:
        out["t1"] = dict(by_date[t1])
    if t2 and t2 in by_date:
        out["t2"] = dict(by_date[t2])
    return out


def evaluate_board_exit(
    buy_price: float,
    stock_code: str,
    *,
    t1_bar: dict[str, float | None],
    t2_bar: dict[str, float | None],
    t1_date: str | None,
    t2_date: str | None,
    sell_offset: int | None = None,
) -> dict[str, Any]:
    """T+2 及以后纯日线平仓评估（打板专用止盈止损）。"""
    offset = int(sell_offset if sell_offset is not None else LUH_SELL_OFFSET)
    offset = max(1, min(2, offset))
    buy_price = float(buy_price)
    stop_px = buy_price * (1.0 - float(LUH_T2_STOP_LOSS))

    t1_close = t1_bar.get("close")
    if t1_close is None or not np.isfinite(float(t1_close)):
        return {
            "status": ORDER_STATUS_HOLDING,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "exit_reason": "await_t1_kline",
        }

    t2_close = t2_bar.get("close")
    if t2_close is None or not np.isfinite(float(t2_close)) or not t2_date:
        return {
            "status": ORDER_STATUS_HOLDING,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 1,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "exit_reason": "await_t2_kline",
        }

    t1_close_f = float(t1_close)
    t2_close_f = float(t2_close)
    t1_prev = buy_price
    t1_pct = (t1_close_f - buy_price) / buy_price if buy_price > 0 else 0.0

    if is_one_word_limit_down(t2_bar, t1_close_f, stock_code):
        return {
            "status": ORDER_STATUS_HOLDING,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 1,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "exit_reason": "t2_one_word_limit_down",
        }

    t2_high = t2_bar.get("high")
    t2_high_f: float | None = None
    t2_high_pct = 0.0
    if t2_high is not None and np.isfinite(float(t2_high)):
        t2_high_f = float(t2_high)
        t2_high_pct = (t2_high_f - buy_price) / buy_price if buy_price > 0 else 0.0

    tier1 = float(LUH_T2_TAKE_PROFIT_TIER1)
    tier2 = float(LUH_T2_TAKE_PROFIT_TIER2)
    tier2_lock = float(LUH_T2_TAKE_PROFIT_TIER2_LOCK)
    ride_thr = float(LUH_T2_RIDE_MIN_PCT)
    hold_days = _MIN_SELL_LAG_TRADING_DAYS

    if t2_high_f is not None:
        if t2_high_pct >= tier1:
            sell_px = (t2_high_f + t2_close_f) / 2.0
            pnl = (sell_px - buy_price) / buy_price
            return {
                "status": ORDER_STATUS_CLOSED,
                "sell_date": t2_date,
                "sell_price": sell_px,
                "hold_days": hold_days,
                "pnl_ratio": pnl,
                "stop_loss_triggered": 0,
                "exit_reason": "t2_take_profit_tier1",
            }
        if t2_high_pct >= tier2:
            sell_px = buy_price * (1.0 + tier2_lock)
            pnl = (sell_px - buy_price) / buy_price
            return {
                "status": ORDER_STATUS_CLOSED,
                "sell_date": t2_date,
                "sell_price": sell_px,
                "hold_days": hold_days,
                "pnl_ratio": pnl,
                "stop_loss_triggered": 0,
                "exit_reason": "t2_take_profit_tier2",
            }

    if t2_close_f < stop_px:
        pnl = (t2_close_f - buy_price) / buy_price
        return {
            "status": ORDER_STATUS_CLOSED,
            "sell_date": t2_date,
            "sell_price": t2_close_f,
            "hold_days": hold_days,
            "pnl_ratio": pnl,
            "stop_loss_triggered": 1,
            "exit_reason": "t2_stop_loss_close",
        }

    if (
        offset >= 2
        and t1_pct >= ride_thr
        and is_bar_limit_up(t2_close_f, t1_close_f, stock_code)
    ):
        exit_reason = "t2_limit_up_ride_exit"
    else:
        exit_reason = "t2_close_exit"

    pnl = (t2_close_f - buy_price) / buy_price
    return {
        "status": ORDER_STATUS_CLOSED,
        "sell_date": t2_date,
        "sell_price": t2_close_f,
        "hold_days": hold_days,
        "pnl_ratio": pnl,
        "stop_loss_triggered": 0,
        "exit_reason": exit_reason,
    }


def evaluate_board_trade(
    signal_close: float,
    stock_code: str,
    *,
    t1_bar: dict[str, float | None],
    t2_bar: dict[str, float | None],
    t1_date: str | None,
    t2_date: str | None,
    sell_offset: int | None = None,
) -> dict[str, Any]:
    """完整打板回合：T+1 买入 + T+2 卖出。"""
    buy_px, skip = resolve_t1_board_entry(signal_close, t1_bar, stock_code)
    if buy_px is None:
        return {
            "status": ORDER_STATUS_SKIPPED,
            "buy_price": None,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "exit_reason": skip or "entry_skipped",
        }
    exit_info = evaluate_board_exit(
        buy_px,
        stock_code,
        t1_bar=t1_bar,
        t2_bar=t2_bar,
        t1_date=t1_date,
        t2_date=t2_date,
        sell_offset=sell_offset,
    )
    exit_info["buy_price"] = buy_px
    return exit_info


def sync_luh_orders_for_signal_day(
    conn: sqlite3.Connection,
    signal_trade_date: str,
    rows: list[dict[str, Any]],
    *,
    commit: bool = True,
) -> dict[str, Any]:
    """为信号日选股同步打板模拟订单。"""
    from .db import ensure_luh_tables, upsert_luh_order

    ensure_luh_tables(conn)
    td = str(signal_trade_date).strip()[:10]
    orders_out: list[dict[str, Any]] = []
    closed_n = holding_n = skipped_n = stop_n = 0

    for row in rows:
        code = str(row.get("stock_code", "")).strip().zfill(6)
        signal_close = float(row.get("close_price") or 0)
        bars = fetch_post_signal_ohlc(conn, code, td)
        t1_date, t2_date = resolve_t1_t2_dates(td, conn)

        exit_info = evaluate_board_trade(
            signal_close,
            code,
            t1_bar=bars["t1"],
            t2_bar=bars["t2"],
            t1_date=t1_date,
            t2_date=t2_date,
        )
        buy_price = exit_info.get("buy_price")
        order = {
            "stock_code": code,
            "stock_name": row.get("stock_name"),
            "buy_date": td,
            "buy_price": buy_price if buy_price is not None else signal_close,
            "sell_date": exit_info.get("sell_date"),
            "sell_price": exit_info.get("sell_price"),
            "hold_days": exit_info.get("hold_days"),
            "pnl_ratio": exit_info.get("pnl_ratio"),
            "status": exit_info.get("status"),
            "stop_loss_triggered": exit_info.get("stop_loss_triggered"),
            "signal_rank": row.get("rank"),
            "board_score": row.get("board_score"),
            "board_streak": row.get("board_streak"),
            "exit_reason": exit_info.get("exit_reason"),
            "hold_plan": row.get("hold_plan") or LUH_HOLD_PLAN,
        }
        upsert_luh_order(conn, order)
        row["is_executed"] = 1 if order.get("status") == ORDER_STATUS_CLOSED else 0
        orders_out.append({**order, **exit_info})

        st = order.get("status")
        if st == ORDER_STATUS_CLOSED:
            closed_n += 1
        elif st == ORDER_STATUS_HOLDING:
            holding_n += 1
        elif st == ORDER_STATUS_SKIPPED:
            skipped_n += 1
        if order.get("stop_loss_triggered"):
            stop_n += 1

    if commit:
        conn.commit()

    return {
        "buy_date": td,
        "orders": orders_out,
        "closed_count": closed_n,
        "holding_count": holding_n,
        "skipped_count": skipped_n,
        "stop_loss_count": stop_n,
    }
