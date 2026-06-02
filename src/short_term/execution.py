# -*- coding: utf-8 -*-
"""
纯日线短线交易执行引擎（A 股 T+1 交割合规）。

- 信号：T 日收盘确认
- 买入：T+1 非对称入场
- 卖出：最早 T+2（当日买入不可当日卖出）
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from .config import (
    SHORT_CLOSE_STOP_RATIO,
    SHORT_ENTRY_DIP_OPEN_THRESHOLD,
    SHORT_ENTRY_MAX_CHASE,
    SHORT_ENTRY_MAX_CHASE_HARD_CAP,
    SHORT_ENTRY_MIN_GAP,
    SHORT_HOLD_PLAN,
    SHORT_MEDIOCRE_STOP_RATIO,
    SHORT_SELL_OFFSET,
    SHORT_STOP_LOSS_RATIO,
    SHORT_T1_STRONG_CLOSE_PCT,
    SHORT_T1_TAKE_PROFIT_TIER1_PCT,
    SHORT_T1_TAKE_PROFIT_TIER2_LOCK,
    SHORT_T1_TAKE_PROFIT_TIER2_PCT,
)
from .review_prices import resolve_t1_t2_dates

ORDER_STATUS_HOLDING = "HOLDING"
ORDER_STATUS_CLOSED = "CLOSED"
ORDER_STATUS_SKIPPED = "SKIPPED"

# 最早可卖出日相对信号日的偏移（T+1 买，T+2 卖）
_MIN_SELL_LAG_TRADING_DAYS = 2


def resolve_t1_entry_price(
    signal_close: float,
    t1_bar: dict[str, float | None],
    *,
    max_chase: float | None = None,
    min_gap: float | None = None,
) -> tuple[float | None, str | None]:
    """
    T+1 非对称入场：允许高开至 5.5% 的人气股，并模拟分时低吸修正成本。

    Returns:
        (买入价, None) 或 (None, skip_reason)。
    """
    chase = min(
        float(SHORT_ENTRY_MAX_CHASE if max_chase is None else max_chase),
        float(SHORT_ENTRY_MAX_CHASE_HARD_CAP),
    )
    dip_thr = float(SHORT_ENTRY_DIP_OPEN_THRESHOLD)
    gap = float(SHORT_ENTRY_MIN_GAP if min_gap is None else min_gap)
    sig = float(signal_close)
    if sig <= 0:
        return None, "invalid_signal_close"

    t1_open = t1_bar.get("open")
    if t1_open is None or not np.isfinite(float(t1_open)):
        return None, "await_t1_open"

    open_px = float(t1_open)
    lo_bound = sig * (1.0 + gap)
    hi_bound = sig * (1.0 + chase)
    if open_px > hi_bound:
        return None, "t1_open_chase_rejected"
    if open_px < lo_bound:
        return None, "t1_open_gap_down_rejected"

    if open_px <= sig * (1.0 + dip_thr):
        return open_px, None

    t1_low = t1_bar.get("low")
    if t1_low is not None and np.isfinite(float(t1_low)):
        buy_px = (open_px + float(t1_low)) / 2.0
        return buy_px, None
    return open_px, None


def stop_loss_trigger_price(buy_price: float) -> float:
    """强势股收盘破位止损线（默认 -6%）。"""
    return float(buy_price) * (1.0 - float(SHORT_CLOSE_STOP_RATIO))


def mediocre_stop_trigger_price(buy_price: float) -> float:
    """平庸股非对称收盘止损线（默认 -4%）。"""
    return float(buy_price) * (1.0 - float(SHORT_MEDIOCRE_STOP_RATIO))


def legacy_intraday_stop_price(buy_price: float) -> float:
    """旧版盘中 -3% 止损价（仅兼容引用）。"""
    return float(buy_price) * (1.0 - float(SHORT_STOP_LOSS_RATIO))


def fetch_post_signal_ohlc(
    conn: sqlite3.Connection,
    stock_code: str,
    signal_trade_date: str,
) -> dict[str, dict[str, float | None]]:
    """读取信号日之后的 T+1 / T+2 日线 OHLC。"""
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
            o = float(row[1])
            hi = float(row[2])
            lo = float(row[3])
            c = float(row[4])
        except (TypeError, ValueError):
            continue
        if all(np.isfinite(x) for x in (o, hi, lo, c)):
            by_date[d] = {"open": o, "high": hi, "low": lo, "close": c}

    if t1 in by_date:
        out["t1"] = dict(by_date[t1])
    if t2 and t2 in by_date:
        out["t2"] = dict(by_date[t2])
    return out


def _closed_exit(
    *,
    buy_price: float,
    sell_px: float,
    sell_date: str,
    hold_days: int,
    stop_px: float,
    exit_reason: str,
    stop_loss_triggered: int = 0,
    t1_low_f: float | None = None,
) -> dict[str, Any]:
    pnl = (sell_px - buy_price) / buy_price if buy_price > 0 else None
    out: dict[str, Any] = {
        "status": ORDER_STATUS_CLOSED,
        "sell_date": sell_date,
        "sell_price": sell_px,
        "hold_days": hold_days,
        "pnl_ratio": pnl,
        "stop_loss_triggered": stop_loss_triggered,
        "stop_loss_price": stop_px,
        "exit_reason": exit_reason,
    }
    if t1_low_f is not None:
        out["t1_low"] = t1_low_f
    return out


def evaluate_daily_exit(
    buy_price: float,
    *,
    t1_bar: dict[str, float | None],
    t2_bar: dict[str, float | None],
    t1_date: str | None,
    t2_date: str | None,
    sell_offset: int | None = None,
) -> dict[str, Any]:
    """
    纯日线模拟平仓（A 股 T+1 交割：T+1 只买不卖，最早 T+2 卖出）。

    T+2 评估顺序：
    1. T+2 双阶梯动态止盈
    2. T+2 非对称收盘止损
    3. offset=2 且 T+1 强势 → T+2 收盘骑乘，否则 T+2 收盘
    """
    offset = int(sell_offset if sell_offset is not None else SHORT_SELL_OFFSET)
    offset = max(1, min(2, offset))
    buy_price = float(buy_price)
    strong_stop_px = stop_loss_trigger_price(buy_price)
    mediocre_stop_px = mediocre_stop_trigger_price(buy_price)

    t1_close = t1_bar.get("close")
    t1_low = t1_bar.get("low")
    t1_low_f: float | None = None
    if t1_low is not None and np.isfinite(float(t1_low)):
        t1_low_f = float(t1_low)

    if t1_close is None or not np.isfinite(float(t1_close)):
        return {
            "status": ORDER_STATUS_HOLDING,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "stop_loss_price": strong_stop_px,
            "exit_reason": "await_t1_kline",
        }

    t1_close_f = float(t1_close)
    t1_close_pct = (t1_close_f - buy_price) / buy_price if buy_price > 0 else 0.0
    strong_thr = float(SHORT_T1_STRONG_CLOSE_PCT)

    t2_close = t2_bar.get("close")
    if t2_close is None or not np.isfinite(float(t2_close)) or not t2_date:
        return {
            "status": ORDER_STATUS_HOLDING,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 1,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "stop_loss_price": strong_stop_px,
            "exit_reason": "await_t2_kline",
            "t1_low": t1_low_f,
        }

    t2_close_f = float(t2_close)

    tier1_pct = float(SHORT_T1_TAKE_PROFIT_TIER1_PCT)
    tier2_pct = float(SHORT_T1_TAKE_PROFIT_TIER2_PCT)
    tier2_lock = float(SHORT_T1_TAKE_PROFIT_TIER2_LOCK)
    hold_days = _MIN_SELL_LAG_TRADING_DAYS

    t2_high = t2_bar.get("high")
    t2_high_f: float | None = None
    t2_high_pct = 0.0
    if t2_high is not None and np.isfinite(float(t2_high)):
        t2_high_f = float(t2_high)
        t2_high_pct = (
            (t2_high_f - buy_price) / buy_price if buy_price > 0 else 0.0
        )

    if t2_high_f is not None:
        if t2_high_pct >= tier1_pct:
            sell_px = (t2_high_f + t2_close_f) / 2.0
            return _closed_exit(
                buy_price=buy_price,
                sell_px=sell_px,
                sell_date=t2_date,
                hold_days=hold_days,
                stop_px=strong_stop_px,
                exit_reason="t2_intraday_take_profit_tier1",
                t1_low_f=t1_low_f,
            )
        if t2_high_pct >= tier2_pct:
            sell_px = buy_price * (1.0 + tier2_lock)
            return _closed_exit(
                buy_price=buy_price,
                sell_px=sell_px,
                sell_date=t2_date,
                hold_days=hold_days,
                stop_px=strong_stop_px,
                exit_reason="t2_intraday_take_profit_tier2",
                t1_low_f=t1_low_f,
            )

    effective_stop_px = (
        mediocre_stop_px if t2_high_pct < tier2_pct else strong_stop_px
    )
    if t2_close_f < effective_stop_px:
        exit_reason = (
            "t2_asymmetric_stop_exit"
            if t2_high_pct < tier2_pct
            else "t2_close_below_stop_limit"
        )
        return _closed_exit(
            buy_price=buy_price,
            sell_px=t2_close_f,
            sell_date=t2_date,
            hold_days=hold_days,
            stop_px=effective_stop_px,
            exit_reason=exit_reason,
            stop_loss_triggered=1,
            t1_low_f=t1_low_f,
        )

    exit_reason = (
        "t2_trend_ride_exit"
        if offset >= 2 and t1_close_pct >= strong_thr and t2_close_f > t1_close_f
        else "t2_close_exit"
    )
    return _closed_exit(
        buy_price=buy_price,
        sell_px=t2_close_f,
        sell_date=t2_date,
        hold_days=hold_days,
        stop_px=strong_stop_px,
        exit_reason=exit_reason,
        t1_low_f=t1_low_f,
    )


def evaluate_short_trade(
    signal_close: float,
    *,
    t1_bar: dict[str, float | None],
    t2_bar: dict[str, float | None],
    t1_date: str | None,
    t2_date: str | None,
    sell_offset: int | None = None,
) -> dict[str, Any]:
    """信号日收盘价 + T+1/T+2 K 线 → 完整入场与平仓结果。"""
    buy_price, skip_reason = resolve_t1_entry_price(signal_close, t1_bar)
    if buy_price is None:
        return {
            "status": ORDER_STATUS_SKIPPED,
            "buy_price": None,
            "signal_close": float(signal_close),
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "stop_loss_price": None,
            "exit_reason": skip_reason or "entry_skipped",
        }

    exit_info = evaluate_daily_exit(
        buy_price,
        t1_bar=t1_bar,
        t2_bar=t2_bar,
        t1_date=t1_date,
        t2_date=t2_date,
        sell_offset=sell_offset,
    )
    return {
        **exit_info,
        "buy_price": buy_price,
        "signal_close": float(signal_close),
    }


def _attach_execution_to_selection_row(
    row: dict[str, Any],
    order: dict[str, Any],
    exit_info: dict[str, Any],
) -> None:
    """把执行结果写入待落库行的复盘字段与 detail_json。"""
    row["t1_open"] = order.get("t1_open")
    row["t1_close"] = order.get("t1_close")
    row["t2_open"] = order.get("t2_open")
    row["t2_close"] = order.get("t2_close")
    row["hold_plan"] = order.get("hold_plan") or SHORT_HOLD_PLAN
    detail = dict(row.get("detail") or {})
    detail["execution"] = {
        "buy_price": order.get("buy_price"),
        "sell_date": order.get("sell_date"),
        "sell_price": order.get("sell_price"),
        "pnl_ratio": order.get("pnl_ratio"),
        "status": order.get("status"),
        "stop_loss_triggered": order.get("stop_loss_triggered"),
        "exit_reason": order.get("exit_reason"),
        "stop_loss_price": exit_info.get("stop_loss_price"),
        "t1_low": exit_info.get("t1_low"),
    }
    row["detail"] = detail


def sync_short_orders_for_signal_day(
    conn: sqlite3.Connection,
    signal_trade_date: str,
    selection_rows: list[dict[str, Any]],
    *,
    sell_offset: int | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """为信号日 Top N 写入/更新 ``short_order_tracker``，并把执行结果挂回 selection 行。"""
    from .db import upsert_short_order

    td = str(signal_trade_date).strip()[:10]
    orders_out: list[dict[str, Any]] = []
    closed_n = 0
    holding_n = 0
    stop_n = 0

    for row in selection_rows:
        code = str(row.get("stock_code", "")).strip().zfill(6)
        signal_close = float(row.get("close_price") or 0.0)
        t1_date, t2_date = resolve_t1_t2_dates(td, conn)
        bars = fetch_post_signal_ohlc(conn, code, td)
        exit_info = evaluate_short_trade(
            signal_close,
            t1_bar=bars["t1"],
            t2_bar=bars["t2"],
            t1_date=t1_date,
            t2_date=t2_date,
            sell_offset=sell_offset,
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
            "rank": row.get("rank"),
            "rule_score": row.get("rule_score"),
            "exit_reason": exit_info.get("exit_reason"),
            "hold_plan": row.get("hold_plan") or SHORT_HOLD_PLAN,
            "t1_open": bars["t1"].get("open"),
            "t1_low": bars["t1"].get("low"),
            "t1_close": bars["t1"].get("close"),
            "t2_open": bars["t2"].get("open"),
            "t2_close": bars["t2"].get("close"),
        }
        upsert_short_order(conn, order)
        row["is_executed"] = (
            1 if order.get("status") == ORDER_STATUS_CLOSED else 0
        )
        _attach_execution_to_selection_row(row, order, exit_info)
        orders_out.append({**order, **exit_info})
        if order["status"] == ORDER_STATUS_CLOSED:
            closed_n += 1
        elif order["status"] == ORDER_STATUS_HOLDING:
            holding_n += 1
        if order.get("stop_loss_triggered"):
            stop_n += 1

    if commit:
        conn.commit()

    return {
        "buy_date": td,
        "orders": orders_out,
        "closed_count": closed_n,
        "holding_count": holding_n,
        "stop_loss_count": stop_n,
    }


def refresh_holding_short_orders(
    conn: sqlite3.Connection,
    buy_date: str | None = None,
    *,
    commit: bool = True,
) -> int:
    """K 线增量后，重新评估仍为 HOLDING 的订单（可指定信号日或全表）。"""
    from .db import ensure_short_term_tables

    ensure_short_term_tables(conn)
    td = str(buy_date).strip()[:10] if buy_date else None
    if td:
        cur = conn.execute(
            """
            SELECT stock_code, stock_name, buy_date, buy_price, signal_rank, rule_score
            FROM short_order_tracker
            WHERE buy_date = ? AND status = ?
            """,
            (td, ORDER_STATUS_HOLDING),
        )
    else:
        cur = conn.execute(
            """
            SELECT stock_code, stock_name, buy_date, buy_price, signal_rank, rule_score
            FROM short_order_tracker
            WHERE status = ?
            """,
            (ORDER_STATUS_HOLDING,),
        )
    pending = cur.fetchall()
    updated = 0
    for code, name, bd, bp, rank, score in pending:
        sel = {
            "stock_code": code,
            "stock_name": name,
            "close_price": bp,
            "rank": rank,
            "rule_score": score,
        }
        sync_short_orders_for_signal_day(
            conn, str(bd)[:10], [sel], commit=False
        )
        updated += 1
    if commit:
        conn.commit()
    return updated
