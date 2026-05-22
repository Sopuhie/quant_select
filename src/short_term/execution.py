# -*- coding: utf-8 -*-
"""
纯日线短线交易执行引擎。

- 买入：信号日 T 收盘价（close）即买入价
- 止损：T+1 日 low < 买入价×(1-止损比例) → 平仓价强制为买入价×(1-止损比例)
- 未止损：在 T+SHORT_SELL_OFFSET 以当日 close 平仓
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from .config import SHORT_HOLD_PLAN, SHORT_SELL_OFFSET, SHORT_STOP_LOSS_RATIO
from .review_prices import resolve_t1_t2_dates

ORDER_STATUS_HOLDING = "HOLDING"
ORDER_STATUS_CLOSED = "CLOSED"


def stop_loss_trigger_price(buy_price: float) -> float:
    """硬止损触发价（买入价 × (1 - 止损比例)）。"""
    return float(buy_price) * (1.0 - float(SHORT_STOP_LOSS_RATIO))


def fetch_post_signal_ohlc(
    conn: sqlite3.Connection,
    stock_code: str,
    signal_trade_date: str,
) -> dict[str, dict[str, float | None]]:
    """
    读取信号日之后的 T+1 / T+2 日线 OHLC。

    Returns:
        ``{"t1": {"open","low","close"}, "t2": {...}}``，无 K 线则对应键为空 dict。
    """
    code = str(stock_code).strip().zfill(6)
    t1, t2 = resolve_t1_t2_dates(signal_trade_date, conn)
    out: dict[str, dict[str, float | None]] = {
        "t1": {"open": None, "low": None, "close": None},
        "t2": {"open": None, "low": None, "close": None},
    }
    if not t1:
        return out

    dates = [t1]
    if t2:
        dates.append(t2)
    placeholders = ",".join("?" * len(dates))
    cur = conn.execute(
        f"""
        SELECT date, open, low, close
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
            lo = float(row[2])
            c = float(row[3])
        except (TypeError, ValueError):
            continue
        if all(np.isfinite(x) for x in (o, lo, c)):
            by_date[d] = {"open": o, "low": lo, "close": c}

    if t1 in by_date:
        out["t1"] = dict(by_date[t1])
    if t2 and t2 in by_date:
        out["t2"] = dict(by_date[t2])
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
    纯日线模拟平仓结果。

    止损评估顺序（仅依赖日线 OHLC）：
    1. T+1 ``low`` 跌破止损线且 ``open <= 止损价``：开盘已在 -3% 下方或一字跌停，
       无法在止损价成交，按 T+1 **收盘价**（封死跌停）或开盘价计提，
       ``exit_reason=t1_open_below_stop_limit``。
    2. T+1 ``low`` 跌破止损线且 ``open > 止损价``：盘中触及止损，按止损价平仓，
       ``exit_reason=t1_intraday_stop_loss``。
    3. 未触发止损：按 ``SHORT_SELL_OFFSET`` 在 T+1 或 T+2 以 ``close`` 平仓。

    T+1 K 线尚未入库时返回 HOLDING。
    """
    offset = int(sell_offset if sell_offset is not None else SHORT_SELL_OFFSET)
    offset = max(1, min(2, offset))
    buy_price = float(buy_price)
    stop_px = stop_loss_trigger_price(buy_price)

    t1_open = t1_bar.get("open")
    t1_low = t1_bar.get("low")
    t1_close = t1_bar.get("close")
    t2_close = t2_bar.get("close")

    # T+1 数据未就绪 → 持仓中
    if t1_low is None or not np.isfinite(float(t1_low)):
        return {
            "status": ORDER_STATUS_HOLDING,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "stop_loss_price": stop_px,
            "exit_reason": "await_t1_kline",
        }

    t1_low_f = float(t1_low)
    t1_open_f: float | None = None
    if t1_open is not None and np.isfinite(float(t1_open)):
        t1_open_f = float(t1_open)

    # T+1 触及止损区（low 跌破止损线）
    if t1_low_f < stop_px and t1_date:
        # 开盘已在止损线下方或一字跌停：实盘无法在 -3% 价位成交，按 T+1 收盘/开盘计提
        if t1_open_f is not None and t1_open_f <= stop_px:
            # 收盘价缺失时降级为开盘价，避免 float(None) 引发 TypeError
            t1_close_f = float(t1_close) if t1_close is not None else t1_open_f
            if t1_close_f is not None and np.isfinite(float(t1_close_f)):
                sell_px = float(t1_close_f)
            elif t1_open_f is not None:
                sell_px = t1_open_f
            else:
                sell_px = stop_px
            pnl = (sell_px - buy_price) / buy_price if buy_price > 0 else None
            return {
                "status": ORDER_STATUS_CLOSED,
                "sell_date": t1_date,
                "sell_price": sell_px,
                "hold_days": 1,
                "pnl_ratio": pnl,
                "stop_loss_triggered": 1,
                "stop_loss_price": stop_px,
                "exit_reason": "t1_open_below_stop_limit",
                "t1_open": t1_open_f,
                "t1_low": t1_low_f,
            }

        # 开盘正常、盘中跌破止损线：可按止损价 -3% 平仓（纯日线等价成交）
        pnl = (stop_px - buy_price) / buy_price if buy_price > 0 else None
        return {
            "status": ORDER_STATUS_CLOSED,
            "sell_date": t1_date,
            "sell_price": stop_px,
            "hold_days": 1,
            "pnl_ratio": pnl,
            "stop_loss_triggered": 1,
            "stop_loss_price": stop_px,
            "exit_reason": "t1_intraday_stop_loss",
            "t1_open": t1_open_f,
            "t1_low": t1_low_f,
        }

    # 未触发止损：T+1 收盘平仓
    if offset == 1:
        if t1_close is None or not np.isfinite(float(t1_close)) or not t1_date:
            return {
                "status": ORDER_STATUS_HOLDING,
                "sell_date": None,
                "sell_price": None,
                "hold_days": 0,
                "pnl_ratio": None,
                "stop_loss_triggered": 0,
                "stop_loss_price": stop_px,
                "exit_reason": "await_t1_close",
            }
        sell_px = float(t1_close)
        pnl = (sell_px - buy_price) / buy_price if buy_price > 0 else None
        return {
            "status": ORDER_STATUS_CLOSED,
            "sell_date": t1_date,
            "sell_price": sell_px,
            "hold_days": 1,
            "pnl_ratio": pnl,
            "stop_loss_triggered": 0,
            "stop_loss_price": stop_px,
            "exit_reason": "t1_close_exit",
            "t1_low": t1_low_f,
        }

    # offset == 2：持有至 T+2 收盘（T+1 已确认未触发止损）
    if t2_close is None or not np.isfinite(float(t2_close)) or not t2_date:
        return {
            "status": ORDER_STATUS_HOLDING,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 1,
            "pnl_ratio": None,
            "stop_loss_triggered": 0,
            "stop_loss_price": stop_px,
            "exit_reason": "await_t2_close",
            "t1_low": t1_low_f,
        }
    sell_px = float(t2_close)
    pnl = (sell_px - buy_price) / buy_price if buy_price > 0 else None
    return {
        "status": ORDER_STATUS_CLOSED,
        "sell_date": t2_date,
        "sell_price": sell_px,
        "hold_days": 2,
        "pnl_ratio": pnl,
        "stop_loss_triggered": 0,
        "stop_loss_price": stop_px,
        "exit_reason": "t2_close_exit",
        "t1_low": t1_low_f,
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
    """
    为信号日 Top N 写入/更新 ``short_order_tracker``，并把执行结果挂回 selection 行。
    """
    from .db import upsert_short_order

    td = str(signal_trade_date).strip()[:10]
    orders_out: list[dict[str, Any]] = []
    closed_n = 0
    holding_n = 0
    stop_n = 0

    for row in selection_rows:
        code = str(row.get("stock_code", "")).strip().zfill(6)
        buy_price = float(row.get("close_price") or 0.0)
        t1_date, t2_date = resolve_t1_t2_dates(td)
        bars = fetch_post_signal_ohlc(conn, code, td)
        exit_info = evaluate_daily_exit(
            buy_price,
            t1_bar=bars["t1"],
            t2_bar=bars["t2"],
            t1_date=t1_date,
            t2_date=t2_date,
            sell_offset=sell_offset,
        )
        order = {
            "stock_code": code,
            "stock_name": row.get("stock_name"),
            "buy_date": td,
            "buy_price": buy_price,
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
        row["is_executed"] = 1 if order.get("status") == ORDER_STATUS_CLOSED else 0
        _attach_execution_to_selection_row(row, order, exit_info)
        orders_out.append({**order, **exit_info})
        if order["status"] == ORDER_STATUS_CLOSED:
            closed_n += 1
        else:
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
    from .db import ensure_short_term_tables, upsert_short_order

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
