"""纯日线短线执行引擎单元测试。"""
from __future__ import annotations

import sqlite3

from src.short_term.execution import (
    ORDER_STATUS_CLOSED,
    ORDER_STATUS_HOLDING,
    evaluate_daily_exit,
    stop_loss_trigger_price,
    sync_short_orders_for_signal_day,
)


def test_stop_loss_triggered_on_t1_low():
    buy = 10.0
    stop_px = stop_loss_trigger_price(buy)
    assert abs(stop_px - 9.7) < 1e-6
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "low": 9.5, "close": 9.6},
        t2_bar={"open": 9.6, "low": 9.4, "close": 9.8},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=1,
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["stop_loss_triggered"] == 1
    assert abs(out["sell_price"] - stop_px) < 1e-6
    assert out["exit_reason"] == "t1_intraday_stop_loss"


def test_limit_down_sealed_bar_uses_close():
    """一字跌停：open==low==close，开盘即封死，按收盘价计提。"""
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 9.0, "low": 9.0, "close": 9.0},
        t2_bar={},
        t1_date="2026-05-16",
        t2_date=None,
        sell_offset=1,
    )
    assert out["exit_reason"] == "t1_open_below_stop_limit"
    assert out["sell_price"] == 9.0
    assert out["pnl_ratio"] == -0.1


def test_stop_loss_gap_down_uses_t1_close_not_stop_px():
    """T+1 开盘已在止损线下方：不能按 -3% 价成交，按收盘价计提。"""
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 9.5, "low": 9.5, "close": 9.2},
        t2_bar={},
        t1_date="2026-05-16",
        t2_date=None,
        sell_offset=1,
    )
    assert out["exit_reason"] == "t1_open_below_stop_limit"
    assert out["sell_price"] == 9.2
    assert out["stop_loss_triggered"] == 1


def test_t1_close_exit_when_no_stop():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.1, "low": 9.8, "close": 10.5},
        t2_bar={"open": 10.4, "low": 10.0, "close": 10.6},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=1,
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["stop_loss_triggered"] == 0
    assert out["sell_price"] == 10.5
    assert out["sell_date"] == "2026-05-16"


def test_t2_close_exit_when_offset_2():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "low": 9.8, "close": 10.2},
        t2_bar={"open": 10.1, "low": 10.0, "close": 10.8},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["sell_price"] == 10.8
    assert out["sell_date"] == "2026-05-19"
    assert out["hold_days"] == 2


def test_holding_when_t1_missing():
    out = evaluate_daily_exit(
        10.0,
        t1_bar={"open": None, "low": None, "close": None},
        t2_bar={},
        t1_date=None,
        t2_date=None,
    )
    assert out["status"] == ORDER_STATUS_HOLDING


def test_sync_orders_writes_tracker():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE stock_daily_kline (
            date TEXT, stock_code TEXT, open REAL, low REAL, close REAL, volume REAL
        );
        INSERT INTO stock_daily_kline VALUES
        ('2026-05-15', '000001', 9.8, 9.7, 10.0, 1000),
        ('2026-05-16', '000001', 10.0, 9.5, 9.6, 1000);
        """
    )
    from unittest.mock import patch

    with patch(
        "src.short_term.execution.resolve_t1_t2_dates",
        return_value=("2026-05-16", "2026-05-19"),
    ):
        rows = [
            {
                "stock_code": "000001",
                "stock_name": "测试",
                "close_price": 10.0,
                "rank": 1,
                "rule_score": 80.0,
                "detail": {},
            }
        ]
        summary = sync_short_orders_for_signal_day(conn, "2026-05-15", rows)
        assert summary["stop_loss_count"] == 1
        row = conn.execute(
            "SELECT status, stop_loss_triggered, sell_price FROM short_order_tracker"
        ).fetchone()
        assert row[0] == ORDER_STATUS_CLOSED
        assert row[1] == 1
        assert abs(row[2] - 9.7) < 1e-6
