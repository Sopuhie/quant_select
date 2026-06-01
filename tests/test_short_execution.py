"""纯日线短线执行引擎单元测试。"""
from __future__ import annotations

import sqlite3

from src.short_term.execution import (
    ORDER_STATUS_CLOSED,
    ORDER_STATUS_SKIPPED,
    evaluate_daily_exit,
    evaluate_short_trade,
    resolve_t1_entry_price,
    stop_loss_trigger_price,
    sync_short_orders_for_signal_day,
)


def test_stop_loss_triggered_on_t1_close():
    buy = 10.0
    stop_px = stop_loss_trigger_price(buy)
    assert abs(stop_px - 9.4) < 1e-6
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.2, "low": 9.5, "close": 9.39},
        t2_bar={"open": 9.6, "low": 9.4, "close": 9.8},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["stop_loss_triggered"] == 1
    assert abs(out["sell_price"] - 9.39) < 1e-6
    assert out["exit_reason"] == "t1_close_below_stop_limit"


def test_entry_dip_cost_correction_on_high_open():
    px, reason = resolve_t1_entry_price(
        10.0,
        {"open": 10.3, "high": 10.5, "low": 9.9, "close": 10.1},
    )
    assert reason is None
    assert abs(px - 10.1) < 1e-6


def test_entry_uses_open_on_micro_gap_up():
    px, reason = resolve_t1_entry_price(
        10.0,
        {"open": 10.1, "high": 10.2, "low": 10.0, "close": 10.15},
    )
    assert reason is None
    assert px == 10.1


def test_entry_chase_rejected_above_55pct():
    px, reason = resolve_t1_entry_price(
        10.0,
        {"open": 10.56, "high": 10.6, "low": 10.4, "close": 10.5},
    )
    assert px is None
    assert reason == "t1_open_chase_rejected"


def test_evaluate_short_trade_allows_moderate_gap_up():
    out = evaluate_short_trade(
        10.0,
        t1_bar={"open": 10.2, "high": 10.3, "low": 10.0, "close": 10.1},
        t2_bar={"open": 10.0, "low": 9.9, "close": 10.0},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
    )
    assert out["status"] != ORDER_STATUS_SKIPPED
    assert out["buy_price"] == 10.2


def test_take_profit_on_t1_high_spike():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.7, "low": 9.9, "close": 10.1},
        t2_bar={"open": 10.0, "low": 9.9, "close": 10.0},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["exit_reason"] == "t1_intraday_take_profit"


def test_mediocre_exit_at_t1_close():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.2, "low": 9.5, "close": 9.8},
        t2_bar={"open": 9.9, "low": 9.7, "close": 10.2},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["exit_reason"] == "t1_mediocre_close_exit"


def test_sync_orders_writes_tracker():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE stock_daily_kline (
            date TEXT, stock_code TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        INSERT INTO stock_daily_kline VALUES
        ('2026-05-15', '000001', 9.8, 9.9, 9.7, 10.0, 1000),
        ('2026-05-16', '000001', 10.0, 10.1, 9.3, 9.39, 1000);
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
