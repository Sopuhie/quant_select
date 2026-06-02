"""纯日线短线执行引擎单元测试。"""
from __future__ import annotations

import sqlite3

from src.short_term.execution import (
    ORDER_STATUS_CLOSED,
    ORDER_STATUS_SKIPPED,
    evaluate_daily_exit,
    evaluate_short_trade,
    mediocre_stop_trigger_price,
    resolve_t1_entry_price,
    sync_short_orders_for_signal_day,
)


def test_stop_loss_asymmetric_on_t2_close():
    buy = 10.0
    stop_px = mediocre_stop_trigger_price(buy)
    assert abs(stop_px - 9.6) < 1e-6
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.5},
        t2_bar={"open": 10.4, "high": 10.2, "low": 9.5, "close": 9.55},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["stop_loss_triggered"] == 1
    assert out["exit_reason"] == "t2_asymmetric_stop_exit"
    assert out["sell_date"] == "2026-05-19"


def test_strong_intraday_hits_tier2_before_stop_on_t2():
    """T+2 曾冲高≥5% 时，双阶梯优先于收盘止损（锁定 +3%）。"""
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.5},
        t2_bar={"open": 10.4, "high": 10.55, "low": 9.3, "close": 9.35},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["exit_reason"] == "t2_intraday_take_profit_tier2"
    assert abs(out["sell_price"] - 10.3) < 1e-6


def test_take_profit_tier1_on_t2_high():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.5},
        t2_bar={"open": 10.4, "high": 10.7, "low": 9.9, "close": 10.1},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["exit_reason"] == "t2_intraday_take_profit_tier1"
    assert abs(out["sell_price"] - 10.4) < 1e-6


def test_take_profit_tier2_on_t2_high():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.5},
        t2_bar={"open": 10.4, "high": 10.52, "low": 9.9, "close": 10.0},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["exit_reason"] == "t2_intraday_take_profit_tier2"
    assert abs(out["sell_price"] - 10.3) < 1e-6


def test_entry_dip_cost_correction_on_high_open():
    px, reason = resolve_t1_entry_price(
        10.0,
        {"open": 10.3, "low": 10.0, "high": 10.4, "close": 10.2},
    )
    assert reason is None
    assert px is not None
    assert abs(px - 10.15) < 1e-6


def test_take_profit_priority_over_stop_on_t2():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.5},
        t2_bar={"open": 10.4, "high": 10.7, "low": 9.4, "close": 9.4},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["exit_reason"] == "t2_intraday_take_profit_tier1"
    assert abs(out["sell_price"] - 10.05) < 1e-6


def test_weak_t1_still_uses_t2_stop_chain():
    """T+1 弱势不再 T+2 开盘平，仍走止盈/止损链。"""
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.2, "low": 9.5, "close": 9.8},
        t2_bar={"open": 9.9, "low": 9.5, "close": 9.55, "high": 10.1},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["stop_loss_triggered"] == 1
    assert out["sell_price"] == 9.55
    assert out["exit_reason"] == "t2_asymmetric_stop_exit"
    assert out["hold_days"] == 2


def test_t2_trend_ride_exit_when_strong():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.5},
        t2_bar={"open": 10.4, "low": 10.0, "high": 10.45, "close": 10.8},
        t1_date="2026-05-16",
        t2_date="2026-05-19",
        sell_offset=2,
    )
    assert out["exit_reason"] == "t2_trend_ride_exit"
    assert out["sell_price"] == 10.8


def test_no_sell_on_t1_even_if_stop_would_trigger():
    buy = 10.0
    out = evaluate_daily_exit(
        buy,
        t1_bar={"open": 10.0, "high": 10.1, "low": 9.3, "close": 9.35},
        t2_bar={},
        t1_date="2026-05-16",
        t2_date=None,
        sell_offset=2,
    )
    assert out["status"] != ORDER_STATUS_CLOSED
    assert out["exit_reason"] == "await_t2_kline"


def test_sync_orders_writes_tracker():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE stock_daily_kline (
            date TEXT, stock_code TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        INSERT INTO stock_daily_kline VALUES
        ('2026-05-15', '000001', 9.8, 9.9, 9.7, 10.0, 1000),
        ('2026-05-16', '000001', 10.0, 10.5, 9.3, 10.5, 1000),
        ('2026-05-19', '000001', 10.0, 10.1, 9.2, 9.35, 1000);
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


def test_evaluate_short_trade_skipped_on_chase():
    out = evaluate_short_trade(
        10.0,
        t1_bar={"open": 10.8, "high": 10.9, "low": 10.7, "close": 10.8},
        t2_bar={},
        t1_date="2026-05-16",
        t2_date=None,
    )
    assert out["status"] == ORDER_STATUS_SKIPPED
