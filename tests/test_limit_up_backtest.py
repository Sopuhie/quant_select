"""打板滚动回测单元测试。"""
from __future__ import annotations

import sqlite3

import pytest

from src.limit_up_hit.backtest import (
    evaluate_open_open_trade,
    evaluate_optimized_backtest_trade,
    simulate_limit_up_backtest_trade,
)
from src.limit_up_hit.execution import ORDER_STATUS_CLOSED, ORDER_STATUS_HOLDING, ORDER_STATUS_SKIPPED


def test_legacy_open_open_trade_t2_exit():
    out = evaluate_open_open_trade(
        10.0,
        "600000",
        t1_bar={"open": 10.5, "high": 11.0, "low": 10.4, "close": 11.0},
        t1_date="2026-05-16",
        post_t1_bars=[
            {
                "date": "2026-05-19",
                "open": 10.8,
                "high": 11.0,
                "low": 10.5,
                "close": 10.9,
            }
        ],
        t2_date="2026-05-19",
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["sell_price"] == pytest.approx(10.8)
    assert out["exit_reason"] == "t2_open_exit"


def test_legacy_ride_one_word_then_open_exit():
    out = evaluate_open_open_trade(
        10.0,
        "600000",
        t1_bar={"open": 10.5, "high": 11.0, "low": 10.4, "close": 11.0},
        t1_date="2026-05-16",
        post_t1_bars=[
            {
                "date": "2026-05-19",
                "open": 12.1,
                "high": 12.1,
                "low": 12.1,
                "close": 12.1,
            },
            {
                "date": "2026-05-20",
                "open": 11.5,
                "high": 12.0,
                "low": 11.2,
                "close": 11.8,
            },
        ],
        t2_date="2026-05-19",
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["exit_reason"] == "ride_open_exit"
    assert out["ride_days"] == 1


def test_optimized_t1_requires_close_limit_and_slippage():
    out = evaluate_optimized_backtest_trade(
        10.0,
        "600000",
        t1_bar={"open": 10.5, "high": 11.0, "low": 10.99, "close": 11.0},
        t1_date="2026-05-16",
        post_t1_bars=[
            {
                "date": "2026-05-19",
                "open": 10.8,
                "high": 11.0,
                "low": 10.5,
                "close": 10.6,
                "turnover": 5.0,
            }
        ],
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["buy_price"] == pytest.approx(10.5 * 1.005)
    assert out["exit_reason"] == "t2_close_exit"


def test_optimized_strong_board_ride_on_t2():
    out = evaluate_optimized_backtest_trade(
        10.0,
        "600000",
        t1_bar={"open": 10.5, "high": 11.0, "low": 10.99, "close": 11.0},
        t1_date="2026-05-16",
        post_t1_bars=[
            {
                "date": "2026-05-19",
                "open": 12.1,
                "high": 12.1,
                "low": 12.0,
                "close": 12.1,
                "turnover": 2.0,
            },
            {
                "date": "2026-05-20",
                "open": 11.5,
                "high": 12.0,
                "low": 11.2,
                "close": 11.8,
                "turnover": 8.0,
            },
        ],
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["sell_date"] == "2026-05-20"
    assert out["exit_reason"] == "ride_open_exit"
    assert out["ride_days"] >= 1


def test_optimized_open_buy_without_t1_close_limit():
    out = evaluate_optimized_backtest_trade(
        10.0,
        "600000",
        t1_bar={"open": 10.5, "high": 10.8, "low": 10.4, "close": 10.6},
        t1_date="2026-05-16",
        post_t1_bars=[
            {
                "date": "2026-05-19",
                "open": 10.8,
                "high": 11.0,
                "low": 10.5,
                "close": 10.6,
                "turnover": 5.0,
            }
        ],
    )
    assert out["status"] == ORDER_STATUS_CLOSED
    assert out["buy_price"] == pytest.approx(10.5 * 1.005)
    assert out["exit_reason"] == "t2_close_exit"


def test_optimized_skip_when_t1_not_limit_close_only_if_required(monkeypatch):
    import src.limit_up_hit.backtest as bt

    monkeypatch.setattr(bt, "LUH_BT_REQUIRE_T1_CLOSE_LIMIT", True)
    out = evaluate_optimized_backtest_trade(
        10.0,
        "600000",
        t1_bar={"open": 10.5, "high": 10.9, "low": 10.4, "close": 10.8},
        t1_date="2026-05-16",
        post_t1_bars=[],
    )
    assert out["status"] == ORDER_STATUS_SKIPPED
    assert out["exit_reason"] == "t1_not_limit_up_close"


def test_simulate_legacy_mode():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE stock_daily_kline (
            date TEXT, stock_code TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        INSERT INTO stock_daily_kline VALUES
        ('2026-05-15', '600000', 10.0, 11.0, 9.9, 11.0, 1000),
        ('2026-05-16', '600000', 10.5, 11.0, 10.4, 11.0, 1000),
        ('2026-05-19', '600000', 10.8, 11.0, 10.5, 10.9, 1000);
        """
    )
    tr = simulate_limit_up_backtest_trade(
        conn,
        "2026-05-15",
        {"stock_code": "600000", "close_price": 11.0, "rank": 1},
        legacy=True,
    )
    assert tr["status"] == ORDER_STATUS_CLOSED
