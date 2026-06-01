"""短线规则滚动回测单元测试。"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from src.short_term.backtest import (
    resolve_scan_end_date,
    run_short_term_rolling_backtest,
    simulate_short_trade,
    summarize_backtest,
)
from src.short_term.execution import ORDER_STATUS_CLOSED


def test_resolve_scan_end_date_offset_1():
    dates = ["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]
    assert resolve_scan_end_date(dates, sell_offset=1) == "2026-05-14"


def test_resolve_scan_end_date_offset_2():
    dates = ["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]
    assert resolve_scan_end_date(dates, sell_offset=2) == "2026-05-13"


def test_summarize_backtest_metrics():
    trades = pd.DataFrame(
        [
            {"status": ORDER_STATUS_CLOSED, "pnl_ratio": 0.05, "exit_reason": "t1_close_exit"},
            {"status": ORDER_STATUS_CLOSED, "pnl_ratio": -0.03, "exit_reason": "t1_close_below_stop_limit"},
        ]
    )
    daily = pd.DataFrame(
        [
            {"signal_date": "2026-05-12", "day_type": "signal", "day_return": 0.01},
            {"signal_date": "2026-05-13", "day_type": "fused", "day_return": None},
        ]
    )
    summary = summarize_backtest(trades, daily, meta={"start_date": "2026-05-12"})
    assert summary["closed_trades"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["signal_days"] == 1
    assert summary["fused_days"] == 1
    assert summary["cum_return_pct"] == pytest.approx(1.0)


def test_simulate_short_trade_t1_exit():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE stock_daily_kline (
            date TEXT, stock_code TEXT, open REAL, low REAL, close REAL, volume REAL
        );
        INSERT INTO stock_daily_kline VALUES
        ('2026-05-15', '000001', 9.8, 9.7, 10.0, 1000),
        ('2026-05-16', '000001', 10.1, 9.8, 10.5, 1000);
        """
    )
    with patch(
        "src.short_term.backtest.resolve_t1_t2_dates",
        return_value=("2026-05-16", "2026-05-19"),
    ):
        tr = simulate_short_trade(
            conn,
            "2026-05-15",
            {"stock_code": "000001", "stock_name": "测试", "close_price": 10.0, "rank": 1},
            sell_offset=1,
        )
    assert tr["status"] == ORDER_STATUS_CLOSED
    assert tr["pnl_ratio"] == 0.05
    assert tr["exit_reason"] == "t1_close_exit"


def test_run_short_term_rolling_backtest_with_mock_scan():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE stock_daily_kline (
            date TEXT, stock_code TEXT, open REAL, low REAL, close REAL, volume REAL
        );
        INSERT INTO stock_daily_kline VALUES
        ('2026-05-14', '000001', 9.8, 9.7, 10.0, 1000),
        ('2026-05-15', '000001', 10.0, 9.5, 9.4, 1000),
        ('2026-05-16', '000001', 9.6, 9.4, 9.8, 1000);
        """
    )

    class _FakeEngine:
        def scan(self, *args, **kwargs):
            return (pd.DataFrame(), "2026-05-14", 60)

        def get_last_persist_rows(self):
            return [
                {
                    "stock_code": "000001",
                    "stock_name": "测试",
                    "close_price": 10.0,
                    "rank": 1,
                    "rule_score": 80.0,
                }
            ]

    with patch(
        "src.short_term.backtest.ShortTermRuleStrategy",
        return_value=_FakeEngine(),
    ), patch(
        "src.short_term.backtest.resolve_t1_t2_dates",
        return_value=("2026-05-15", "2026-05-16"),
    ):
        out = run_short_term_rolling_backtest(
            conn,
            "2026-05-14",
            "2026-05-15",
            top_n=1,
            verbose=False,
        )

    assert out["summary"]["closed_trades"] == 1
    assert out["summary"]["signal_days"] == 1
    assert out["trades"].iloc[0]["stop_loss_triggered"] == 1
