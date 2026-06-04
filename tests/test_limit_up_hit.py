"""打板模块执行引擎单元测试。"""
from __future__ import annotations

from src.limit_up_hit.execution import (
    ORDER_STATUS_CLOSED,
    ORDER_STATUS_SKIPPED,
    evaluate_board_exit,
    evaluate_board_trade,
    is_one_word_limit_up,
    resolve_t1_board_entry,
    theoretical_limit_up_price,
)


def test_theoretical_limit_up_price_main_board():
    assert theoretical_limit_up_price(10.0, "600000") == 11.0


def test_one_word_limit_up_blocks_entry():
    bar = {"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0}
    assert is_one_word_limit_up(bar, 10.0, "600000") is True
    px, reason = resolve_t1_board_entry(10.0, bar, "600000")
    assert px is None
    assert reason == "t1_one_word_limit_up"


def test_board_entry_on_t1_limit_close():
    bar = {"open": 10.5, "high": 11.0, "low": 10.4, "close": 11.0}
    px, reason = resolve_t1_board_entry(10.0, bar, "600000")
    assert reason is None
    assert px == 11.0


def test_board_entry_skip_when_t1_not_limit():
    bar = {"open": 10.5, "high": 10.8, "low": 10.2, "close": 10.6}
    px, reason = resolve_t1_board_entry(10.0, bar, "600000")
    assert px is None
    assert reason == "t1_not_limit_up_close"


def test_t2_take_profit_tier1():
    buy = 11.0
    out = evaluate_board_exit(
        buy,
        "600000",
        t1_bar={"open": 11.0, "high": 11.0, "low": 10.8, "close": 11.0},
        t2_bar={"open": 11.5, "high": 12.2, "low": 11.0, "close": 12.0},
        t1_date="2026-05-20",
        t2_date="2026-05-21",
    )
    assert out["exit_reason"] == "t2_take_profit_tier1"
    assert out["status"] == ORDER_STATUS_CLOSED


def test_full_trade_skipped_on_one_word():
    out = evaluate_board_trade(
        10.0,
        "600000",
        t1_bar={"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0},
        t2_bar={"open": 11.0, "high": 11.0, "low": 10.5, "close": 10.8},
        t1_date="2026-05-20",
        t2_date="2026-05-21",
    )
    assert out["status"] == ORDER_STATUS_SKIPPED


def test_board_streak_count():
    import pandas as pd

    from src.limit_up_hit.strategy import count_board_streak

    closes = pd.Series([9.0, 9.9, 10.89, 11.98])
    assert count_board_streak(closes, "600000") == 3
