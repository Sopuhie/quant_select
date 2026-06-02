"""短线可操作价位指南单元测试。"""
from __future__ import annotations

from src.short_term.trade_guide import build_trade_action_guide


def test_trade_guide_t1_open_range():
    g = build_trade_action_guide(10.0)
    assert g["valid"] is True
    assert g["t1"]["open_valid_lo"] == 9.8
    assert g["t1"]["open_valid_hi"] == 10.55
    assert g["t1"]["dip_open_hi"] == 10.15


def test_trade_guide_t2_levels_at_signal_close():
    g = build_trade_action_guide(10.0)
    ref = g["t2_ref"]
    assert ref["tp_tier1_high"] == 10.6
    assert ref["tp_tier2_sell"] == 10.3
    assert ref["stop_mediocre_close"] == 9.6
    assert ref["stop_strong_close"] == 9.4
    assert ref["t1_strong_close"] == 10.4


def test_trade_guide_display_columns():
    g = build_trade_action_guide(10.0)
    disp = g["display"]
    assert "9.80~10.55" in disp["T+1开盘区间"]
    assert "10.60" in disp["T+2止盈触发"]
    assert "9.60" in disp["T+2止损线"]


def test_trade_guide_summary_text():
    g = build_trade_action_guide(10.0)
    text = g["summary_text"]
    assert "T+1 买入" in text
    assert "T+2 卖出" in text
    assert "9.80" in text
