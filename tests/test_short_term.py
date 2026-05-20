"""短线规则模块单元测试（不依赖完整行情库）。"""
from __future__ import annotations

import pandas as pd

from src.short_term.dingtalk import build_short_selection_markdown
from src.short_term.strategy import ShortTermRuleStrategy, _is_st_name


def test_is_st_name():
    assert _is_st_name("*ST 测试")
    assert _is_st_name("ST某某")
    assert not _is_st_name("贵州茅台")


def test_evaluate_rules_passes_synthetic():
    strat = ShortTermRuleStrategy.__new__(ShortTermRuleStrategy)
    curr = pd.Series(
        {
            "close": 10.5,
            "open": 10.0,
            "ma_fast": 10.2,
            "ma_slow": 10.0,
            "change_pct": 0.03,
            "return_5d": 0.08,
            "vol_ratio_5d": 1.5,
            "vol_ratio_1d": 1.2,
            "macd_diff": 0.1,
            "macd_dea": 0.05,
            "macd_bar": 0.08,
            "kdj_k": 55,
            "kdj_d": 50,
            "kdj_j": 60,
        }
    )
    prev = pd.Series(
        {
            "kdj_j": 52,
            "kdj_k": 48,
            "kdj_d": 50,
            "macd_bar": 0.05,
        }
    )
    ok, score, detail = strat.evaluate_rules("000001", "测试", curr, prev)
    assert ok is True
    assert score > 0
    assert detail["checks"]["trend_ma"] is True


def test_build_short_markdown_contains_codes():
    title, text = build_short_selection_markdown(
        "2026-05-19",
        [
            {
                "rank": 1,
                "stock_code": "000001",
                "stock_name": "平安银行",
                "close_price": 10.0,
                "day_change_pct": 0.02,
                "vol_ratio_5d": 1.3,
                "rule_score": 80.0,
                "advice_text": "测试建议",
            }
        ],
        market_score=60,
    )
    assert "短线" in title
    assert "000001" in text
    assert "平安银行" in text
