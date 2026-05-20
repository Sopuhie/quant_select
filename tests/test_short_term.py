"""短线规则模块单元测试（不依赖完整行情库）。"""
from __future__ import annotations

from src.short_term.dingtalk import build_short_selection_markdown
from src.short_term.strategy import ShortTermRuleStrategy, _is_st_name


def test_is_st_name():
    assert _is_st_name("*ST 测试")
    assert _is_st_name("ST某某")
    assert not _is_st_name("贵州茅台")


def test_rule_score_positive():
    score = ShortTermRuleStrategy._rule_score(0.03, 1.5, 0.08, 0.05, 8.0)
    assert score > 0


def test_advice_text_high_j():
    assert "J 偏高" in ShortTermRuleStrategy.advice_text(90.0, 0.02)


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
