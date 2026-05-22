"""短线规则模块单元测试（不依赖完整行情库）。"""
from __future__ import annotations

import sqlite3

from src.short_term.dingtalk import build_short_selection_markdown
from src.short_term.review_prices import enrich_rows_with_review_prices, resolve_t1_t2_dates
from src.short_term.strategy import ShortTermRuleStrategy, _is_st_name


def test_is_st_name():
    assert _is_st_name("*ST 测试")
    assert _is_st_name("ST某某")
    assert not _is_st_name("贵州茅台")


def test_get_sector_near_limit_main_board():
    from src.short_term.strategy import get_sector_near_limit_threshold

    assert get_sector_near_limit_threshold("000001") == 0.095


def test_get_sector_near_limit_gem():
    from src.short_term.strategy import get_sector_near_limit_threshold

    assert get_sector_near_limit_threshold("300001") == 0.192


def test_rule_score_positive():
    score = ShortTermRuleStrategy._rule_score(0.03, 1.5, 0.08, 0.05, 8.0, 50.0)
    assert score > 0


def test_rule_score_j_overbought_penalty():
    low_j = ShortTermRuleStrategy._rule_score(0.04, 1.5, 0.08, 0.05, 8.0, 50.0)
    high_j = ShortTermRuleStrategy._rule_score(0.04, 1.5, 0.08, 0.05, 8.0, 90.0)
    assert high_j < low_j


def test_advice_text_high_j():
    assert "J 偏高" in ShortTermRuleStrategy.advice_text(90.0, 0.02)


def test_vol_ratio_clip_max_10():
    import pandas as pd

    from src.short_term.config import SHORT_VOL_RATIO_CLIP_MAX

    vol = pd.Series([100.0, 1.0, 50.0])
    vr1 = (vol / vol.shift(1).replace(0, float("nan"))).fillna(1.0).clip(
        upper=SHORT_VOL_RATIO_CLIP_MAX
    )
    assert float(vr1.iloc[-1]) == SHORT_VOL_RATIO_CLIP_MAX


def test_hold_plan_uses_t_close_buy():
    from src.short_term.config import SHORT_HOLD_PLAN

    assert "收盘" in SHORT_HOLD_PLAN
    assert "开盘买入" not in SHORT_HOLD_PLAN


def test_resolve_t1_t2_dates_chain():
    t1, t2 = resolve_t1_t2_dates("2026-05-15")
    assert t1 is not None
    assert t2 is not None
    assert t1 > "2026-05-15"
    assert t2 > t1


def test_enrich_review_prices_from_kline():
    from unittest.mock import patch

    with patch(
        "src.short_term.review_prices.resolve_t1_t2_dates",
        return_value=("2026-05-16", "2026-05-19"),
    ):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE stock_daily_kline (
                date TEXT, stock_code TEXT, open REAL, close REAL, volume REAL
            );
            INSERT INTO stock_daily_kline VALUES
            ('2026-05-16', '000001', 10.0, 10.5, 1000),
            ('2026-05-19', '000001', 11.0, 11.2, 1000);
            """
        )
        rows = [{"stock_code": "000001"}]
        enriched = enrich_rows_with_review_prices(conn, "2026-05-15", rows)
        assert enriched[0]["t1_open"] == 10.0
        assert enriched[0]["t1_close"] == 10.5
        assert enriched[0]["t2_open"] == 11.0
        assert enriched[0]["t2_close"] == 11.2


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
