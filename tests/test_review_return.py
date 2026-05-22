"""T1 买 T2 卖涨跌幅计算。"""
from src.short_term.review_prices import calc_t1_buy_t2_sell_return


def test_t1_open_to_t2_close_return():
    r = calc_t1_buy_t2_sell_return(10.0, 10.5, 11.0)
    assert r is not None
    assert abs(r - 0.1) < 1e-6


def test_fallback_t1_close_when_no_open():
    r = calc_t1_buy_t2_sell_return(None, 10.0, 10.5)
    assert r is not None
    assert abs(r - 0.05) < 1e-6


def test_missing_t2_returns_none():
    assert calc_t1_buy_t2_sell_return(10.0, 10.1, None) is None
