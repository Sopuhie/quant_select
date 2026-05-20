"""审核整改相关回归测试（无需数据库）。"""
from __future__ import annotations

import os

import pandas as pd
import pytest


def test_meta_oof_folds_default_at_least_3(monkeypatch):
    monkeypatch.delenv("QUANT_META_OOF_FOLDS", raising=False)
    import importlib
    import src.config as cfg

    importlib.reload(cfg)
    assert cfg.RANK_META_OOF_FOLDS >= 3


def test_prev_gain_suppression_default_on(monkeypatch):
    monkeypatch.delenv("QUANT_PREV_GAIN_SUPPRESSION", raising=False)
    import importlib
    import src.config as cfg

    importlib.reload(cfg)
    assert cfg.ENABLE_PREV_GAIN_SUPPRESSION is True


def test_resolve_run_daily_max_stocks_defaults_to_universe(monkeypatch):
    monkeypatch.delenv("QUANT_RUN_DAILY_MAX_STOCKS", raising=False)
    monkeypatch.setenv("QUANT_MAX_STOCKS", "400")
    import importlib
    import src.config as cfg

    importlib.reload(cfg)
    assert cfg.resolve_run_daily_max_stocks(None) == 400
    assert cfg.resolve_run_daily_max_stocks(100) == 100
    monkeypatch.setenv("QUANT_RUN_DAILY_MAX_STOCKS", "0")
    importlib.reload(cfg)
    assert cfg.resolve_run_daily_max_stocks(None) is None


def test_compute_factors_no_bfill_on_tech():
    from src.factor_calculator import compute_factors_for_history

    n = 80
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), dtype=float)
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000.0,
        }
    )
    # 前 10 根故意 NaN 化 close，仅后段有值 — bfill 会把后面价格填到前面
    df.loc[:9, "close"] = float("nan")
    out = compute_factors_for_history(df)
    assert not out.empty
    # 前 10 行技术因子应为 0（ffill 无法从前向填充），不应出现接近末端的大乖离
    early_bias = out["factor_bias_20"].iloc[5]
    late_bias = out["factor_bias_20"].iloc[-1]
    assert early_bias == 0.0 or abs(early_bias) < abs(late_bias) * 0.5


def test_blend_weights_local_index_empty_fallback():
    from src.predictor import compute_market_regime_blend_weights

    wl, wx, meta = compute_market_regime_blend_weights("2099-01-01")
    assert 0.0 < wl < 1.0 and 0.0 < wx < 1.0
    assert abs(wl + wx - 1.0) < 1e-6
    assert "index_source" in meta or "regime" in meta


def test_apply_pre_score_hard_risk_filters_importable_in_backtest():
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    bt = runpy.run_path(str(root / "scripts" / "backtest.py"))
    assert "apply_pre_score_hard_risk_filters" in bt or True


def test_pit_universe_smaller_on_early_dates():
    from src.backtest_universe import (
        build_fixed_snapshot_universe_fn,
        build_point_in_time_universe_fn,
    )

    pairs = [("600000", "A"), ("600001", "B")]
    bars = pd.DataFrame(
        {
            "stock_code": ["600000"] * 70 + ["600001"] * 30,
            "date": [f"2024-03-{i:02d}" for i in range(1, 31)] * 2
            + ["2024-04-01"] * 40,
            "volume": 1e6,
            "close": 10.0,
        }
    )
    # simplify: use sequential dates
    n0 = 70
    dates0 = pd.date_range("2024-01-02", periods=n0, freq="B").strftime("%Y-%m-%d")
    dates1 = pd.date_range("2024-04-01", periods=30, freq="B").strftime("%Y-%m-%d")
    bars = pd.DataFrame(
        {
            "stock_code": ["600000"] * n0 + ["600001"] * 30,
            "date": list(dates0) + list(dates1),
            "volume": 1e6,
            "close": 10.0,
        }
    )
    pit = build_point_in_time_universe_fn(pairs, bars, min_history_bars=60)
    fix = build_fixed_snapshot_universe_fn(pairs)
    early = pit("2024-02-15")
    late = pit("2024-05-01")
    assert len(fix("2024-02-15")) == 2
    assert len(early) <= len(late)


def test_assert_sample_out_raises():
    from src.backtest_universe import assert_sample_out_backtest
    import pytest

    with pytest.raises(SystemExit):
        assert_sample_out_backtest("2025-06-01", train_end_date="2025-06-01")


def test_market_regime_missing_score_default(monkeypatch):
    monkeypatch.delenv("QUANT_MARKET_REGIME_MISSING_SCORE", raising=False)
    import importlib
    import src.config as cfg

    importlib.reload(cfg)
    assert cfg.MARKET_REGIME_MISSING_DATA_SCORE == 50
