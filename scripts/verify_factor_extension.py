"""
本地验证：PIT 财报对齐、扩展因子计算、截面清洗（不依赖外网）。

用法（项目根目录）:
  python scripts/verify_factor_extension.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import ALT_FEATURE_COLUMNS, FEATURE_COLUMNS, TECH_FEATURE_COLUMNS
from src.database import fetch_joined_fundamental_moneyflow_panel, init_db
from src.factor_calculator import clean_cross_sectional_features, compute_factors_for_history
from src.panel_enrichment import merge_point_in_time_financials


def _test_pit_no_lookahead() -> None:
    hist = pd.DataFrame(
        {
            "date": [f"2025-01-{d:02d}" for d in range(1, 11)],
            "close": np.linspace(10, 11, 10),
            "open": np.linspace(10, 11, 10),
            "high": np.linspace(10.2, 11.2, 10),
            "low": np.linspace(9.8, 10.8, 10),
            "volume": np.full(10, 1e6),
        }
    )
    fin = pd.DataFrame(
        {
            "pub_date": ["2025-01-05", "2025-01-09"],
            "roe": [10.0, 20.0],
            "net_profit_growth": [5.0, 15.0],
            "revenue_growth": [8.0, 18.0],
        }
    )
    merged = merge_point_in_time_financials(hist, fin)
    before = merged.loc[merged["date"] < "2025-01-05", "roe"]
    assert before.isna().all(), "公告日前不应可见 ROE"
    assert float(merged.loc[merged["date"] == "2025-01-05", "roe"].iloc[0]) == 10.0
    assert float(merged.loc[merged["date"] == "2025-01-08", "roe"].iloc[0]) == 10.0
    assert float(merged.loc[merged["date"] == "2025-01-09", "roe"].iloc[0]) == 20.0
    print("[OK] PIT 财报 merge_asof 无前视")


def _test_compute_factors_shape() -> None:
    n = 80
    hist = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
            "close": 10 + np.cumsum(np.random.randn(n) * 0.02),
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "volume": np.random.uniform(1e6, 2e6, n),
            "turnover_rate": np.random.uniform(1, 8, n),
            "pe_ttm": np.random.uniform(8, 40, n),
            "roe": np.full(n, 12.0),
            "net_profit_growth": np.full(n, 10.0),
            "revenue_growth": np.full(n, 8.0),
            "big_net_ratio": np.random.randn(n) * 0.01,
        }
    )
    fac = compute_factors_for_history(hist)
    assert list(fac.columns) == FEATURE_COLUMNS
    assert len(TECH_FEATURE_COLUMNS) == 13 and len(ALT_FEATURE_COLUMNS) == 4
    assert not fac.empty
    assert fac["factor_ep_ttm"].notna().any()
    print("[OK] compute_factors_for_history 输出 17 维")


def _test_cross_section_clean() -> None:
    rows = []
    for i in range(12):
        rows.append(
            {
                "date": "2025-05-15",
                "stock_code": f"{i:06d}",
                "industry": "银行" if i < 6 else "计算机",
                "mcap": 1e10 * (1 + i),
                **{c: float(np.random.randn()) for c in FEATURE_COLUMNS},
            }
        )
    df = pd.DataFrame(rows)
    cleaned = clean_cross_sectional_features(df)
    assert len(cleaned) == len(df)
    for c in FEATURE_COLUMNS:
        assert c in cleaned.columns
    print("[OK] clean_cross_sectional_features 17 维截面清洗")


def _test_db_join() -> None:
    import sqlite3

    from src.config import DB_PATH

    if not DB_PATH.is_file():
        print("[SKIP] 无本地 stocks.db，跳过 fetch_joined 测试")
        return
    try:
        panel = fetch_joined_fundamental_moneyflow_panel(
            "2026-05-15", ["000001", "600519"]
        )
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            print("[SKIP] stocks.db 被占用（如 Streamlit 正在运行），跳过 fetch_joined 测试")
            return
        raise
    assert "stock_code" in panel.columns
    print(f"[OK] fetch_joined_fundamental_moneyflow_panel 行数={len(panel)}")


def main() -> int:
    _test_pit_no_lookahead()
    _test_compute_factors_shape()
    _test_cross_section_clean()
    _test_db_join()
    print("\n全部本地校验通过。")
    print(
        "提示：扩展因子后需重训模型: python train_model.py --fast-train --max-stocks 200"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
