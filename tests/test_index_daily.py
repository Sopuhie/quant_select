"""index_daily 建表、写入与 market_regime 打分回归。"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def _make_hs300_above_ma20_db(db_path: Path, *, n: int = 30) -> None:
    from src.database import init_db, upsert_index_daily_rows

    init_db(db_path)
    dates = pd.date_range("2026-03-01", periods=n, freq="B")
    closes = [3000.0 + i * 10.0 for i in range(n)]
    rows = [
        {
            "index_code": "000300",
            "date": d.strftime("%Y-%m-%d"),
            "close": c,
        }
        for d, c in zip(dates, closes)
    ]
    upsert_index_daily_rows(rows, db_path=db_path)


def test_init_db_creates_index_daily_table():
    from src.database import init_db

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        init_db(db)
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='index_daily'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None


def test_market_regime_score_from_index_daily():
    from src.market_regime import get_hs300_market_environment_score

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        _make_hs300_above_ma20_db(db)
        score = get_hs300_market_environment_score("2026-05-25", db_path=db)
        assert score == 60


def test_market_regime_missing_table_scores_50():
    from src import market_regime as mr

    mr._SCORE_CACHE.clear()
    mr._MISSING_INDEX_WARNED.clear()

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "empty.db"
        from src.database import init_db

        init_db(db)
        score = mr.get_hs300_market_environment_score("2026-05-25", db_path=db)
        assert score == 50


def test_fetch_index_daily_max_date():
    from src.database import fetch_index_daily_max_date

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        _make_hs300_above_ma20_db(db)
        mx = fetch_index_daily_max_date("000300", db_path=db)
        assert mx is not None
        assert str(mx)[:10] >= "2026-04-01"


@pytest.mark.network
def test_fetch_index_daily_hist_network_smoke():
    pytest.importorskip("akshare")
    from src.index_daily_sync import fetch_index_daily_hist

    df = fetch_index_daily_hist("000300", "2025-01-02", "2025-06-30")
    if df.empty:
        pytest.skip("AkShare 未返回沪深300样本区间数据（网络或接口限流）")
    assert "000300" in df["index_code"].unique()
