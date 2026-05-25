# -*- coding: utf-8 -*-
"""热门题材历史落库与收益回填。"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from src.database import init_db
from src.theme.db import ensure_theme_tables, load_theme_selections_df, save_theme_selections
from src.theme.history_review import format_theme_selections_display_df, load_theme_review_bundle
from src.theme.return_fill import fill_theme_returns_for_date


def _make_kline(conn: sqlite3.Connection, code: str, dates_closes: list[tuple[str, float]]) -> None:
    for d, c in dates_closes:
        conn.execute(
            """
            INSERT INTO stock_daily_kline (
                date, stock_code, stock_name, open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (d, code, "测试股", c, c, c, c, 1_000_000),
        )


def test_save_and_fill_returns(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    init_db(db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    code = "600000"
    dates = [f"2024-01-{i:02d}" for i in range(1, 16)]
    closes = [10.0 + i for i in range(15)]
    _make_kline(conn, code, list(zip(dates, closes)))
    conn.commit()

    signal = "2024-01-05"
    theme_df = pd.DataFrame(
        [
            {
                "股票代码": code,
                "股票名称": "测试",
                "题材标签": "测试题材",
                "最新价格": "14.00 元",
                "当前量比": "2.50 倍",
                "KDJ_J值": 55.0,
                "MACD红柱": 0.12,
                "实盘决策建议结论": "买点",
            }
        ]
    )
    ensure_theme_tables(conn)
    n = save_theme_selections(conn, signal, theme_df, market_score=80)
    assert n == 1
    conn.commit()

    stats = fill_theme_returns_for_date(conn, signal, db_path=db, commit=True)
    assert stats["rows_touched"] == 1

    bundle = load_theme_review_bundle(conn, signal, auto_fill_returns=False)
    disp = bundle["selections_display"]
    assert "1日收益率" in disp.columns
    assert disp.iloc[0]["1日收益率"] != "—"

    raw = load_theme_selections_df(conn, signal)
    assert raw.iloc[0]["ret_1d"] is not None
    conn.close()
