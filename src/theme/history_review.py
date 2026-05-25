# -*- coding: utf-8 -*-
"""热门题材历史选股复盘：查询与展示用 DataFrame。"""
from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from .db import ensure_theme_tables, load_theme_selections_df
from .return_fill import fill_theme_returns_for_date


def list_theme_selection_trade_dates(conn: sqlite3.Connection) -> list[str]:
    ensure_theme_tables(conn)
    rows = conn.execute(
        """
        SELECT DISTINCT trade_date
        FROM theme_daily_selections
        ORDER BY trade_date DESC
        """
    ).fetchall()
    return [str(r[0]).strip()[:10] for r in rows if r[0]]


def format_theme_selections_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for src, label in (
        ("ret_1d", "1日收益率"),
        ("ret_5d", "5日收益率"),
        ("ret_10d", "10日收益率"),
        ("ret_60d", "60日收益率"),
    ):
        if src in out.columns:
            out[label] = out[src].apply(
                lambda x: f"{float(x):.2%}" if pd.notna(x) else "—"
            )
            out = out.drop(columns=[src])
    if "选股日收盘价" in out.columns:
        out["选股日收盘价"] = pd.to_numeric(
            out["选股日收盘价"], errors="coerce"
        ).round(2)
    if "五日量比" in out.columns:
        out["五日量比"] = pd.to_numeric(out["五日量比"], errors="coerce").round(2)
    for col in ("KDJ_J", "MACD柱"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    order = [
        "排名",
        "代码",
        "名称",
        "题材标签",
        "五日量比",
        "KDJ_J",
        "MACD柱",
        "选股日收盘价",
        "1日收益率",
        "5日收益率",
        "10日收益率",
        "60日收益率",
        "实盘决策建议结论",
        "大盘环境分",
        "筛选关键词",
    ]
    return out[[c for c in order if c in out.columns]]


def load_theme_review_bundle(
    conn: sqlite3.Connection,
    trade_date: str,
    *,
    auto_fill_returns: bool = True,
) -> dict[str, Any]:
    td = str(trade_date).strip()[:10]
    fill_stats: dict[str, int] = {}
    if auto_fill_returns:
        fill_stats = fill_theme_returns_for_date(conn, td, commit=True)
    sel = load_theme_selections_df(conn, td)
    mkt = None
    if not sel.empty and "大盘环境分" in sel.columns:
        s = pd.to_numeric(sel["大盘环境分"], errors="coerce").dropna()
        if len(s):
            mkt = int(s.iloc[0])
    return {
        "trade_date": td,
        "selections": sel,
        "selections_display": format_theme_selections_display_df(sel),
        "count": len(sel),
        "market_score": mkt,
        "return_fill": fill_stats,
    }
