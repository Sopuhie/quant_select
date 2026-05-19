"""K 线历史面板：按 ``pub_date`` 对齐基本面、合并资金流（防前视）。"""
from __future__ import annotations

import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DB_PATH


def _norm_code(code: object) -> str:
    return str(code).strip().zfill(6)


def load_financial_panels_bulk(
    stock_codes: Iterable[str],
    *,
    db_path: object | None = None,
) -> dict[str, pd.DataFrame]:
    """批量读取 ``stock_financial_data``，按 ``pub_date`` 升序。"""
    codes = sorted({_norm_code(c) for c in stock_codes if len(_norm_code(c)) == 6})
    if not codes:
        return {}
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    try:
        out: dict[str, pd.DataFrame] = {}
        chunk = 400
        for i in range(0, len(codes), chunk):
            sub = codes[i : i + chunk]
            ph = ",".join("?" * len(sub))
            df = pd.read_sql_query(
                f"""
                SELECT stock_code, pub_date, report_date, roe, net_profit_growth, revenue_growth
                FROM stock_financial_data
                WHERE stock_code IN ({ph})
                ORDER BY stock_code, pub_date ASC, report_date ASC
                """,
                conn,
                params=tuple(sub),
            )
            if df.empty:
                continue
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
            df["pub_date"] = df["pub_date"].astype(str).str[:10]
            for code, g in df.groupby("stock_code", sort=False):
                out[str(code)] = g.reset_index(drop=True)
        return out
    finally:
        conn.close()


def load_money_flow_panel(
    stock_code: str,
    *,
    end_date: str | None = None,
    db_path: object | None = None,
) -> pd.DataFrame:
    """单股日频 ``big_net_ratio``（``trade_date`` 对齐 K 线 ``date``）。"""
    code = _norm_code(stock_code)
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    try:
        if end_date:
            df = pd.read_sql_query(
                """
                SELECT trade_date AS date, big_net_ratio
                FROM stock_money_flow_daily
                WHERE stock_code = ? AND trade_date <= ?
                ORDER BY trade_date ASC
                """,
                conn,
                params=(code, str(end_date).strip()[:10]),
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT trade_date AS date, big_net_ratio
                FROM stock_money_flow_daily
                WHERE stock_code = ?
                ORDER BY trade_date ASC
                """,
                conn,
                params=(code,),
            )
    finally:
        conn.close()
    if df.empty:
        return df
    df["date"] = df["date"].astype(str).str[:10]
    df["big_net_ratio"] = pd.to_numeric(df["big_net_ratio"], errors="coerce")
    return df


def merge_point_in_time_financials(
    hist: pd.DataFrame,
    fin: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    对每个交易日仅使用 ``pub_date <= date`` 的最新财报行（``merge_asof`` backward）。
    """
    if hist is None or hist.empty:
        return hist
    out = hist.copy()
    out["date"] = out["date"].astype(str).str[:10]
    for col in ("roe", "net_profit_growth", "revenue_growth"):
        out.drop(columns=[col], errors="ignore", inplace=True)
    if fin is None or fin.empty:
        for col in ("roe", "net_profit_growth", "revenue_growth"):
            out[col] = np.nan
        return out
    f = fin.copy()
    f["pub_date"] = pd.to_datetime(f["pub_date"], errors="coerce")
    out["_dt"] = pd.to_datetime(out["date"], errors="coerce")
    f = f.dropna(subset=["pub_date"]).sort_values("pub_date")
    out = out.sort_values("_dt")
    merged = pd.merge_asof(
        out,
        f[["pub_date", "roe", "net_profit_growth", "revenue_growth"]],
        left_on="_dt",
        right_on="pub_date",
        direction="backward",
    )
    merged = merged.drop(columns=["_dt", "pub_date"], errors="ignore")
    return merged


def merge_money_flow_on_dates(hist: pd.DataFrame, mf: pd.DataFrame | None) -> pd.DataFrame:
    """按 ``date`` 左连接 ``big_net_ratio``（无未来数据）。"""
    if hist is None or hist.empty:
        return hist
    out = hist.copy()
    out["date"] = out["date"].astype(str).str[:10]
    if mf is None or mf.empty:
        if "big_net_ratio" not in out.columns:
            out["big_net_ratio"] = np.nan
        return out
    m = mf.copy()
    m["date"] = m["date"].astype(str).str[:10]
    out = out.merge(m[["date", "big_net_ratio"]], on="date", how="left")
    return out


def enrich_ohlcv_history(
    hist: pd.DataFrame,
    *,
    stock_code: str,
    financial_cache: dict[str, pd.DataFrame] | None = None,
    db_path: object | None = None,
) -> pd.DataFrame:
    """
    为单股 K 线附加：PIT 财务三指标、日频资金流、保留 ``turnover_rate`` / ``pe_ttm`` 列。
    """
    if hist is None or hist.empty:
        return hist
    code = _norm_code(stock_code)
    end = str(hist["date"].astype(str).str[:10].max())
    fin: pd.DataFrame | None
    if financial_cache is not None:
        fin = financial_cache.get(code)
    else:
        from .database import fetch_stock_financial_panel

        fin = fetch_stock_financial_panel(code, db_path=db_path)  # type: ignore[arg-type]
    w = merge_point_in_time_financials(hist, fin)
    mf = load_money_flow_panel(code, end_date=end, db_path=db_path)
    w = merge_money_flow_on_dates(w, mf)
    if "turnover_rate" in w.columns:
        w["turnover_rate"] = pd.to_numeric(w["turnover_rate"], errors="coerce")
    if "pe_ttm" in w.columns:
        w["pe_ttm"] = pd.to_numeric(w["pe_ttm"], errors="coerce")
    return w
