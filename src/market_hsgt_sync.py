"""全市场北向资金净流入：AkShare 下载并持久化至 SQLite（因子层只读本地）。"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ensure_eastmoney_no_proxy_if_configured
from .database import upsert_market_hsgt_flow_rows


def _parse_hsgt_hist_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["trade_date", "net_inflow"])
    df = raw.copy()
    date_col = next(
        (c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"),
        df.columns[0],
    )
    val_col = next(
        (c for c in df.columns if "净流入" in str(c)),
        None,
    )
    if val_col is None:
        val_col = next(
            (
                c
                for c in df.columns
                if "净" in str(c)
                and "流入" in str(c)
                and "成交" not in str(c)
            ),
            None,
        )
    if val_col is None:
        val_col = next(
            (
                c
                for c in df.columns
                if "净" in str(c)
                or "流入" in str(c)
                or str(c).lower() in ("value", "net", "amount")
            ),
            None,
        )
    if val_col is None:
        num_cols = [
            c
            for c in df.columns
            if c != date_col and str(df[c].dtype).startswith(("float", "int"))
        ]
        if not num_cols:
            return pd.DataFrame(columns=["trade_date", "net_inflow"])
        val_col = num_cols[-1]
    ser = pd.to_numeric(df[val_col], errors="coerce")
    dt = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    g = pd.DataFrame({"trade_date": dt, "net_inflow": ser}).dropna(
        subset=["trade_date", "net_inflow"]
    )
    return g


def _fetch_hsgt_hist_raw() -> pd.DataFrame | None:
    """AkShare 当前版本 ``stock_hsgt_hist_em`` 仅接受 ``symbol``，返回全历史。"""
    ensure_eastmoney_no_proxy_if_configured()
    import akshare as ak  # type: ignore[import-untyped]

    last_exc: Exception | None = None
    for sym in ("北向资金", "沪港通", "沪股通"):
        try:
            raw = ak.stock_hsgt_hist_em(symbol=sym)
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return None


def download_market_hsgt_flow_em(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """从 AkShare 拉取北向资金历史净流入（仅用于同步脚本，禁止在 factor_calculator 内调用）。"""
    start_s = str(start_date).strip()[:10]
    end_s = str(end_date).strip()[:10]
    raw = _fetch_hsgt_hist_raw()
    g = _parse_hsgt_hist_df(raw) if raw is not None else pd.DataFrame()
    if g.empty:
        return g
    m = (g["trade_date"] >= start_s) & (g["trade_date"] <= end_s)
    return g.loc[m].reset_index(drop=True)


def sync_market_hsgt_flow_to_db(
    start_date: str,
    end_date: str,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """下载并 upsert 至 ``market_hsgt_flow_daily``。"""
    g = download_market_hsgt_flow_em(start_date, end_date)
    if g.empty:
        if verbose:
            print("[北向全市场] 未获取到有效净流入数据。", flush=True)
        return {"rows": 0}
    rows = g.to_dict("records")
    n = upsert_market_hsgt_flow_rows(rows)
    if verbose:
        print(f"[北向全市场] 已写入/更新 {n} 个交易日净流入。", flush=True)
    return {"rows": int(n)}
