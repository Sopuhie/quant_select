"""
沪深300等指数日线同步至本地 ``index_daily``，供 ``market_regime`` / ``run_daily`` 择时熔断使用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH, ensure_eastmoney_no_proxy_if_configured
from .database import (
    fetch_index_daily_max_date,
    init_db,
    upsert_index_daily_rows,
)
from .utils import get_kline_incremental_end_trade_date

HS300_INDEX_CODE = "000300"

_INDEX_SOURCES: dict[str, dict[str, str]] = {
    HS300_INDEX_CODE: {
        "ak_hist_symbol": "000300",
        "em_name": "沪深300",
        "sina_symbol": "sh000300",
    },
    "000852": {
        "ak_hist_symbol": "000852",
        "em_name": "中证1000",
        "sina_symbol": "sh000852",
    },
}


def _normalize_ak_index_frame(
    df: pd.DataFrame,
    *,
    index_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """将 AkShare 指数表规范为 index_daily 行（列：index_code, date, open, high, low, close, volume）。"""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["index_code", "date", "open", "high", "low", "close", "volume"]
        )

    date_col = next(
        (c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"),
        df.columns[0],
    )
    close_col = next(
        (
            c
            for c in df.columns
            if "收盘" in str(c) or str(c).lower() in ("close", "收盘价")
        ),
        None,
    )
    if close_col is None:
        return pd.DataFrame(
            columns=["index_code", "date", "open", "high", "low", "close", "volume"]
        )

    open_col = next((c for c in df.columns if "开盘" in str(c)), None)
    high_col = next((c for c in df.columns if "最高" in str(c)), None)
    low_col = next((c for c in df.columns if "最低" in str(c)), None)
    vol_col = next(
        (c for c in df.columns if "成交量" in str(c) or str(c).lower() == "volume"),
        None,
    )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    out["open"] = (
        pd.to_numeric(df[open_col], errors="coerce") if open_col is not None else pd.NA
    )
    out["high"] = (
        pd.to_numeric(df[high_col], errors="coerce") if high_col is not None else pd.NA
    )
    out["low"] = (
        pd.to_numeric(df[low_col], errors="coerce") if low_col is not None else pd.NA
    )
    out["volume"] = (
        pd.to_numeric(df[vol_col], errors="coerce") if vol_col is not None else pd.NA
    )
    out["index_code"] = str(index_code).strip().zfill(6)
    out = out.dropna(subset=["date", "close"])
    if start_date:
        out = out[out["date"] >= str(start_date)[:10]]
    if end_date:
        out = out[out["date"] <= str(end_date)[:10]]
    return out.sort_values("date").reset_index(drop=True)


def fetch_index_daily_hist(
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    从 AkShare 拉取指数日线（与 ``scripts/backtest.fetch_benchmark_close`` 同源接口）。
    """
    code = str(index_code).strip().zfill(6)
    meta = _INDEX_SOURCES.get(code)
    if meta is None:
        raise ValueError(f"未配置的指数代码: {code}")

    ensure_eastmoney_no_proxy_if_configured()
    import akshare as ak  # type: ignore[import-untyped]

    s = str(start_date).replace("-", "")[:8]
    e = str(end_date).replace("-", "")[:8]
    raw = pd.DataFrame()

    try:
        tmp = ak.index_zh_a_hist(
            symbol=meta["ak_hist_symbol"],
            period="daily",
            start_date=s,
            end_date=e,
        )
        if tmp is not None and not tmp.empty:
            raw = tmp
    except Exception:
        pass

    out = _normalize_ak_index_frame(
        raw, index_code=code, start_date=start_date, end_date=end_date
    )
    if not out.empty:
        return out

    try:
        tmp = ak.stock_zh_index_daily_em(symbol=meta["em_name"])
        if tmp is not None and not tmp.empty:
            out = _normalize_ak_index_frame(
                tmp, index_code=code, start_date=start_date, end_date=end_date
            )
    except Exception:
        pass

    if out.empty and meta.get("sina_symbol"):
        try:
            tmp = ak.stock_zh_index_daily(symbol=str(meta["sina_symbol"]))
            if tmp is not None and not tmp.empty:
                out = _normalize_ak_index_frame(
                    tmp, index_code=code, start_date=start_date, end_date=end_date
                )
        except Exception:
            pass

    return out


def sync_index_daily(
    db_path: Path | None = None,
    *,
    index_codes: list[str] | None = None,
    initial_lookback_days: int = 550,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    增量同步指数日线至 ``index_daily``。

    - 无本地记录：回溯 ``initial_lookback_days`` 自然日；
    - 已有记录：从最近交易日次日拉至 ``get_kline_incremental_end_trade_date()``。
    """
    path = db_path or DB_PATH
    init_db(path)
    codes = index_codes or [HS300_INDEX_CODE, "000852"]
    end_date = get_kline_incremental_end_trade_date()
    stats: dict[str, Any] = {
        "end_date": end_date,
        "indices": {},
        "rows_upserted": 0,
    }

    for code in codes:
        code = str(code).strip().zfill(6)
        if code not in _INDEX_SOURCES:
            if verbose:
                print(f"[指数] 跳过未配置指数 {code}", flush=True)
            continue

        last = fetch_index_daily_max_date(code, path)
        if last and str(last)[:10] >= end_date:
            if verbose:
                print(
                    f"[指数] {code} 已对齐至 {end_date}，跳过拉取。",
                    flush=True,
                )
            stats["indices"][code] = {"skipped": True, "last_date": last, "rows": 0}
            continue

        if last:
            start_date = (
                pd.Timestamp(str(last)[:10]) + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            start_date = (
                pd.Timestamp(end_date) - pd.Timedelta(days=int(initial_lookback_days))
            ).strftime("%Y-%m-%d")

        if start_date > end_date:
            stats["indices"][code] = {"skipped": True, "last_date": last, "rows": 0}
            continue

        label = _INDEX_SOURCES[code].get("em_name", code)
        if verbose:
            print(
                f"[指数] 拉取 {label}({code})：{start_date} ~ {end_date} …",
                flush=True,
            )

        df = fetch_index_daily_hist(code, start_date, end_date)
        if df.empty:
            if verbose:
                print(f"[指数] {code} 未获取到行情，请检查网络或 AkShare 接口。", flush=True)
            stats["indices"][code] = {
                "skipped": False,
                "last_date": last,
                "rows": 0,
                "error": "empty_fetch",
            }
            continue

        rows = df.to_dict(orient="records")
        n = upsert_index_daily_rows(rows, db_path=path)
        new_last = str(df["date"].iloc[-1])[:10]
        if verbose:
            print(
                f"[指数] {code} 已入库 {n} 条，本地最新 {new_last}。",
                flush=True,
            )
        stats["indices"][code] = {
            "skipped": False,
            "last_date": new_last,
            "rows": n,
            "start_date": start_date,
        }
        stats["rows_upserted"] += int(n)

    return stats
