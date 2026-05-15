"""
AkShare 增量补全：按缺失 (trade_date, stock_code) 拉取个股资金流与北向持股，写入 SQLite。

- 资金流：东财个股资金流 ``stock_individual_fund_flow``（日 K 线接口），大单+超大单「净占比」合成 ``big_net_ratio``。
  该接口通常仅返回**近端约百余个交易日**；早于返回区间的日期无法通过本接口回补（缺失键会留空，仍走 OHLCV 代理）。
- 北向：``stock_hsgt_individual_em``，取持股占 A 股比例列及其日差为 ``hold_pct_chg``（历史行相对更长，仍以接口实际返回为准）。

请求间默认小睡以降低限流概率；可通过环境变量 ``QUANT_AUX_SYNC_SLEEP`` 调整。
"""
from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ensure_eastmoney_no_proxy_if_configured
from .database import (
    fetch_existing_auxiliary_key_sets,
    init_db,
    upsert_stock_money_flow_rows,
    upsert_stock_north_hold_rows,
)


def _ak_money_flow_market(code6: str) -> str:
    c = str(code6).strip().zfill(6)
    if c.startswith(("0", "3")):
        return "sz"
    if c.startswith("6"):
        return "sh"
    if c.startswith(("4", "8", "9")):
        return "bj"
    return "sh"


def _download_money_flow_raw(code6: str) -> pd.DataFrame | None:
    ensure_eastmoney_no_proxy_if_configured()
    try:
        import akshare as ak  # type: ignore[import-untyped]
    except Exception:
        return None
    c = str(code6).strip().zfill(6)
    for mk in (_ak_money_flow_market(c), "sh", "sz"):
        try:
            raw = ak.stock_individual_fund_flow(stock=c, market=mk)
            if raw is not None and not getattr(raw, "empty", True):
                return raw
        except Exception:
            continue
    return None


def _money_flow_ultra_large_pct_columns(df: pd.DataFrame) -> tuple[int | None, int | None]:
    """返回「超大单净占比」「大单净占比」列位置；无法识别时退回 (6, 8)。"""
    ultra_i: int | None = None
    large_i: int | None = None
    for i, c in enumerate(df.columns):
        s = str(c)
        if "超大" in s and ("占比" in s or "净占比" in s):
            ultra_i = i
        if "大单" in s and "超大" not in s and ("占比" in s or "净占比" in s):
            large_i = i
    if ultra_i is not None and large_i is not None:
        return ultra_i, large_i
    if df.shape[1] >= 9:
        return 6, 8
    return None, None


def _money_flow_rows_for_dates(
    df: pd.DataFrame,
    code6: str,
    want_dates: set[str],
) -> list[tuple[str, str, float]]:
    """从东财返回表中筛 ``want_dates``，输出 (trade_date, code, big_net_ratio)。"""
    if df is None or df.empty or df.shape[1] < 3:
        return []
    dser = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    ui, li = _money_flow_ultra_large_pct_columns(df)
    if ui is None or li is None:
        return []
    u = pd.to_numeric(df.iloc[:, ui], errors="coerce")
    l = pd.to_numeric(df.iloc[:, li], errors="coerce")
    comb = u.fillna(0.0) + l.fillna(0.0)
    finite = comb.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) > 0 and float(finite.abs().quantile(0.9)) > 1.5:
        comb = comb / 100.0
    out: list[tuple[str, str, float]] = []
    for i in range(len(df)):
        if pd.isna(dser.iloc[i]):
            continue
        ds = dser.iloc[i].strftime("%Y-%m-%d")[:10]
        if ds not in want_dates:
            continue
        v = float(comb.iloc[i]) if pd.notna(comb.iloc[i]) else float("nan")
        if np.isfinite(v):
            out.append((ds, str(code6).strip().zfill(6), float(v)))
    return out


def _download_hsgt_raw(code6: str) -> pd.DataFrame | None:
    ensure_eastmoney_no_proxy_if_configured()
    try:
        import akshare as ak  # type: ignore[import-untyped]
    except Exception:
        return None
    c = str(code6).strip().zfill(6)
    try:
        raw = ak.stock_hsgt_individual_em(symbol=c)
        if raw is not None and not getattr(raw, "empty", True):
            return raw
    except Exception:
        return None
    return None


def _north_hold_rows_for_dates(
    df: pd.DataFrame,
    code6: str,
    want_dates: set[str],
) -> list[tuple[str, str, float, float]]:
    """持股占 A 股比例（列推断）及相邻交易日差分，筛 ``want_dates``。"""
    if df is None or df.empty or df.shape[1] < 6:
        return []
    dser = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    hp = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    sub = pd.DataFrame({"dt": dser, "hp": hp}).dropna(subset=["dt"])
    sub = sub.sort_values("dt").reset_index(drop=True)
    sub["chg"] = sub["hp"].diff()
    out: list[tuple[str, str, float, float]] = []
    for i in range(len(sub)):
        ds = sub["dt"].iloc[i].strftime("%Y-%m-%d")[:10]
        if ds not in want_dates:
            continue
        hpv = float(sub["hp"].iloc[i]) if pd.notna(sub["hp"].iloc[i]) else 0.0
        cgv = float(sub["chg"].iloc[i]) if pd.notna(sub["chg"].iloc[i]) else 0.0
        out.append((ds, str(code6).strip().zfill(6), hpv, cgv))
    return out


def sync_auxiliary_for_date_code_grid(
    dates: Iterable[str],
    codes: Iterable[str],
    *,
    fill_money_flow: bool = True,
    fill_north: bool = True,
    sleep_sec: float | None = None,
    max_codes: int | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """
    对 ``dates × codes`` 全网格，找出库中缺失键，按股票逐只拉 AkShare 并 upsert。

    ``max_codes``：最多对多少只不同的股票发起网络请求（其余缺失键留待下次）；``None`` 不限制。
    返回统计：``mf_rows``, ``nh_rows``, ``mf_codes_fetched``, ``nh_codes_fetched``, ``errors``。
    """
    init_db()
    ds = sorted({str(d).strip()[:10] for d in dates if d})
    cs_all = sorted({str(c).strip().zfill(6) for c in codes if c})
    if not ds or not cs_all:
        return {"mf_rows": 0, "nh_rows": 0, "mf_codes_fetched": 0, "nh_codes_fetched": 0, "errors": 0}

    mf_ex, nh_ex = fetch_existing_auxiliary_key_sets(ds, cs_all)
    required = {(d, c) for d in ds for c in cs_all}
    mf_miss = required - mf_ex if fill_money_flow else set()
    nh_miss = required - nh_ex if fill_north else set()

    mf_by_code: dict[str, set[str]] = {}
    for d, c in mf_miss:
        mf_by_code.setdefault(c, set()).add(d)
    nh_by_code: dict[str, set[str]] = {}
    for d, c in nh_miss:
        nh_by_code.setdefault(c, set()).add(d)

    stats = {"mf_rows": 0, "nh_rows": 0, "mf_codes_fetched": 0, "nh_codes_fetched": 0, "errors": 0}

    if sleep_sec is None:
        try:
            sleep_sec = float(os.environ.get("QUANT_AUX_SYNC_SLEEP", "0.25"))
        except ValueError:
            sleep_sec = 0.25

    mf_codes_todo = sorted(mf_by_code.keys())
    nh_codes_todo = sorted(nh_by_code.keys())
    codes_union = sorted(set(mf_codes_todo) | set(nh_codes_todo))
    if max_codes is not None and int(max_codes) > 0:
        codes_union = codes_union[: int(max_codes)]

    for code in codes_union:
        if fill_money_flow and code in mf_by_code:
            want = mf_by_code.get(code, set())
            if want:
                try:
                    raw = _download_money_flow_raw(code)
                    rows = (
                        _money_flow_rows_for_dates(raw, code, want) if raw is not None else []
                    )
                    if rows:
                        upsert_stock_money_flow_rows(rows)
                        stats["mf_rows"] += len(rows)
                    stats["mf_codes_fetched"] += 1
                except Exception:
                    stats["errors"] += 1
                if sleep_sec > 0:
                    time.sleep(float(sleep_sec))

        if fill_north and code in nh_by_code:
            want = nh_by_code.get(code, set())
            if want:
                try:
                    raw = _download_hsgt_raw(code)
                    rows = (
                        _north_hold_rows_for_dates(raw, code, want)
                        if raw is not None
                        else []
                    )
                    if rows:
                        upsert_stock_north_hold_rows(rows)
                        stats["nh_rows"] += len(rows)
                    stats["nh_codes_fetched"] += 1
                except Exception:
                    stats["errors"] += 1
                if sleep_sec > 0:
                    time.sleep(float(sleep_sec))

    if verbose and (stats["mf_rows"] or stats["nh_rows"] or stats["errors"]):
        print(
            "[auxiliary_ak_sync] "
            f"资金流 upsert 行={stats['mf_rows']} 股票请求数={stats['mf_codes_fetched']}; "
            f"北向 upsert 行={stats['nh_rows']} 股票请求数={stats['nh_codes_fetched']}; "
            f"errors={stats['errors']}",
            flush=True,
        )
    return stats


def maybe_autofill_auxiliary_from_env(
    dates: list[str],
    codes: list[str],
    *,
    verbose: bool = True,
) -> None:
    """
    当 ``QUANT_AUTOFILL_AUX_FEATURES=1`` 时，对 ``dates × codes`` 尝试补库后再由调用方重新 ``fetch`` merge。
    可用 ``QUANT_AUX_AUTOFILL_MAX_CODES`` 限制单轮最多请求股票数（默认 120）。
    """
    flag = str(os.environ.get("QUANT_AUTOFILL_AUX_FEATURES", "")).strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    try:
        max_cells = int(os.environ.get("QUANT_AUX_AUTOFILL_MAX_CELLS", "12000"))
    except ValueError:
        max_cells = 12000
    if len(dates) * len(codes) > max_cells:
        if verbose:
            print(
                f"[auxiliary_ak_sync] 跳过自动补全：网格 {len(dates)}×{len(codes)} 超过 QUANT_AUX_AUTOFILL_MAX_CELLS={max_cells}",
                flush=True,
            )
        return
    try:
        max_codes = int(os.environ.get("QUANT_AUX_AUTOFILL_MAX_CODES", "120"))
    except ValueError:
        max_codes = 120
    if max_codes <= 0:
        max_codes = None
    sync_auxiliary_for_date_code_grid(
        dates,
        codes,
        sleep_sec=None,
        max_codes=max_codes,
        verbose=verbose,
    )
