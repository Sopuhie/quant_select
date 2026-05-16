"""
东方财富概念板块：热门题材列表 + 成份股 → JSON / SQLite。

供「热门题材高爆选股」下拉与 ``stock_concept_boards`` 题材过滤使用。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import pandas as pd

from .board_stocks import BOARD_STOCKS_PATH, load_board_mapping, save_board_mapping
from .config import DATA_DIR, DB_PATH, ensure_eastmoney_no_proxy_if_configured
from .database import sync_concept_boards_from_json
from .hot_sectors import (
    HOT_SECTORS_PATH,
    build_default_tags,
    load_hot_sectors_meta,
    load_tags,
    save_tags,
)

DEFAULT_HOT_TOP_N = 15
DEFAULT_BOARD_SLEEP = 0.35
_FETCH_RETRIES = 3
_FETCH_RETRY_SLEEP = 2.0


def _with_retries(fn, *args, retries: int = _FETCH_RETRIES, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(_FETCH_RETRY_SLEEP * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry exhausted")


def _latest_kline_date(db_path=None) -> str:
    import sqlite3

    from .database import init_db

    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT MAX(date) FROM stock_daily_kline").fetchone()
        if row and row[0]:
            return str(row[0]).strip()[:10]
    finally:
        conn.close()
    return datetime.now().strftime("%Y-%m-%d")


def fetch_hot_concept_names_em(top_n: int = DEFAULT_HOT_TOP_N) -> list[str]:
    """从东方财富概念板块行情取涨幅靠前的板块名称（AkShare）。"""
    import akshare as ak

    ensure_eastmoney_no_proxy_if_configured()
    n = max(1, int(top_n))
    df = _with_retries(ak.stock_board_concept_name_em)
    if df is None or df.empty:
        return []
    name_col = "板块名称"
    if name_col not in df.columns:
        raise RuntimeError(f"概念板块表缺少「{name_col}」，列: {list(df.columns)}")
    sort_col = "涨跌幅"
    work = df.copy()
    if sort_col in work.columns:
        work[sort_col] = pd.to_numeric(work[sort_col], errors="coerce")
        work = work.sort_values(sort_col, ascending=False, na_position="last")
    names: list[str] = []
    seen: set[str] = set()
    for raw in work[name_col].tolist():
        s = str(raw).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        names.append(s)
        if len(names) >= n:
            break
    return names


def _parse_cons_codes(cons_df: pd.DataFrame) -> list[str]:
    if cons_df is None or cons_df.empty:
        return []
    code_col = "代码"
    if code_col not in cons_df.columns:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in cons_df[code_col].tolist():
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        c = str(raw).strip().zfill(6)
        if len(c) != 6 or not c.isdigit() or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def fetch_board_constituents_em(
    board_name: str,
    *,
    sleep_sec: float = DEFAULT_BOARD_SLEEP,
) -> list[str]:
    import akshare as ak

    ensure_eastmoney_no_proxy_if_configured()
    name = str(board_name).strip()
    if not name:
        return []
    try:
        cons = _with_retries(ak.stock_board_concept_cons_em, symbol=name)
    except Exception:
        time.sleep(sleep_sec)
        return []
    time.sleep(sleep_sec)
    return _parse_cons_codes(cons)


def sync_constituents_for_boards(
    board_names: list[str],
    *,
    sleep_sec: float = DEFAULT_BOARD_SLEEP,
    replace: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """拉取指定概念板块成份股并写入 board_stocks.json（默认整板覆盖）。"""
    names = [str(x).strip() for x in board_names if str(x).strip()]
    if not names:
        return {"boards": 0, "stocks": 0, "failed": 0}

    mapping: dict[str, list[str]] = {}
    failed = 0
    for i, board in enumerate(names, 1):
        codes = fetch_board_constituents_em(board, sleep_sec=sleep_sec)
        if not codes:
            failed += 1
            if verbose:
                print(f"[ConceptSync] 跳过（无成份）: {board}", flush=True)
            continue
        mapping[board] = codes
        if verbose and i % 5 == 0:
            print(
                f"[ConceptSync] 已拉取 {i}/{len(names)} 个板块…",
                flush=True,
            )

    if not mapping:
        return {"boards": 0, "stocks": 0, "failed": failed}

    save_board_mapping(mapping, merge=not replace)
    n_stocks = len(load_board_mapping())
    return {
        "boards": len(mapping),
        "stocks": n_stocks,
        "failed": failed,
    }


def ensure_hot_sectors_for_trade_date(
    trade_date: str | None = None,
    *,
    top_n: int = DEFAULT_HOT_TOP_N,
    sync_constituents: bool = True,
    force_refresh: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    若 ``hot_sectors.json`` 的 ``date`` 不是目标交易日则刷新热门题材；
    可选同步这些板块的成份股到 JSON + SQLite。
    """
    td = (trade_date or _latest_kline_date()).strip()[:10]
    meta = load_hot_sectors_meta()
    cached_date = str(meta.get("date", "")).strip()[:10]
    tags = load_tags()

    need_tags = force_refresh or cached_date != td or not tags
    stats: dict[str, Any] = {
        "trade_date": td,
        "tags_refreshed": False,
        "boards_synced": 0,
        "stock_mappings": len(load_board_mapping()),
        "tags": tags,
        "error": None,
    }

    if need_tags:
        try:
            fresh = fetch_hot_concept_names_em(top_n=top_n)
            if fresh:
                save_tags(
                    fresh,
                    metadata={
                        "trade_date": td,
                        "top_n": top_n,
                        "rank_by": "涨跌幅",
                    },
                )
                tags = fresh
                stats["tags_refreshed"] = True
                stats["tags"] = tags
                if verbose:
                    print(f"[ConceptSync] 已更新热门题材 {len(tags)} 个（{td}）", flush=True)
        except Exception as exc:
            stats["error"] = str(exc)
            if verbose:
                print(f"[ConceptSync] 拉取热门题材失败，沿用缓存: {exc}", flush=True)

    if sync_constituents and tags:
        boards_path = BOARD_STOCKS_PATH
        board_meta_date = ""
        if boards_path.exists():
            try:
                with open(boards_path, "r", encoding="utf-8") as f:
                    board_meta_date = str(json.load(f).get("date", "")).strip()[:10]
            except Exception:
                board_meta_date = ""
        need_boards = force_refresh or board_meta_date != td or stats["tags_refreshed"]
        if need_boards:
            try:
                bstats = sync_constituents_for_boards(
                    tags,
                    replace=True,
                    verbose=verbose,
                )
                stats["boards_synced"] = int(bstats.get("boards", 0))
                n = sync_concept_boards_from_json()
                stats["db_rows"] = n
                stats["stock_mappings"] = len(load_board_mapping())
            except Exception as exc:
                if stats["error"] is None:
                    stats["error"] = str(exc)
                if verbose:
                    print(f"[ConceptSync] 成份股同步失败: {exc}", flush=True)

    return stats
