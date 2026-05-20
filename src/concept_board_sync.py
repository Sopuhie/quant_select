"""
概念板块：热门题材列表 + 成份股 → JSON / SQLite。

默认数据源：同花顺概念资金流涨幅榜（``ths_fundflow``），按涨跌幅排序，网络稳定性优于东财 push2。

环境变量 ``QUANT_HOT_CONCEPT_SOURCE``：``ths_fundflow``（默认）| ``ths_list`` | ``em`` | ``auto``（同花顺优先，东财兜底）
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime
from functools import lru_cache
from io import StringIO
from typing import Any, Callable

import pandas as pd
import requests

from .board_stocks import BOARD_STOCKS_PATH, load_board_mapping, save_board_mapping
from .config import (
    DATA_DIR,
    DB_PATH,
    HOT_CONCEPT_BOARD_SLEEP_SEC,
    HOT_CONCEPT_CONS_MERGE,
    HOT_CONCEPT_SOURCE,
    HOT_CONCEPT_TOP_N,
    ensure_eastmoney_no_proxy_if_configured,
)
from .database import sync_concept_boards_from_json
from .hot_sectors import (
    HOT_SECTORS_PATH,
    build_default_tags,
    load_hot_sectors_meta,
    load_tags,
    save_tags,
)

DEFAULT_HOT_TOP_N = HOT_CONCEPT_TOP_N
DEFAULT_BOARD_SLEEP = HOT_CONCEPT_BOARD_SLEEP_SEC
_FETCH_RETRIES = 3
_FETCH_RETRY_SLEEP = 2.0

_EM_CLIST_HOSTS = (
    "https://82.push2.eastmoney.com",
    "https://79.push2.eastmoney.com",
    "https://7.push2.eastmoney.com",
    "https://80.push2.eastmoney.com",
    "https://push2.eastmoney.com",
)

_EM_CLIST_PARAMS = {
    "pn": "1",
    "pz": "100",
    "po": "1",
    "np": "1",
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": "2",
    "invt": "2",
    "fid": "f3",
    "fs": "m:90 t:3 f:!50",
    "fields": "f2,f3,f4,f8,f12,f14,f15,f16,f17,f18,f20,f21,f24,f25,f22,f33,f11,f62,f128,f124,f107,f104,f105,f136",
}


def _hot_concept_source() -> str:
    raw = str(HOT_CONCEPT_SOURCE or "ths_fundflow").strip().lower()
    if raw in ("em", "eastmoney", "东财"):
        return "em"
    if raw in ("ths", "ths_fundflow", "同花顺", "10jqka"):
        return "ths_fundflow"
    if raw in ("ths_list",):
        return "ths_list"
    return "auto"


def _with_retries(
    fn: Callable[..., Any],
    *args: Any,
    retries: int = _FETCH_RETRIES,
    **kwargs: Any,
) -> Any:
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


def _latest_kline_date(db_path: object | None = None) -> str:
    import sqlite3

    from .database import init_db

    path = db_path or DB_PATH
    init_db(path)  # type: ignore[arg-type]
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT MAX(date) FROM stock_daily_kline").fetchone()
        if row and row[0]:
            return str(row[0]).strip()[:10]
    finally:
        conn.close()
    return datetime.now().strftime("%Y-%m-%d")


def _ths_v_cookie() -> str:
    import py_mini_racer
    from akshare.datasets import get_ths_js

    js_code = py_mini_racer.MiniRacer()
    with open(get_ths_js("ths.js"), encoding="utf-8") as f:
        js_code.eval(f.read())
    return str(js_code.call("v"))


def _ths_request_headers(*, referer: str | None = None) -> dict[str, str]:
    v = _with_retries(_ths_v_cookie)
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Cookie": f"v={v}",
        "hexin-v": v,
    }
    if referer:
        h["Referer"] = referer
    return h


@lru_cache(maxsize=1)
def _ths_concept_name_code_map() -> dict[str, str]:
    """同花顺概念名 → 概念代码。"""
    import akshare as ak

    df = _with_retries(ak.stock_board_concept_name_ths)
    if df is None or df.empty:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        code = str(row.get("code", "")).strip()
        if name and code:
            out[name] = code
    return out


def _em_session_get(url: str, params: dict[str, str], *, timeout: float = 20.0) -> requests.Response:
    ensure_eastmoney_no_proxy_if_configured()
    last_exc: Exception | None = None
    for host in _EM_CLIST_HOSTS:
        try:
            with requests.Session() as session:
                adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                full_url = f"{host.rstrip('/')}/api/qt/clist/get"
                resp = session.get(full_url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp
        except Exception as exc:
            last_exc = exc
            time.sleep(0.3 + random.uniform(0, 0.2))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("eastmoney clist: all hosts failed")


def _fetch_concept_board_df_em_direct() -> pd.DataFrame:
    """直连东财 push2 概念板块列表（按涨跌幅排序）。"""
    params = dict(_EM_CLIST_PARAMS)
    r = _em_session_get("", params)
    data_json = r.json()
    diff = (data_json.get("data") or {}).get("diff") or []
    if not diff:
        return pd.DataFrame()
    first = pd.DataFrame(diff)
    per_page = max(1, len(first))
    total = int((data_json.get("data") or {}).get("total") or per_page)
    total_page = max(1, math.ceil(total / per_page))
    frames = [first]
    for page in range(2, total_page + 1):
        time.sleep(random.uniform(0.25, 0.6))
        p2 = dict(params)
        p2["pn"] = str(page)
        r2 = _em_session_get("", p2)
        j2 = r2.json()
        part = (j2.get("data") or {}).get("diff") or []
        if part:
            frames.append(pd.DataFrame(part))
    temp_df = pd.concat(frames, ignore_index=True)
    if "f3" in temp_df.columns:
        temp_df["f3"] = pd.to_numeric(temp_df["f3"], errors="coerce")
        temp_df = temp_df.sort_values("f3", ascending=False, na_position="last")
    if "f14" not in temp_df.columns:
        return pd.DataFrame()
    out = temp_df.rename(columns={"f14": "板块名称", "f3": "涨跌幅"})
    return out[["板块名称", "涨跌幅"]].reset_index(drop=True)


def fetch_hot_concept_names_em(top_n: int = DEFAULT_HOT_TOP_N) -> list[str]:
    """东方财富概念板块涨幅榜（直连 push2，失败则走 AkShare）。"""
    n = max(1, int(top_n))
    df: pd.DataFrame | None = None
    try:
        df = _fetch_concept_board_df_em_direct()
    except Exception:
        df = None
    if df is None or df.empty:
        import akshare as ak

        ensure_eastmoney_no_proxy_if_configured()
        raw = _with_retries(ak.stock_board_concept_name_em)
        if raw is None or raw.empty:
            return []
        name_col = "板块名称"
        if name_col not in raw.columns:
            raise RuntimeError(f"概念板块表缺少「{name_col}」，列: {list(raw.columns)}")
        sort_col = "涨跌幅"
        df = raw.copy()
        if sort_col in df.columns:
            df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
            df = df.sort_values(sort_col, ascending=False, na_position="last")
    names: list[str] = []
    seen: set[str] = set()
    for raw in df["板块名称"].tolist():
        s = str(raw).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        names.append(s)
        if len(names) >= n:
            break
    return names


def fetch_hot_concept_names_ths_fundflow(top_n: int = DEFAULT_HOT_TOP_N) -> list[str]:
    """
    同花顺数据中心 — 概念资金流（按涨跌幅 tradezdf 降序）。
    与东财「概念涨幅榜」语义接近，网络可达性更好。
    """
    from bs4 import BeautifulSoup

    n = max(1, int(top_n))
    v = _with_retries(_ths_v_cookie)
    headers = {
        "Accept": "text/html, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "hexin-v": v,
        "Host": "data.10jqka.com.cn",
        "Referer": "http://data.10jqka.com.cn/funds/gnzjl/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }
    base_url = (
        "http://data.10jqka.com.cn/funds/gnzjl/field/tradezdf/order/desc/page/{}/ajax/1/free/1/"
    )
    r0 = requests.get(base_url.format(1), headers=headers, timeout=20)
    r0.raise_for_status()
    soup = BeautifulSoup(r0.text, features="lxml")
    def _parse_page(html: str) -> pd.DataFrame | None:
        try:
            part = pd.read_html(StringIO(html))[0]
        except Exception:
            return None
        return part if part is not None and not part.empty else None

    page1 = _parse_page(r0.text)
    if page1 is None:
        raise RuntimeError("同花顺概念资金流：首页表格解析失败")

    page_info = soup.find(name="span", attrs={"class": "page_info"})
    page_num = 1
    if page_info is not None:
        try:
            page_num = max(1, int(str(page_info.text).split("/")[1].strip()))
        except Exception:
            page_num = 1

    frames: list[pd.DataFrame] = [page1]
    # 按涨跌幅已排序，通常首页即可凑满 top_n；最多再拉 2 页作缓冲
    max_pages = min(page_num, max(1, (n + 19) // 20) + 1, 3)
    for page in range(2, max_pages + 1):
        time.sleep(random.uniform(0.15, 0.35))
        v = _with_retries(_ths_v_cookie)
        headers["hexin-v"] = v
        rp = requests.get(base_url.format(page), headers=headers, timeout=20)
        rp.raise_for_status()
        part = _parse_page(rp.text)
        if part is not None:
            frames.append(part)

    big = pd.concat(frames, ignore_index=True)
    name_col = next((c for c in big.columns if str(c).strip() == "行业"), None)
    if name_col is None:
        name_col = next((c for c in big.columns if "行业" in str(c)), None)
    if name_col is None:
        raise RuntimeError(f"同花顺概念资金流缺少行业列: {list(big.columns)}")
    names: list[str] = []
    seen: set[str] = set()
    for raw in big[name_col].tolist():
        s = str(raw).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        names.append(s)
        if len(names) >= n:
            break
    return names


def fetch_hot_concept_names_ths_list(top_n: int = DEFAULT_HOT_TOP_N) -> list[str]:
    """同花顺概念板块全量列表（无涨幅排序，仅作最后兜底）。"""
    import akshare as ak

    n = max(1, int(top_n))
    df = _with_retries(ak.stock_board_concept_name_ths)
    if df is None or df.empty:
        return []
    names: list[str] = []
    for raw in df["name"].tolist():
        s = str(raw).strip()
        if s:
            names.append(s)
        if len(names) >= n:
            break
    return names


def fetch_hot_concept_names(
    top_n: int = DEFAULT_HOT_TOP_N,
) -> tuple[list[str], str, str | None]:
    """
  按配置拉取热门概念名称。

    Returns:
        (名称列表, 数据源标识, 末次错误信息)
    """
    n = max(1, int(top_n))
    mode = _hot_concept_source()
    chain: list[tuple[str, Callable[[int], list[str]]]] = []
    if mode == "em":
        chain = [("eastmoney", fetch_hot_concept_names_em)]
    elif mode == "ths_fundflow":
        chain = [("ths_fundflow", fetch_hot_concept_names_ths_fundflow)]
    elif mode == "ths_list":
        chain = [("ths_list", fetch_hot_concept_names_ths_list)]
    else:
        # auto：同花顺优先（稳定），东财仅作兜底
        chain = [
            ("ths_fundflow", fetch_hot_concept_names_ths_fundflow),
            ("ths_list", fetch_hot_concept_names_ths_list),
            ("eastmoney", fetch_hot_concept_names_em),
        ]

    errors: list[str] = []
    for source_key, fn in chain:
        try:
            names = fn(n)
            if names:
                return names, source_key, None
            errors.append(f"{source_key}: 返回空列表")
        except Exception as exc:
            errors.append(f"{source_key}: {exc}")
    return [], "", "; ".join(errors) if errors else "all sources empty"


def _parse_cons_codes(cons_df: pd.DataFrame) -> list[str]:
    if cons_df is None or cons_df.empty:
        return []
    code_col = next(
        (c for c in cons_df.columns if str(c).strip() in ("代码", "股票代码")),
        None,
    )
    if code_col is None:
        code_col = "代码" if "代码" in cons_df.columns else None
    if code_col is None:
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


def _resolve_ths_concept_code(board_name: str) -> str | None:
    name = str(board_name).strip()
    if not name:
        return None
    code_map = _ths_concept_name_code_map()
    code = code_map.get(name)
    if code:
        return str(code)
    for k, v in code_map.items():
        if name in k or k in name:
            return str(v)
    return None


def fetch_board_constituents_ths(
    board_name: str,
    *,
    sleep_sec: float = DEFAULT_BOARD_SLEEP,
) -> list[str]:
    """同花顺概念详情页成份股（首屏表格；分页接口常 403，以详情页为准）。"""
    code = _resolve_ths_concept_code(board_name)
    if not code:
        time.sleep(sleep_sec)
        return []

    detail_url = f"http://q.10jqka.com.cn/gn/detail/code/{code}/"
    headers = _ths_request_headers(referer=detail_url)
    seen: set[str] = set()
    out: list[str] = []

    try:
        sess = requests.Session()
        sess.get(detail_url, headers=headers, timeout=20)
        ajax_tpl = (
            "http://q.10jqka.com.cn/gn/detail/field/stocklist/{code}/"
            "order/desc/ajax/1/page/{page}/free/1/"
        )
        for page in range(1, 31):
            ajax_url = ajax_tpl.format(code=code, page=page)
            h = dict(headers)
            h["X-Requested-With"] = "XMLHttpRequest"
            h["Referer"] = detail_url
            r = sess.get(ajax_url, headers=h, timeout=20)
            if r.status_code != 200 or len(r.text) < 80:
                break
            try:
                part = _parse_cons_codes(pd.read_html(StringIO(r.text))[0])
            except Exception:
                break
            if not part:
                break
            for c in part:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            if len(part) < 8:
                break
            time.sleep(0.15)
    except Exception:
        pass

    if not out:
        try:
            r = requests.get(detail_url, headers=headers, timeout=20)
            r.raise_for_status()
            for tbl in pd.read_html(StringIO(r.text)):
                for c in _parse_cons_codes(tbl):
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
        except Exception:
            pass

    time.sleep(sleep_sec)
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


def fetch_board_constituents(
    board_name: str,
    *,
    sleep_sec: float = DEFAULT_BOARD_SLEEP,
) -> tuple[list[str], str]:
    """成份股：默认同花顺；``em`` 仅东财；``auto`` 同花顺优先、东财兜底。"""
    mode = _hot_concept_source()
    if mode == "em":
        codes = fetch_board_constituents_em(board_name, sleep_sec=sleep_sec)
        return (codes, "eastmoney") if codes else ([], "")

    codes = fetch_board_constituents_ths(board_name, sleep_sec=sleep_sec)
    if codes:
        return codes, "ths"
    if mode in ("ths_fundflow", "ths_list", "ths"):
        return [], ""

    codes = fetch_board_constituents_em(board_name, sleep_sec=sleep_sec)
    if codes:
        return codes, "eastmoney"
    return [], ""


def sync_constituents_for_boards(
    board_names: list[str],
    *,
    sleep_sec: float = DEFAULT_BOARD_SLEEP,
    replace: bool = False,
    verbose: bool = True,
    source_label: str | None = None,
) -> dict[str, int]:
    """拉取指定概念板块成份股并写入 board_stocks.json（默认 merge 合并，避免覆盖旧成份）。"""
    names = [str(x).strip() for x in board_names if str(x).strip()]
    if not names:
        return {"boards": 0, "stocks": 0, "failed": 0}

    mapping: dict[str, list[str]] = {}
    failed = 0
    partial = 0
    for i, board in enumerate(names, 1):
        codes, src = fetch_board_constituents(board, sleep_sec=sleep_sec)
        if not codes:
            failed += 1
            if verbose:
                print(f"[ConceptSync] 跳过（无成份）: {board}", flush=True)
            continue
        if src == "ths" and len(codes) < 15:
            partial += 1
        mapping[board] = codes
        if verbose and i % 5 == 0:
            print(
                f"[ConceptSync] 已拉取 {i}/{len(names)} 个板块…",
                flush=True,
            )

    if not mapping:
        return {"boards": 0, "stocks": 0, "failed": failed, "partial": partial}

    merge_boards = HOT_CONCEPT_CONS_MERGE if not replace else False
    src = (source_label or "").strip()
    if not src or src == "auto":
        mode = _hot_concept_source()
        src = "ths_fundflow" if mode in ("auto", "ths_fundflow", "ths", "ths_list") else mode
    save_board_mapping(mapping, merge=merge_boards, source=src)
    n_stocks = len(load_board_mapping())
    return {
        "boards": len(mapping),
        "stocks": n_stocks,
        "failed": failed,
        "partial": partial,
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
        "source": str(meta.get("source", "")),
    }

    if need_tags:
        try:
            fresh, source_key, fetch_err = fetch_hot_concept_names(top_n=top_n)
            if fresh:
                save_tags(
                    fresh,
                    metadata={
                        "trade_date": td,
                        "top_n": top_n,
                        "rank_by": "涨跌幅",
                        "source": source_key,
                    },
                )
                tags = fresh
                stats["tags_refreshed"] = True
                stats["tags"] = tags
                stats["source"] = source_key
                if verbose:
                    print(
                        f"[ConceptSync] 已更新热门题材 {len(tags)} 个（{td}，来源 {source_key}）",
                        flush=True,
                    )
            elif fetch_err:
                stats["error"] = fetch_err
                if verbose:
                    print(
                        f"[ConceptSync] 拉取热门题材失败，沿用缓存: {fetch_err}",
                        flush=True,
                    )
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
                    replace=not HOT_CONCEPT_CONS_MERGE,
                    verbose=verbose,
                    source_label=str(stats.get("source") or ""),
                )
                stats["boards_synced"] = int(bstats.get("boards", 0))
                stats["boards_partial"] = int(bstats.get("partial", 0))
                n = sync_concept_boards_from_json()
                stats["db_rows"] = n
                stats["stock_mappings"] = len(load_board_mapping())
            except Exception as exc:
                if stats["error"] is None:
                    stats["error"] = str(exc)
                if verbose:
                    print(f"[ConceptSync] 成份股同步失败: {exc}", flush=True)

    return stats
