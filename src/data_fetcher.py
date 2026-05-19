"""A 股日线数据获取（AkShare 优先，失败时可 Baostock 兜底）。"""
from __future__ import annotations

import atexit
import logging
import os
import random
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
import numpy as np
import pandas as pd

from .config import (
    AKSHARE_FETCH_RETRIES,
    AKSHARE_FETCH_RETRY_SLEEP,
    AKSHARE_REQUEST_TIMEOUT,
    BAOSTOCK_FIRST_IF_RANGE_DAYS,
    MIN_HISTORY_BARS,
    PREDICT_HISTORY_CALENDAR_DAYS,
    STOCK_POOL,
    USE_BAOSTOCK_FALLBACK,
    ensure_eastmoney_no_proxy_if_configured,
)

_BS_LOCK = threading.Lock()
_BS_LOGGED_IN = False
_BS_ATEXIT_REGISTERED = False

_log = logging.getLogger(__name__)


def _is_transient_network_failure(exc: BaseException | None) -> bool:
    """AkShare 常见瞬时失败：连接被重置、代理、超时等，适合重试或换 Baostock。"""
    if exc is None:
        return False
    try:
        import requests.exceptions as rq_exc

        if isinstance(
            exc,
            (
                rq_exc.ConnectionError,
                rq_exc.Timeout,
                rq_exc.ProxyError,
                rq_exc.ChunkedEncodingError,
                rq_exc.SSLError,
            ),
        ):
            return True
    except Exception:
        pass
    chain: BaseException | None = exc
    seen = 0
    while chain is not None and seen < 12:
        seen += 1
        name = type(chain).__name__
        if name in (
            "RemoteDisconnected",
            "ProtocolError",
            "ConnectionResetError",
            "BrokenPipeError",
        ):
            return True
        chain = chain.__cause__ or chain.__context__
    return False


def _baostock_logout() -> None:
    global _BS_LOGGED_IN
    try:
        import baostock as bs

        with _BS_LOCK:
            if _BS_LOGGED_IN:
                bs.logout()
                _BS_LOGGED_IN = False
    except Exception:
        pass


def _compact_to_iso_date(compact: str) -> str:
    s = str(compact).replace("-", "").strip()
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return compact


def _compact_yyyymmdd(raw: str) -> str:
    """将 start/end 参数规范为 8 位 YYYYMMDD（非法时返回原串去横线的前 8 位）。"""
    s = str(raw).replace("-", "").strip()
    return s[:8] if len(s) >= 8 else s.ljust(8, "0")[:8]


def _calendar_span_days(start_date: str, end_date: str) -> int:
    """自然日跨度（含首尾），解析失败返回大数表示「非短区间」。"""
    try:
        a = _compact_yyyymmdd(start_date)
        b = _compact_yyyymmdd(end_date)
        if not (a.isdigit() and b.isdigit() and len(a) == 8 and len(b) == 8):
            return 9999
        da = datetime.strptime(a, "%Y%m%d").date()
        db = datetime.strptime(b, "%Y%m%d").date()
        return max(0, (db - da).days)
    except Exception:
        return 9999


def _fetch_daily_hist_baostock(
    symbol: str,
    start_compact: str,
    end_compact: str,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Baostock 日线，列与 AkShare 处理后一致。adjustflag=2 为前复权，对齐 AkShare qfq。
    全程在锁内执行 query，避免多线程并发访问 SDK。

    ``force=True`` 时在未设置 ``QUANT_BAOSTOCK_FALLBACK`` 下仍尝试（用于 AkShare 连接类失败后的自动兜底）。
    """
    if not USE_BAOSTOCK_FALLBACK and not force:
        return pd.DataFrame()

    code = str(symbol).strip().zfill(6)
    if len(code) != 6 or not code.isdigit():
        return pd.DataFrame()

    sec = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
    sd = _compact_to_iso_date(start_compact)
    ed = _compact_to_iso_date(end_compact)

    try:
        import baostock as bs
    except ImportError:
        return pd.DataFrame()

    data_list: list[list[str]] = []
    with _BS_LOCK:
        global _BS_LOGGED_IN, _BS_ATEXIT_REGISTERED
        if not _BS_LOGGED_IN:
            lg = bs.login()
            if lg.error_code != "0":
                return pd.DataFrame()
            _BS_LOGGED_IN = True
            if not _BS_ATEXIT_REGISTERED:
                atexit.register(_baostock_logout)
                _BS_ATEXIT_REGISTERED = True
        try:
            rs = bs.query_history_k_data_plus(
                sec,
                "date,open,high,low,close,volume",
                start_date=sd,
                end_date=ed,
                frequency="d",
                adjustflag="2",
            )
            while rs.error_code == "0" and rs.next():
                data_list.append(rs.get_row_data())
        except Exception:
            return pd.DataFrame()

    if not data_list:
        return pd.DataFrame()

    out = pd.DataFrame(
        data_list, columns=["date", "open", "high", "low", "close", "volume"]
    )
    return out


def _finalize_hist_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一数值列与日期格式；保留 ``turnover_rate``（换手率 %）若存在。"""
    if df.empty:
        return df
    need = {"date", "open", "high", "low", "close", "volume"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "turnover_rate" in df.columns:
        df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

_POOL_CACHE: dict[str, tuple[float, list[tuple[str, str]]]] = {}
_POOL_TTL_SEC = 3600.0
# 成分列解析逻辑变更时递增，避免 TTL 内继续使用错误的缓存池
_POOL_CACHE_KEY_VER = "2"

# 风险警示板（ST）代码缓存，避免每次选股全量请求
_ST_BOARD_CACHE: tuple[float, frozenset[str]] | None = None
_ST_BOARD_TTL_SEC = 1800.0


def get_risk_st_stock_codes() -> frozenset[str]:
    """
    东方财富风险警示板当日列表中的 6 位代码集合（与名称规则互补，避免漏网）。
    """
    global _ST_BOARD_CACHE
    now = time.time()
    if _ST_BOARD_CACHE is not None and now - _ST_BOARD_CACHE[0] < _ST_BOARD_TTL_SEC:
        return _ST_BOARD_CACHE[1]
    codes: set[str] = set()
    try:
        ensure_eastmoney_no_proxy_if_configured()
        df = ak.stock_zh_a_st_em()
        if df is not None and not df.empty:
            col = "代码" if "代码" in df.columns else "code"
            for _, row in df.iterrows():
                c = str(row[col]).strip().zfill(6)
                if len(c) == 6 and c.isdigit():
                    codes.add(c)
    except Exception:
        pass
    fs = frozenset(codes)
    _ST_BOARD_CACHE = (now, fs)
    return fs


def compact_start_calendar_days_ago(days: int | None = None) -> str:
    """用于选股：只拉最近一段 K 线，加快请求。"""
    d = int(days if days is not None else PREDICT_HISTORY_CALENDAR_DAYS)
    return (datetime.now().date() - timedelta(days=d)).strftime("%Y%m%d")


def get_hs300_stocks(as_of_date: str | None = None) -> list[tuple[str, str]]:
    """沪深300 成分，as_of_date 为 YYYY-MM-DD 时取该日及以前最近一期成份。"""
    return _get_index_constituents("000300", as_of_date)


def get_zz500_stocks(as_of_date: str | None = None) -> list[tuple[str, str]]:
    """中证500 成份。"""
    return _get_index_constituents("000905", as_of_date)


def _is_index_only_metadata_col(col_name: object) -> bool:
    """排除「指数代码/指数名称」等列，避免误当成成分股代码列。"""
    s = str(col_name)
    return "指数" in s and "成分" not in s and "成份" not in s


def _pick_index_constituent_code_col(cols: list) -> str | None:
    """优先匹配成分券代码列，绝不能选用「指数代码」列（否则全体变成 000300/000905 一行）。"""
    for c in cols:
        if _is_index_only_metadata_col(c):
            continue
        sc = str(c)
        if "券代码" in sc:
            return c
        if ("成分" in sc or "成份" in sc) and (
            "代码" in sc or "code" in sc.lower()
        ):
            return c
    for c in cols:
        if _is_index_only_metadata_col(c):
            continue
        sc = str(c)
        if "代码" in sc or "code" in sc.lower():
            return c
    return None


def _pick_index_constituent_name_col(cols: list) -> str | None:
    for c in cols:
        if _is_index_only_metadata_col(c):
            continue
        sc = str(c)
        if ("成分" in sc or "成份" in sc) and (
            "名称" in sc or "name" in sc.lower()
        ):
            return c
    for c in cols:
        if _is_index_only_metadata_col(c):
            continue
        sc = str(c)
        if "名称" in sc or "name" in sc.lower():
            return c
    return None


def _extract_stock_code6(raw: object) -> str | None:
    """从 '600000'、'600000.SH'、'sz.000001' 等解析 6 位 A 股代码。"""
    s = str(raw).strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 6:
        return digits[-6:]
    if digits.isdigit() and 1 <= len(digits) <= 6:
        return digits.zfill(6)
    return None


def _get_index_constituents(index_code: str, as_of_date: str | None) -> list[tuple[str, str]]:
    df = None
    # 尝试中证官方接口
    try:
        df = ak.index_stock_cons_csindex(symbol=index_code)
    except Exception:
        df = None

    # 如果官方接口挂了或为空，用新浪接口作为第一级备用方案
    if df is None or df.empty:
        try:
            symbol_map = {"000300": "hs300", "000905": "zz500"}
            if index_code in symbol_map:
                df = ak.index_stock_cons(symbol=symbol_map[index_code])
        except Exception:
            df = None

    # 如果依然为空，用东方财富成分股接口做终极兜底
    if df is None or df.empty:
        try:
            symbol_em_map = {"000300": "000300.SH", "000905": "000905.SH"}
            if index_code in symbol_em_map:
                df = ak.index_stock_cons_em(symbol=index_code)
        except Exception:
            df = None

    if df is None or df.empty:
        return []

    # 规范化解析字段（须与「指数代码」列区分，否则会只剩指数代码一条）
    cols = list(df.columns)
    dcol = next((c for c in cols if "日期" in str(c) or "date" in str(c).lower()), None)
    ccol = _pick_index_constituent_code_col(cols)
    ncol = _pick_index_constituent_name_col(cols)

    if ccol is None:
        return []

    df = df.copy()
    if dcol and dcol in df.columns:
        df["_d"] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=["_d"])
        if as_of_date:
            cut = pd.Timestamp(str(as_of_date)[:10])
            df = df[df["_d"] <= cut]
        if df.empty:
            # 如果按日期过滤空了，不退避，直接取最新的历史成份，保证回测能跑
            df = pd.DataFrame()
            try:
                df = ak.index_stock_cons_csindex(symbol=index_code)
                df["_d"] = pd.to_datetime(df[dcol], errors="coerce")
            except Exception:
                pass
        if not df.empty:
            latest = df["_d"].max()
            df = df[df["_d"] == latest]

    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        c = _extract_stock_code6(row[ccol])
        if c is None:
            continue
        n = str(row[ncol]).strip() if ncol is not None else ""
        pairs.append((c, n))

    return sorted(set(pairs), key=lambda x: x[0])


def get_stock_pool(
    as_of_date: str | None = None,
    pool_type: str | None = None,
    max_count: int | None = None,
) -> list[tuple[str, str]]:
    """
    按配置返回 (code6, name) 列表。
    pool_type: all | hs300 | zz500；默认读取 config.STOCK_POOL。
    """
    pt = (pool_type or STOCK_POOL).lower().strip()
    cache_key = f"{_POOL_CACHE_KEY_VER}|{pt}|{as_of_date}|{max_count}"
    now = time.time()
    hit = _POOL_CACHE.get(cache_key)
    if hit and now - hit[0] < _POOL_TTL_SEC:
        return list(hit[1])

    if pt == "all":
        out = list_a_stock_codes(max_count=max_count)
    elif pt == "hs300":
        out = get_hs300_stocks(as_of_date)
        if max_count is not None:
            out = out[: int(max_count)]
    elif pt == "zz500":
        out = get_zz500_stocks(as_of_date)
        if max_count is not None:
            out = out[: int(max_count)]
    else:
        out = list_a_stock_codes(max_count=max_count)

    # 终极兜底：如果前面全部获取失败，直接用 A 股全市场股票列表前max_count个股票塞满，防止回测彻底崩溃
    if not out:
        print("⚠️ 警告: 无法获取特定的指数成分股。已切换至全市场股票前 100 只作为兜底股票池进行回测...")
        out = list_a_stock_codes(max_count=100)
    elif pt in ("hs300", "zz500") and len(out) < 50:
        # 成分列误解析时可能只剩指数代码 1 条，此处强制回退全市场，避免回测只拉一只「伪成分」
        print(
            f"⚠️ 警告: {pt.upper()} 成分数量异常（仅 {len(out)} 只），疑似接口字段解析错误。"
            "已改用全市场股票列表兜底。"
        )
        cap = max(int(max_count), 100) if max_count is not None else 100
        out = list_a_stock_codes(max_count=cap)

    _POOL_CACHE[cache_key] = (now, list(out))
    return out


def _bs_code_is_listed_a_share(bs_code: str) -> bool:
    """Baostock 证券代码（如 sh.600000）是否为沪深北 A 股普通股票（排除指数等）。"""
    s = str(bs_code).strip().lower()
    if "." not in s:
        return False
    mkt, num = s.split(".", 1)
    if len(num) != 6 or not num.isdigit():
        return False
    if mkt == "sh":
        return num.startswith(("60", "688", "689"))
    if mkt == "sz":
        return num.startswith(("000", "001", "002", "003", "300", "301"))
    if mkt == "bj":
        return num.startswith(("43", "82", "83", "87", "88", "92"))
    return False


def _pairs_from_akshare_stock_info_df(
    df: pd.DataFrame, max_count: int | None
) -> list[tuple[str, str]]:
    if df is None or df.empty:
        return []
    cols = list(df.columns)
    code_col = next(
        (c for c in cols if str(c).lower() == "code" or "代码" in str(c)),
        cols[0],
    )
    name_col = next(
        (c for c in cols if str(c).lower() == "name" or "名称" in str(c)),
        cols[1] if len(cols) > 1 else cols[0],
    )
    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        c = str(row[code_col]).strip().zfill(6)
        n = str(row[name_col]).strip()
        if len(c) == 6 and c.isdigit():
            pairs.append((c, n))
        if max_count is not None and len(pairs) >= int(max_count):
            break
    return pairs


def _list_a_stock_codes_via_baostock(max_count: int | None) -> list[tuple[str, str]]:
    """
    AkShare ``stock_info_a_code_name`` 依赖证交所页面，易被断开；用 Baostock 全量证券表兜底。
    """
    try:
        import baostock as bs
    except ImportError:
        return []

    pairs_map: dict[str, str] = {}
    with _BS_LOCK:
        global _BS_LOGGED_IN, _BS_ATEXIT_REGISTERED
        if not _BS_LOGGED_IN:
            lg = bs.login()
            if lg.error_code != "0":
                return []
            _BS_LOGGED_IN = True
            if not _BS_ATEXIT_REGISTERED:
                atexit.register(_baostock_logout)
                _BS_ATEXIT_REGISTERED = True
        try:
            for day_offset in range(0, 28):
                d = (datetime.now().date() - timedelta(days=day_offset)).strftime(
                    "%Y-%m-%d"
                )
                rs = bs.query_all_stock(day=d)
                if rs.error_code != "0":
                    continue
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if len(row) < 3:
                        continue
                    code_raw, trade_status, name = row[0], row[1], row[2]
                    if str(trade_status).strip() != "1":
                        continue
                    code_raw_s = str(code_raw).strip()
                    if not _bs_code_is_listed_a_share(code_raw_s):
                        continue
                    _, num = code_raw_s.lower().split(".", 1)
                    code6 = num.zfill(6)
                    name_s = str(name).strip() if name else ""
                    if len(code6) == 6 and code6.isdigit():
                        pairs_map.setdefault(code6, name_s)
                if pairs_map:
                    break
        except Exception:
            return []

    out = sorted(pairs_map.items(), key=lambda x: x[0])
    if max_count is not None:
        out = out[: int(max_count)]
    return list(out)


def list_a_stock_codes(max_count: int | None = None) -> list[tuple[str, str]]:
    """
    返回 (code6, name) 列表。code6 为 6 位数字，不含市场前缀。

    优先 AkShare ``stock_info_a_code_name``（带重试）；连接被重置等失败时改用 Baostock
    ``query_all_stock``，减轻对深交所列表页的依赖。
    """
    ensure_eastmoney_no_proxy_if_configured()
    retries = max(1, int(AKSHARE_FETCH_RETRIES))
    sleep_base = float(AKSHARE_FETCH_RETRY_SLEEP)
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            df = ak.stock_info_a_code_name()
            pairs = _pairs_from_akshare_stock_info_df(df, max_count)
            if pairs:
                return pairs
        except Exception as exc:
            last_exc = exc
            transient = _is_transient_network_failure(exc)
            if attempt + 1 < retries and transient:
                delay = sleep_base * (2**attempt) + random.uniform(0, 0.6)
                _log.warning(
                    "stock_info_a_code_name 失败 (%s)，%s 后重试 (%s/%s)",
                    type(exc).__name__,
                    f"{delay:.1f}s",
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)
                continue
            if attempt + 1 < retries:
                delay = sleep_base * (2**attempt) + random.uniform(0, 0.6)
                time.sleep(delay)
                continue

    fallback = _list_a_stock_codes_via_baostock(max_count=max_count)
    if fallback:
        if last_exc is not None:
            _log.warning(
                "已改用 Baostock 证券列表作为 A 股代码表（AkShare 末次错误: %s）",
                last_exc,
            )
        return fallback
    if last_exc is not None:
        _log.error("list_a_stock_codes 失败且无 Baostock 兜底: %s", last_exc)
    return []


def _normalize_baostock_industry_label(raw: object) -> str:
    """Baostock industry 常为「J66货币金融服务」，去掉前置字母数字行业编码。"""
    s = str(raw or "").strip()
    if not s:
        return ""
    cleaned = re.sub(r"^[A-Za-z]+\d+", "", s).strip()
    return cleaned or s


def _fetch_industry_map_via_baostock() -> dict[str, str]:
    """Baostock ``query_stock_industry`` 一次返回全市场，不依赖东方财富。"""
    try:
        import baostock as bs
    except ImportError:
        return {}

    code_to_industry: dict[str, str] = {}
    with _BS_LOCK:
        global _BS_LOGGED_IN, _BS_ATEXIT_REGISTERED
        if not _BS_LOGGED_IN:
            lg = bs.login()
            if lg.error_code != "0":
                return []
            _BS_LOGGED_IN = True
            if not _BS_ATEXIT_REGISTERED:
                atexit.register(_baostock_logout)
                _BS_ATEXIT_REGISTERED = True
        try:
            rs = bs.query_stock_industry()
            if rs.error_code != "0":
                return {}
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                if len(row) < 4:
                    continue
                code6 = _extract_stock_code6(row[1])
                if code6 is None:
                    continue
                label = _normalize_baostock_industry_label(row[3])
                if not label:
                    continue
                if code6 not in code_to_industry:
                    code_to_industry[code6] = label
        except Exception:
            return {}
    return code_to_industry


def fetch_industry_map_via_em_boards(
    *,
    board_sleep_sec: float = 0.2,
    max_boards: int | None = None,
    verbose: bool = False,
) -> dict[str, str]:
    """东方财富行业板块—成份映射（AkShare ``stock_board_industry_*_em``）。"""
    ensure_eastmoney_no_proxy_if_configured()
    code_to_industry: dict[str, str] = {}
    boards_ok = 0
    boards_fail = 0
    n_loop = 0

    try:
        boards_df = ak.stock_board_industry_name_em()
    except Exception as exc:
        raise RuntimeError(
            f"拉取东方财富行业板块列表失败: {exc}"
        ) from exc

    if boards_df is None or boards_df.empty:
        raise RuntimeError("东方财富行业板块列表为空")

    name_col = "板块名称"
    if name_col not in boards_df.columns:
        raise RuntimeError(
            f"行业板块表缺少列「{name_col}」，当前列: {list(boards_df.columns)}"
        )

    for _, row in boards_df.iterrows():
        if max_boards is not None and n_loop >= max_boards:
            break
        n_loop += 1
        board_name = str(row[name_col]).strip()
        if not board_name:
            continue
        try:
            cons = ak.stock_board_industry_cons_em(symbol=board_name)
        except Exception:
            boards_fail += 1
            time.sleep(board_sleep_sec)
            continue
        if cons is None or cons.empty or "代码" not in cons.columns:
            boards_fail += 1
            time.sleep(board_sleep_sec)
            continue
        boards_ok += 1
        for _, crow in cons.iterrows():
            raw = crow.get("代码")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            c = str(raw).strip().zfill(6)
            if len(c) != 6 or not c.isdigit():
                continue
            if c not in code_to_industry:
                code_to_industry[c] = board_name
        if verbose and boards_ok % 15 == 0:
            print(
                f"  [行业/东财] 板块 {n_loop}，成功 {boards_ok}，"
                f"失败 {boards_fail}，已映射 {len(code_to_industry)} 只…",
                flush=True,
            )
        time.sleep(board_sleep_sec)

    if not code_to_industry:
        raise RuntimeError(
            f"东方财富行业成份解析为空（板块成功 {boards_ok}，失败 {boards_fail}）"
        )
    return code_to_industry


def fetch_a_share_industry_map(
    *,
    source: str | None = None,
    board_sleep_sec: float = 0.2,
    max_boards: int | None = None,
    verbose: bool = False,
) -> dict[str, str]:
    """
    A 股代码→行业名称。``source`` / ``QUANT_INDUSTRY_SOURCE``：

    - ``auto``（默认）：东方财富板块 → Baostock
    - ``em``：仅东方财富
    - ``baostock``：仅 Baostock（推荐在东财被限流/断开时使用）
    """
    src = (source or os.environ.get("QUANT_INDUSTRY_SOURCE", "auto")).strip().lower()

    if src == "baostock":
        m = _fetch_industry_map_via_baostock()
        if not m:
            raise RuntimeError("Baostock query_stock_industry 未返回有效行业数据")
        if verbose:
            print(f"行业映射来源: Baostock（{len(m)} 只）", flush=True)
        return m

    if src == "em":
        m = fetch_industry_map_via_em_boards(
            board_sleep_sec=board_sleep_sec,
            max_boards=max_boards,
            verbose=verbose,
        )
        if verbose:
            print(f"行业映射来源: 东方财富行业板块（{len(m)} 只）", flush=True)
        return m

    # auto
    em_err: BaseException | None = None
    try:
        m = fetch_industry_map_via_em_boards(
            board_sleep_sec=board_sleep_sec,
            max_boards=max_boards,
            verbose=verbose,
        )
        if verbose:
            print(f"行业映射来源: 东方财富行业板块（{len(m)} 只）", flush=True)
        return m
    except Exception as exc:
        em_err = exc
        if verbose:
            print(
                f"东方财富行业板块不可用（{exc}），改用 Baostock…",
                flush=True,
            )

    m = _fetch_industry_map_via_baostock()
    if m:
        if verbose:
            print(f"行业映射来源: Baostock（{len(m)} 只）", flush=True)
        return m

    raise RuntimeError(
        "行业同步失败：东方财富与 Baostock 均不可用。"
        + (f" 东财末次错误: {em_err}" if em_err else "")
    )


def fetch_single_market_cap_yuan(code6: str) -> float | None:
    """
    单只股票总市值（元）。``auto`` 时依次尝试：

    1. 东财 push2 ``stock_individual_info_em``
    2. 东财 datacenter ``stock_value_em``（push2 被断开时常仍可用）
    """
    import numpy as np

    ensure_eastmoney_no_proxy_if_configured()
    c = str(code6).strip().zfill(6)
    src = os.environ.get("QUANT_MCAP_SOURCE", "auto").strip().lower()

    def _from_push2() -> float | None:
        df = ak.stock_individual_info_em(symbol=c)
        mc_col = df[df["item"] == "总市值"]
        if mc_col.empty:
            return None
        mc = float(mc_col.iloc[0]["value"])
        if np.isfinite(mc) and mc > 0:
            return mc
        return None

    def _from_value_em() -> float | None:
        df = ak.stock_value_em(symbol=c)
        if df is None or df.empty or "总市值" not in df.columns:
            return None
        mc = float(df.iloc[-1]["总市值"])
        if np.isfinite(mc) and mc > 0:
            return mc
        return None

    if src in ("em", "em_push2", "push2"):
        try:
            return _from_push2()
        except Exception:
            return None
    if src in ("em_value", "value", "datacenter"):
        try:
            return _from_value_em()
        except Exception:
            return None

    # auto
    for fn in (_from_push2, _from_value_em):
        for attempt in range(2):
            try:
                mc = fn()
                if mc is not None:
                    return mc
            except Exception:
                if attempt == 0:
                    time.sleep(0.4 * (attempt + 1))
    return None


def fetch_market_cap_map(
    codes: list[str],
    *,
    max_workers: int = 8,
    sleep_sec: float | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """并发拉取多只股票总市值（元）。"""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if sleep_sec is None:
        try:
            sleep_sec = float(os.environ.get("QUANT_MCAP_SYNC_SLEEP", "0.35"))
        except ValueError:
            sleep_sec = 0.35

    uniq = sorted({str(c).strip().zfill(6) for c in codes if c})
    code_to_mcap: dict[str, float] = {}
    lock = threading.Lock()
    total = len(uniq)

    def _one(code: str) -> None:
        mc = fetch_single_market_cap_yuan(code)
        if mc is not None:
            with lock:
                code_to_mcap[code] = mc
        if sleep_sec and sleep_sec > 0:
            time.sleep(float(sleep_sec))

    workers_n = max(1, min(int(max_workers), 16))
    done = 0
    with ThreadPoolExecutor(max_workers=workers_n) as ex:
        futs = [ex.submit(_one, c) for c in uniq]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if verbose and done % 200 == 0:
                print(
                    f"  [市值] {done}/{total} 已完成，成功 {len(code_to_mcap)} 只…",
                    flush=True,
                )
    return code_to_mcap


def fetch_daily_hist(
    symbol: str,
    start_date: str = "20180101",
    end_date: Optional[str] = None,
    adjust: str = "qfq",
    timeout: Optional[float] = None,
    max_retries: int | None = None,
) -> pd.DataFrame:
    """
    获取单只股票日线。列：date, open, high, low, close, volume（统一小写）。
    网络失败或超时时返回空表，不向外抛 requests 异常。
    内置有限次重试与指数退避；短自然日区间内优先 Baostock，减轻东财限流。
    AkShare 仍失败或最终无数据时写入 ``logging`` 警告（含末次异常）。
    若 AkShare 因连接/超时等瞬时错误失败，即使未设置 ``QUANT_BAOSTOCK_FALLBACK``，也会尝试 Baostock（已安装时）。
    """
    if end_date is None:
        end_date = datetime_today_compact()
    ensure_eastmoney_no_proxy_if_configured()
    to = float(timeout if timeout is not None else AKSHARE_REQUEST_TIMEOUT)
    retries = int(max_retries if max_retries is not None else AKSHARE_FETCH_RETRIES)
    retries = max(1, retries)
    sleep_base = AKSHARE_FETCH_RETRY_SLEEP

    start_c = start_date.replace("-", "") if "-" in str(start_date) else str(start_date)
    end_c = end_date.replace("-", "") if "-" in str(end_date) else str(end_date)
    span_days = _calendar_span_days(start_c, end_c)
    short_range = (
        BAOSTOCK_FIRST_IF_RANGE_DAYS > 0
        and span_days <= BAOSTOCK_FIRST_IF_RANGE_DAYS
    )
    # 短区间：东财常限流，AkShare 少试几次；长区间：保持配置次数
    ak_attempts = min(retries, 2) if short_range else retries
    delay_cap = 4.0 if short_range else 20.0

    df: pd.DataFrame | None = None
    last_exc: BaseException | None = None

    if short_range:
        df_bs0 = _fetch_daily_hist_baostock(
            symbol,
            start_c,
            end_c,
            force=True,
        )
        if df_bs0 is not None and not df_bs0.empty:
            df = df_bs0
            last_exc = None

    if df is None or df.empty:
        for attempt in range(ak_attempts):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_c,
                    end_date=end_c,
                    adjust=adjust,
                    timeout=to,
                )
            except Exception as e:
                df = None
                last_exc = e
            if df is not None and not df.empty:
                last_exc = None
                break
            if _is_transient_network_failure(last_exc):
                df_bs = _fetch_daily_hist_baostock(
                    symbol,
                    start_c,
                    end_c,
                    force=True,
                )
                if df_bs is not None and not df_bs.empty:
                    df = df_bs
                    last_exc = None
                    break
            if attempt < ak_attempts - 1:
                mult = 2 ** min(attempt, 3)
                delay = min(delay_cap, sleep_base * mult) + random.random() * 0.4
                time.sleep(delay)

    if df is None or df.empty:
        sym = str(symbol).strip().zfill(6)
        sd_c = str(start_c).strip()[:8]
        ed_c = str(end_c).strip()[:8] if end_c else ""
        if last_exc is not None:
            _log.warning(
                "AkShare 日线失败 symbol=%s start=%s end=%s retries=%s 末次异常: %r",
                sym,
                sd_c,
                ed_c,
                ak_attempts,
                last_exc,
            )
        else:
            _log.warning(
                "AkShare 日线无有效数据 symbol=%s start=%s end=%s retries=%s（未抛异常，可能为空表或该区间无行情）",
                sym,
                sd_c,
                ed_c,
                ak_attempts,
            )
        df_bs = _fetch_daily_hist_baostock(
            symbol,
            start_c,
            end_c,
            force=USE_BAOSTOCK_FALLBACK or _is_transient_network_failure(last_exc),
        )
        if df_bs is not None and not df_bs.empty:
            df = df_bs
            if not USE_BAOSTOCK_FALLBACK and _is_transient_network_failure(last_exc):
                _log.info(
                    "AkShare 失败后已用 Baostock 自动兜底 symbol=%s start=%s end=%s",
                    sym,
                    sd_c,
                    ed_c,
                )
        else:
            if USE_BAOSTOCK_FALLBACK or _is_transient_network_failure(last_exc):
                _log.warning(
                    "日线最终无数据 symbol=%s start=%s end=%s（AkShare 末次异常=%r，Baostock 兜底亦空）",
                    sym,
                    sd_c,
                    ed_c,
                    last_exc,
                )
            else:
                _log.warning(
                    "日线最终无数据 symbol=%s start=%s end=%s（AkShare 末次异常=%r，未启用 Baostock 兜底）",
                    sym,
                    sd_c,
                    ed_c,
                    last_exc,
                )
            return pd.DataFrame()

    cols_now = set(df.columns)
    need = {"date", "open", "high", "low", "close", "volume"}
    if not need.issubset(cols_now):
        rename = {}
        for c in df.columns:
            lc = str(c).lower()
            if "日期" in str(c) or lc == "date":
                rename[c] = "date"
            elif "开盘" in str(c) or lc == "open":
                rename[c] = "open"
            elif "收盘" in str(c) or lc == "close":
                rename[c] = "close"
            elif "最高" in str(c) or lc == "high":
                rename[c] = "high"
            elif "最低" in str(c) or lc == "low":
                rename[c] = "low"
            elif "成交量" in str(c) or "volume" in lc:
                rename[c] = "volume"
            elif "换手" in str(c):
                rename[c] = "turnover_rate"
        df = df.rename(columns=rename)

    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    return _finalize_hist_columns(df)


def fetch_spot_pe_turnover_map() -> dict[str, tuple[float | None, float | None]]:
    """
    东财 A 股实时快照：``(pe_ttm, turnover_rate)`` 按 6 位代码。
    用于增量同步最新交易日估值字段；失败返回空 dict。
    """
    ensure_eastmoney_no_proxy_if_configured()
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as exc:
        _log.warning("stock_zh_a_spot_em 失败: %r", exc)
        return {}
    if df is None or df.empty:
        return {}

    code_col = next((c for c in df.columns if str(c) in ("代码", "code")), None)
    if code_col is None:
        return {}
    pe_col = next(
        (c for c in df.columns if "市盈" in str(c) and "动" in str(c)),
        next((c for c in df.columns if str(c) == "市盈率-动态"), None),
    )
    if pe_col is None:
        pe_col = next((c for c in df.columns if "市盈" in str(c)), None)
    tr_col = next((c for c in df.columns if "换手" in str(c)), None)

    out: dict[str, tuple[float | None, float | None]] = {}
    for _, row in df.iterrows():
        code = _extract_stock_code6(row.get(code_col))
        if code is None:
            continue
        pe_v: float | None = None
        tr_v: float | None = None
        if pe_col is not None:
            try:
                pe_raw = float(row[pe_col])
                if np.isfinite(pe_raw):
                    pe_v = pe_raw
            except (TypeError, ValueError):
                pe_v = None
        if tr_col is not None:
            try:
                tr_raw = float(row[tr_col])
                if np.isfinite(tr_raw):
                    tr_v = tr_raw
            except (TypeError, ValueError):
                tr_v = None
        out[code] = (pe_v, tr_v)
    return out


def resolve_incremental_daily_fetch_window(
    last_date_raw: object | None,
    latest_trade_date: str,
    end_compact: str,
    *,
    new_stock_history_days: int = 365,
) -> tuple[str | None, str]:
    """
    计算本地增量同步时 ``fetch_daily_hist`` 的 ``start_date``（紧凑 ``YYYYMMDD``）。

    Returns:
        (start_compact, kind)。kind 为 ``fetch`` 时应发起网络请求；
        ``uptodate`` / ``bad_range`` 时 ``start_compact`` 为 ``None``（不应请求）。
    """
    from .utils import next_trade_day_after

    lt = str(latest_trade_date).strip()[:10]
    ec = str(end_compact).strip().replace("-", "")[:8]

    if last_date_raw is not None:
        ld = str(last_date_raw).strip()[:10]
        if ld >= lt:
            return None, "uptodate"

    if last_date_raw is None:
        start_compact = (
            datetime.now().date() - timedelta(days=int(new_stock_history_days))
        ).strftime("%Y%m%d")
    else:
        ld_raw = str(last_date_raw).strip()[:10]
        nxt_trade = next_trade_day_after(ld_raw)
        if nxt_trade:
            start_compact = nxt_trade.replace("-", "")[:8]
        else:
            try:
                day_next = datetime.strptime(ld_raw, "%Y-%m-%d").date() + timedelta(
                    days=1
                )
            except ValueError:
                day_next = datetime.now().date() - timedelta(
                    days=int(new_stock_history_days)
                )
            start_compact = day_next.strftime("%Y%m%d")

    if start_compact > ec:
        return None, "bad_range"
    return start_compact, "fetch"


def load_stock_latest_date_map(db_path: Path | str | None = None) -> dict[str, str]:
    """
    返回各股票在本地库中的最新交易日 ``YYYY-MM-DD``（``stock_code -> max(date)``）。
    供增量同步脚本快速跳过已更新标的；等价于 ``database.fetch_stock_code_max_dates``。
    """
    from .database import DB_PATH, fetch_stock_code_max_dates

    p = Path(db_path) if db_path is not None else DB_PATH
    return fetch_stock_code_max_dates(p)


def datetime_today_compact() -> str:
    return datetime.now().strftime("%Y%m%d")


def has_enough_history(df: pd.DataFrame) -> bool:
    return len(df) >= MIN_HISTORY_BARS
