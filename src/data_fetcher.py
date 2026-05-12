"""A 股日线数据获取（AkShare 优先，失败时可 Baostock 兜底）。"""
from __future__ import annotations

import atexit
import logging
import random
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
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
    """统一数值列与日期格式。"""
    if df.empty:
        return df
    need = {"date", "open", "high", "low", "close", "volume"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
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


def list_a_stock_codes(max_count: int | None = None) -> list[tuple[str, str]]:
    """
    返回 (code6, name) 列表。code6 为 6 位数字，不含市场前缀。
    """
    df = ak.stock_info_a_code_name()
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
        if max_count is not None and len(pairs) >= max_count:
            break
    return pairs


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
        df = df.rename(columns=rename)

    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    return _finalize_hist_columns(df)


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
