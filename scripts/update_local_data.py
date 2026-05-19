"""
全量初始化与每日增量更新本地 SQLite 数据库日线行情。

在 quant_select 根目录执行:
  python scripts/update_local_data.py

逻辑：本地无该代码记录则拉取约 365 自然日；已有则从「最近收盘日」次日增量拉取并 UPSERT。
交易日默认本地 15:00 起将当日收盘线纳入增量锚点（环境变量 ``QUANT_MARKET_CLOSE_HOUR`` / ``QUANT_MARKET_CLOSE_MINUTE`` 可调）；此前不含当日未收盘 K。
启动前一次性查询 ``SELECT stock_code, MAX(date) GROUP BY stock_code`` 建查找表；已对齐最近交易日的标的跳过网络请求。

全 A：使用 ``--all-stocks`` 或 ``--max-stocks 0`` 取消默认 6000 只上限，按 AkShare 全市场列表尽可能同步每一只（耗时更长）。

并发策略（SQLite 友好）：
  - **阶段 1**：``ThreadPoolExecutor`` 内仅执行网络拉取，不写数据库；
    进度约按任务数（默认至多每 50 只一批）打印，且默认 **每 30 秒** 心跳一行（可用环境变量 ``QUANT_UPDATE_FETCH_HEARTBEAT_SEC`` 调整）；
  - **阶段 2**：线程池结束后，由 **主线程** 将内存中的 K 线分批 ``UPSERT`` 并 ``commit``，避免多线程竞争连接导致锁等待或死锁。

行业字段：
  - K 线写入后默认调用 ``sync_stock_industries``，生成 ``{stock_code: 行业名}`` 并批量 UPDATE。
  - 数据源（``QUANT_INDUSTRY_SOURCE``）：``auto`` 时先东方财富行业板块，失败则 **Baostock** 全市场行业表。
  - 市值（``QUANT_MCAP_SOURCE``）：``auto`` 时先东财 push2，失败则用东财 datacenter ``stock_value_em``。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import DB_PATH, ensure_eastmoney_no_proxy_if_configured
from src.database import (
    fetch_stock_code_max_dates,
    get_connection,
    init_db,
    open_sqlite_connection,
    upsert_stock_daily_klines,
    upsert_stock_financial_rows,
)
from src.data_fetcher import (
    fetch_a_share_industry_map,
    fetch_daily_hist,
    fetch_market_cap_map,
    fetch_spot_pe_turnover_map,
    get_stock_pool,
    list_a_stock_codes,
    resolve_incremental_daily_fetch_window,
)
from src.utils import get_kline_incremental_end_trade_date

DEFAULT_WORKERS = max(1, int(os.environ.get("QUANT_UPDATE_WORKERS", "4")))
# 主线程单次 executemany 行数（300~500 降低 WAL 长事务锁竞争）
_raw_flush = int(os.environ.get("QUANT_UPSERT_FLUSH_ROWS", "400"))
UPSERT_FLUSH_ROWS = max(300, min(500, _raw_flush))
COMMIT_EVERY_FLUSHES = max(1, int(os.environ.get("QUANT_COMMIT_EVERY_FLUSHES", "3")))
DB_WRITE_BATCH_SIZE = max(300, min(500, int(os.environ.get("QUANT_DB_WRITE_BATCH", "400"))))
DB_WRITE_YIELD_SEC = float(os.environ.get("QUANT_DB_WRITE_YIELD_SEC", "0.01"))
DEFAULT_INDUSTRY_BOARD_SLEEP = float(os.environ.get("QUANT_INDUSTRY_BOARD_SLEEP", "0.2"))


def _db_write_yield() -> None:
    if DB_WRITE_YIELD_SEC > 0:
        time.sleep(DB_WRITE_YIELD_SEC)


def _chunked_executemany(
    conn,
    sql: str,
    rows: list,
    *,
    batch_size: int | None = None,
    commit_every: int = 1,
) -> int:
    """分批 executemany + 微休眠，减轻与 Streamlit 前台写入的锁竞争。"""
    if not rows:
        return 0
    bs = batch_size or DB_WRITE_BATCH_SIZE
    n_batches = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        conn.executemany(sql, chunk)
        n_batches += 1
        if n_batches % commit_every == 0:
            conn.commit()
        _db_write_yield()
    if n_batches % commit_every != 0:
        conn.commit()
    return n_batches


def sync_stock_industries(
    db_path: Path | None = None,
    *,
    verbose: bool = True,
    board_sleep_sec: float | None = None,
    max_boards: int | None = None,
) -> dict[str, int | str]:
    """
    拉取 A 股行业并写入 ``stock_daily_kline.industry``。

    默认 ``QUANT_INDUSTRY_SOURCE=auto``：东方财富行业板块 → Baostock 全市场行业表。

    Returns:
        统计字典：``n_codes_mapped``, ``rows_updated``, ``labeled_rows``, ``total_rows`` 等。
    """
    path = db_path or DB_PATH
    init_db(path)
    sleep_s = (
        float(board_sleep_sec)
        if board_sleep_sec is not None
        else DEFAULT_INDUSTRY_BOARD_SLEEP
    )

    if verbose:
        print("正在同步 A 股行业字段（东财板块 / Baostock 自动回退）…", flush=True)
    code_to_industry = fetch_a_share_industry_map(
        board_sleep_sec=sleep_s,
        max_boards=max_boards,
        verbose=verbose,
    )

    if verbose:
        print(
            f"行业映射构建完成：去重后股票 {len(code_to_industry)} 只；开始批量 UPDATE SQLite…",
            flush=True,
        )

    industry_pairs = [
        (str(ind).strip(), str(code).strip().zfill(6))
        for code, ind in code_to_industry.items()
        if len(str(code).strip().zfill(6)) == 6
    ]
    rows_updated = 0
    conn_ind = open_sqlite_connection(path)
    try:
        before = conn_ind.total_changes
        _chunked_executemany(
            conn_ind,
            "UPDATE stock_daily_kline SET industry = ? WHERE stock_code = ?",
            industry_pairs,
            batch_size=DB_WRITE_BATCH_SIZE,
        )
        rows_updated = int(conn_ind.total_changes - before)
    finally:
        conn_ind.close()

    with get_connection(path) as conn:
        total_rows = int(
            conn.execute("SELECT COUNT(*) FROM stock_daily_kline").fetchone()[0]
        )
        labeled_rows = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM stock_daily_kline
                WHERE industry IS NOT NULL AND TRIM(industry) != ''
                """
            ).fetchone()[0]
        )
        labeled_codes = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT stock_code) FROM stock_daily_kline
                WHERE industry IS NOT NULL AND TRIM(industry) != ''
                """
            ).fetchone()[0]
        )

    if verbose:
        print(
            f"行业同步完成：UPDATE 累计变更约 {rows_updated} 行；"
            f" K 线总行数 {total_rows}，其中 industry 非空行 {labeled_rows}；"
            f" 至少有一条 industry 非空的股票数 {labeled_codes}。",
            flush=True,
        )

    return {
        "n_codes_mapped": len(code_to_industry),
        "rows_updated_reported": rows_updated,
        "total_kline_rows": total_rows,
        "labeled_kline_rows": labeled_rows,
        "labeled_distinct_codes": labeled_codes,
    }


def _worker_fetch_only(
    code: str,
    name: str,
    last_date_raw: object | None,
    latest_trade: str,
    end_compact: str,
) -> tuple[list[dict] | None, str]:
    """
    仅在 worker 线程中执行网络请求与内存组装，**禁止**访问 SQLite。
    返回 (records, status)；status ∈ fetched | uptodate | bad_range | empty | error
    """
    code = str(code).strip().zfill(6)
    start_compact, kind = resolve_incremental_daily_fetch_window(
        last_date_raw,
        latest_trade,
        end_compact,
    )
    if kind != "fetch":
        return None, kind

    try:
        df = fetch_daily_hist(
            code,
            start_date=start_compact,
            end_date=end_compact,
            adjust="qfq",
        )
    except Exception:
        return None, "error"

    if df is None or df.empty:
        return None, "empty"

    records: list[dict] = []
    for _, row in df.iterrows():
        d = row["date"]
        ds = d if isinstance(d, str) else pd.Timestamp(d).strftime("%Y-%m-%d")
        rec: dict = {
            "date": ds,
            "stock_code": code,
            "stock_name": str(name).strip(),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        if "turnover_rate" in row.index and pd.notna(row.get("turnover_rate")):
            rec["turnover_rate"] = float(row["turnover_rate"])
        records.append(rec)
    return records, "fetched"


def update_database_kline(
    *,
    max_stocks: int = 6000,
    pool_type: str = "all",
    workers: int = DEFAULT_WORKERS,
    run_industry_sync: bool = True,
    run_market_cap_sync: bool = True,
    industry_board_limit: int = 0,
) -> None:
    init_db()

    max_count = None if max_stocks <= 0 else max_stocks

    print("正在拉取全市场股票代码列表...", flush=True)
    if max_count is None:
        print(
            "股票数量上限: 不限制（全 A，按接口返回列表遍历；未对齐最近交易日的才会请求 K 线）。",
            flush=True,
        )
    stock_pool = get_stock_pool(pool_type=pool_type, max_count=max_count)
    if not stock_pool:
        print("错误: 无法获取股票池，请检查网络连接！", flush=True)
        raise SystemExit(1)

    t0 = time.perf_counter()
    date_mapping = fetch_stock_code_max_dates(DB_PATH)
    t_map = time.perf_counter()

    total = len(stock_pool)
    latest_trade = get_kline_incremental_end_trade_date()
    end_compact = latest_trade.replace("-", "")

    print(
        f"全市场共获取到 {total} 只股票；本地已有缓存 {len(date_mapping)} 只。"
        f" MAX(date) 映射查询耗时 {(t_map - t0) * 1000:.1f} ms。"
        f" 增量截止(最近交易日): {latest_trade}；并发拉取线程: {workers}。",
        flush=True,
    )

    skipped_uptodate = 0
    pending: list[tuple[str, str, object | None]] = []
    for c, n in stock_pool:
        code = str(c).strip().zfill(6)
        last = date_mapping.get(code)
        if last is not None and str(last).strip()[:10] >= latest_trade:
            skipped_uptodate += 1
            continue
        pending.append((code, str(n).strip(), last))

    n_pending = len(pending)
    print(
        f"待拉取（未对齐 {latest_trade}）: {n_pending} 只；已跳过最新: {skipped_uptodate}。",
        flush=True,
    )

    # ---------- 阶段 1：仅并发拉取（零 SQLite） ----------
    t_fetch0 = time.perf_counter()
    outcomes: list[tuple[list[dict] | None, str]] = []
    workers_n = max(1, min(workers, 32))

    # 单只股票 HTTP 可能数十秒（超时+重试），若仅按「每 500 只」打印会长时间无输出，误判卡死
    progress_every = max(1, min(50, max(n_pending // 40, 1)))
    heartbeat_sec = float(os.environ.get("QUANT_UPDATE_FETCH_HEARTBEAT_SEC", "30"))

    print(
        f"阶段 1：并发拉取 {n_pending} 只股票（线程 {workers_n}），"
        f"约每 {progress_every} 只或每 {heartbeat_sec:.0f}s 打印进度…",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=workers_n) as ex:
        futs = [
            ex.submit(
                _worker_fetch_only,
                code,
                name,
                last_raw,
                latest_trade,
                end_compact,
            )
            for code, name, last_raw in pending
        ]
        done = 0
        last_progress_print = t_fetch0
        for fut in as_completed(futs):
            done += 1
            now = time.perf_counter()
            elapsed = now - t_fetch0
            due_count = done % progress_every == 0
            due_final = done == n_pending
            due_heartbeat = heartbeat_sec > 0 and (now - last_progress_print) >= heartbeat_sec
            if n_pending and (done == 1 or due_count or due_final or due_heartbeat):
                rate = done / elapsed if elapsed > 1e-6 else 0.0
                eta_s = (n_pending - done) / rate if rate > 1e-9 else float("nan")
                eta_txt = (
                    f"，估计剩余约 {eta_s / 60:.1f} 分钟"
                    if eta_s == eta_s and eta_s < 864000
                    else ""
                )
                print(
                    f"  [拉取] {done}/{n_pending} 已完成；已用时 {elapsed:.0f}s"
                    f"{eta_txt}…",
                    flush=True,
                )
                last_progress_print = now
            try:
                outcomes.append(fut.result())
            except Exception:
                outcomes.append((None, "error"))

    t_fetch1 = time.perf_counter()
    print(
        f"阶段 1 结束：并发拉取耗时 {t_fetch1 - t_fetch0:.1f}s；开始主线程批量写入…",
        flush=True,
    )

    skipped_bad_range = 0
    empty_fetch = 0
    errors = 0
    fetch_jobs = 0
    pending_writes: list[list[dict]] = []

    for records, status in outcomes:
        if status == "uptodate":
            skipped_uptodate += 1
            continue
        if status == "bad_range":
            skipped_bad_range += 1
            continue
        if status == "empty":
            empty_fetch += 1
            continue
        if status == "error":
            errors += 1
            continue
        if status == "fetched" and records:
            fetch_jobs += 1
            pending_writes.append(records)

    # ---------- 阶段 2：主线程分批 UPSERT ----------
    stocks_batches = 0
    buf: list[dict] = []
    flush_ops = 0
    conn = open_sqlite_connection(DB_PATH)
    try:
        for recs in pending_writes:
            buf.extend(recs)
            stocks_batches += 1
            while len(buf) >= UPSERT_FLUSH_ROWS:
                chunk = buf[:UPSERT_FLUSH_ROWS]
                del buf[:UPSERT_FLUSH_ROWS]
                upsert_stock_daily_klines(chunk, connection=conn)
                flush_ops += 1
                if flush_ops % COMMIT_EVERY_FLUSHES == 0:
                    conn.commit()
                    print(
                        f"  [写入] 已执行 upsert 批次 {flush_ops} "
                        f"（约每批 {UPSERT_FLUSH_ROWS} 行）…",
                        flush=True,
                    )
                _db_write_yield()

        if buf:
            upsert_stock_daily_klines(buf, connection=conn)
            flush_ops += 1
        conn.commit()
        _db_write_yield()
    finally:
        conn.close()

    t_end = time.perf_counter()
    print(
        f"本轮结束：共 {total} 只股票；无需请求(已跟上最近交易日): {skipped_uptodate}；"
        f" 区间无效: {skipped_bad_range}；空返回: {empty_fetch}；拉取异常: {errors}；"
        f" 有效拉取股票数: {fetch_jobs}；参与写入的股票批次数: {stocks_batches}；"
        f" DB upsert 次数: {flush_ops}；总耗时 {t_end - t0:.1f}s。",
        flush=True,
    )

    if run_industry_sync:
        print("", flush=True)
        lim = industry_board_limit if industry_board_limit > 0 else None
        try:
            sync_stock_industries(DB_PATH, verbose=True, max_boards=lim)
        except RuntimeError as exc:
            print(f"警告: 行业字段同步未成功（K 线仍已写入）: {exc}", flush=True)

    if run_market_cap_sync:
        print("", flush=True)
        try:
            sync_market_cap_batch(DB_PATH, verbose=True)
        except RuntimeError as exc:
            print(f"警告: 市值同步未成功（K 线仍已写入）: {exc}", flush=True)


def sync_market_cap_batch(
    db_path: Path | None = None,
    *,
    verbose: bool = True,
    max_workers: int = 8,
    sleep_sec: float = 1.5,
) -> dict[str, int | str]:
    """
    并发获取全 A 总市值（元）并 UPDATE ``stock_daily_kline.market_cap``。

    ``QUANT_MCAP_SOURCE=auto`` 时：东财 push2 → 东财 datacenter（``stock_value_em``）。
    代码表优先 AkShare，失败时 Baostock。
    """
    path = db_path or DB_PATH
    init_db(path)

    if verbose:
        print("正在获取全 A 股代码列表...", flush=True)
    pairs = list_a_stock_codes()
    if not pairs:
        raise RuntimeError("获取股票代码列表失败（AkShare 与 Baostock 均无数据）")
    codes = [c for c, _ in pairs]
    if verbose:
        print(
            f"共 {len(codes)} 只 A 股，开始并发获取市值（线程={max_workers}，"
            f"间隔={sleep_sec}s，push2 失败自动换 datacenter）…",
            flush=True,
        )

    code_to_mcap = fetch_market_cap_map(
        codes,
        max_workers=max_workers,
        sleep_sec=sleep_sec,
        verbose=verbose,
    )
    n_got = len(code_to_mcap)
    total = len(codes)
    if verbose:
        print(f"市值获取完成：{n_got}/{total} 只有效市值", flush=True)

    if n_got == 0:
        raise RuntimeError("未能获取任何有效市值数据")

    # Batch UPDATE
    if verbose:
        print("正在批量 UPDATE stock_daily_kline.market_cap...", flush=True)
    conn = open_sqlite_connection(path)
    try:
        pairs = [(float(code_to_mcap[c]), c) for c in code_to_mcap]
        _chunked_executemany(
            conn,
            """
            UPDATE stock_daily_kline
            SET market_cap = ?
            WHERE stock_code = ?
            """,
            pairs,
            batch_size=DB_WRITE_BATCH_SIZE,
        )
        updated = int(
            conn.execute(
                "SELECT COUNT(*) FROM stock_daily_kline WHERE market_cap IS NOT NULL AND market_cap > 0"
            ).fetchone()[0]
        )
    finally:
        conn.close()

    result = {
        "n_codes_total": total,
        "n_codes_with_mcap": n_got,
        "n_kline_rows_with_mcap": updated,
    }
    if verbose:
        print(
            f"市值同步完成：{n_got} 只股票有有效市值，"
            f"已为 {updated} 行 K 线记录写入 market_cap。",
            flush=True,
        )
    return result


def sync_spot_pe_turnover_to_latest_kline(
    db_path: Path | None = None,
    *,
    verbose: bool = True,
) -> int:
    """
    用东财实时快照为各股「最近一根 K 线」写入 ``pe_ttm`` / ``turnover_rate``（若快照有值）。
    """
    from src.utils import get_kline_incremental_end_trade_date

    path = db_path or DB_PATH
    init_db(path)
    latest = get_kline_incremental_end_trade_date()
    spot = fetch_spot_pe_turnover_map()
    if not spot:
        if verbose:
            print("[估值] 实时快照为空或拉取失败，跳过 pe/换手 写入。", flush=True)
        return 0

    conn = open_sqlite_connection(path)
    updated = 0
    try:
        for code, (pe_v, tr_v) in spot.items():
            if pe_v is None and tr_v is None:
                continue
            cur = conn.execute(
                """
                UPDATE stock_daily_kline
                SET pe_ttm = COALESCE(?, pe_ttm),
                    turnover_rate = COALESCE(?, turnover_rate)
                WHERE stock_code = ? AND date = (
                    SELECT MAX(date) FROM stock_daily_kline WHERE stock_code = ?
                )
                """,
                (pe_v, tr_v, code, code),
            )
            updated += int(cur.rowcount or 0)
        conn.commit()
    finally:
        conn.close()
    if verbose:
        print(
            f"[估值] 已为最近交易日 {latest} 更新 pe/换手 字段（触及约 {updated} 只）。",
            flush=True,
        )
    return updated


def sync_fundamental_data(db_path: Path | None = None) -> None:
    """
    使用 AkShare 东方财富业绩报表接口增量同步 ``stock_financial_data``（季报关键字段）。
    与日线拼接时以 ``pub_date`` 为公告可见日，避免前视偏差。
    """
    from datetime import datetime

    import akshare as ak

    ensure_eastmoney_no_proxy_if_configured()
    path = db_path or DB_PATH
    init_db(path)

    print("[基本面] 开始增量同步上市公司季度业绩报表（东方财富）...", flush=True)
    now = datetime.now()
    cy = now.year
    quarters = [
        f"{cy}0331",
        f"{cy}0630",
        f"{cy}0930",
        f"{cy}1231",
        f"{cy - 1}1231",
        f"{cy - 1}0930",
    ]

    def _norm_code(raw: object) -> str:
        s = str(raw).strip()
        if "." in s:
            s = s.split(".")[0]
        return s.zfill(6)[:6]

    def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
        for n in names:
            if n in df.columns:
                return df[n]
        return None

    for report_date in quarters:
        print(f"[基本面] 拉取报告期 {report_date} ...", flush=True)
        df_finance = None
        try:
            if hasattr(ak, "stock_yjbb_em"):
                df_finance = ak.stock_yjbb_em(date=report_date)
            elif hasattr(ak, "stock_em_yjbb"):
                df_finance = ak.stock_em_yjbb(date=report_date)
        except Exception as exc:
            print(f"[基本面] 报告期 {report_date} 接口异常，跳过: {exc}", flush=True)
            continue

        if df_finance is None or df_finance.empty:
            print(f"[基本面] 报告期 {report_date} 返回空表，跳过。", flush=True)
            continue

        code_s = _first_col(df_finance, ("股票代码", "代码"))
        if code_s is None:
            print(f"[基本面] 报告期 {report_date} 无股票代码列，跳过。", flush=True)
            continue

        pub_s = _first_col(df_finance, ("最新公告日期", "公告日期"))
        roe_s = _first_col(df_finance, ("净资产收益率",))
        np_s = _first_col(
            df_finance,
            ("净利润-同比增长", "净利润同比增长", "净利润同比"),
        )
        rev_s = _first_col(
            df_finance,
            ("营业收入-同比增长", "营业收入同比增长", "营业收入同比"),
        )

        rows: list[tuple[object, ...]] = []
        for i in range(len(df_finance)):
            code = _norm_code(code_s.iloc[i])
            if len(code) != 6 or not code.isdigit():
                continue
            if pub_s is not None:
                pd_pub = pd.to_datetime(pub_s.iloc[i], errors="coerce")
                pub_date = (
                    pd_pub.strftime("%Y-%m-%d")
                    if pd.notna(pd_pub)
                    else now.strftime("%Y-%m-%d")
                )
            else:
                pub_date = now.strftime("%Y-%m-%d")
            roe = (
                pd.to_numeric(roe_s.iloc[i], errors="coerce") if roe_s is not None else np.nan
            )
            ng = (
                pd.to_numeric(np_s.iloc[i], errors="coerce") if np_s is not None else np.nan
            )
            rg = (
                pd.to_numeric(rev_s.iloc[i], errors="coerce") if rev_s is not None else np.nan
            )
            if pd.isna(roe) and pd.isna(ng):
                continue

            def _sql_val(x: object) -> float | None:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return None
                v = float(x)
                return v if np.isfinite(v) else None

            rows.append(
                (code, pub_date, report_date, _sql_val(roe), _sql_val(ng), _sql_val(rg))
            )

        if not rows:
            print(f"[基本面] 报告期 {report_date} 无有效行，跳过。", flush=True)
            continue
        try:
            n = upsert_stock_financial_rows(rows, db_path=path)
            print(f"[基本面] 报告期 {report_date} 已入库 {n} 条。", flush=True)
        except Exception as exc:
            print(f"[基本面] 报告期 {report_date} 写入失败: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="同步本地 stock_daily_kline")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=6000,
        help="最多同步的股票数量上限；0 表示不限制（全 A）。等价于 --all-stocks",
    )
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="同步 AkShare 全 A 列表中的全部代码（等同于 --max-stocks 0；耗时与流量更大）",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="all",
        help="股票池类型：all | hs300 | zz500（默认 all）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发拉取线程数（默认 {DEFAULT_WORKERS}，可用环境变量 QUANT_UPDATE_WORKERS）",
    )
    parser.add_argument(
        "--skip-industry-sync",
        action="store_true",
        help="不同步行业字段到 stock_daily_kline.industry",
    )
    parser.add_argument(
        "--industry-source",
        type=str,
        default="",
        choices=["", "auto", "em", "baostock"],
        help="行业数据源：auto=东财板块失败后 Baostock；baostock=仅 Baostock（东财不可用时推荐）",
    )
    parser.add_argument(
        "--skip-market-cap-sync",
        action="store_true",
        help="不同步全 A 股总市值到 stock_daily_kline.market_cap",
    )
    parser.add_argument(
        "--skip-fundamental-sync",
        action="store_true",
        help="不同步东方财富季度业绩报表到 stock_financial_data",
    )
    parser.add_argument(
        "--only-industry-sync",
        action="store_true",
        help="仅执行行业同步（不拉取 K 线）",
    )
    parser.add_argument(
        "--only-fundamental-sync",
        action="store_true",
        help="仅执行基本面财报同步（不拉取 K 线）",
    )
    parser.add_argument(
        "--industry-board-limit",
        type=int,
        default=0,
        help="调试：最多遍历前 N 个行业板块（0 表示不限制）",
    )
    args = parser.parse_args()

    if args.industry_source:
        os.environ["QUANT_INDUSTRY_SOURCE"] = args.industry_source

    if args.only_industry_sync:
        init_db()
        lim = args.industry_board_limit if args.industry_board_limit > 0 else None
        sync_stock_industries(DB_PATH, verbose=True, max_boards=lim)
        return 0

    if args.only_fundamental_sync:
        init_db()
        sync_fundamental_data(DB_PATH)
        return 0

    eff_max = 0 if args.all_stocks else args.max_stocks

    update_database_kline(
        max_stocks=eff_max,
        pool_type=args.pool,
        workers=args.workers,
        run_industry_sync=not args.skip_industry_sync,
        industry_board_limit=args.industry_board_limit,
        run_market_cap_sync=not args.skip_market_cap_sync,
    )
    if not args.skip_fundamental_sync:
        try:
            sync_fundamental_data(DB_PATH)
        except Exception as exc:
            print(f"警告: 基本面财报同步失败（K 线已写入）: {exc}", flush=True)
    try:
        sync_spot_pe_turnover_to_latest_kline(DB_PATH)
    except Exception as exc:
        print(f"警告: 实时 pe/换手 写入失败: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
