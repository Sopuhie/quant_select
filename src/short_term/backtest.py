# -*- coding: utf-8 -*-
"""
短线规则历史滚动回测（纯日线，与 ``execution.evaluate_daily_exit`` 一致）。

流程
----
对每个信号日 T（在 ``[start_date, scan_end_date]`` 内）：

1. ``ShortTermRuleStrategy.scan(T)`` — 与实盘扫描相同，无未来数据；
2. 对 Top N 入选逐笔模拟：T 收盘价买入 → T+1 止损 / T+N 收盘卖出；
3. 信号日等权汇总当日收益，再复利拼接净值曲线。

说明
----
- **不写库**：仅在内存中回放，不覆盖 ``short_daily_selections``。
- **摩擦成本**：默认不计佣金/印花税（与 ``execution.py`` 一致）；可在汇总层自行解读。
- **性能**：逐日全市场扫描较慢，长区间建议命令行并适当 ``--max-scan-stocks``。
- **区间截止**：自动将 ``end_date`` 前移，保证最后一笔信号有足够 T+1/T+2 K 线可平仓。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_DIR

from .config import (
    SHORT_MIN_MARKET_SCORE,
    SHORT_SELL_OFFSET,
    SHORT_STOP_LOSS_RATIO,
    SHORT_TOP_N,
)
from .execution import (
    ORDER_STATUS_CLOSED,
    ORDER_STATUS_HOLDING,
    evaluate_daily_exit,
    fetch_post_signal_ohlc,
)
from .review_prices import resolve_t1_t2_dates
from .strategy import ShortTermRuleStrategy

SHORT_TERM_BACKTEST_TRADES_CSV = DATA_DIR / "short_term_backtest_trades.csv"
SHORT_TERM_BACKTEST_DAILY_CSV = DATA_DIR / "short_term_backtest_daily.csv"
SHORT_TERM_BACKTEST_SUMMARY_JSON = DATA_DIR / "short_term_backtest_summary.json"


def list_kline_trade_dates(
    conn: sqlite3.Connection,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    """库内 ``stock_daily_kline`` 交易日列表（升序）。"""
    clauses: list[str] = []
    params: list[str] = []
    if start_date:
        clauses.append("date >= ?")
        params.append(str(start_date).strip()[:10])
    if end_date:
        clauses.append("date <= ?")
        params.append(str(end_date).strip()[:10])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT DISTINCT date FROM stock_daily_kline {where} ORDER BY date ASC",
        params,
    ).fetchall()
    return [str(r[0]).strip()[:10] for r in rows if r[0]]


def resolve_scan_end_date(
    trade_dates: list[str],
    *,
    sell_offset: int | None = None,
) -> str | None:
    """
    最后一个可完整平仓的信号日。

    信号日 T 需存在 T+``sell_offset`` 的 K 线（见 ``SHORT_SELL_OFFSET``）。
    """
    offset = int(sell_offset if sell_offset is not None else SHORT_SELL_OFFSET)
    offset = max(1, min(2, offset))
    if len(trade_dates) <= offset:
        return None
    return trade_dates[-(offset + 1)]


def simulate_short_trade(
    conn: sqlite3.Connection,
    signal_date: str,
    row: dict[str, Any],
    *,
    sell_offset: int | None = None,
) -> dict[str, Any]:
    """单笔短线模拟（不写订单表）。"""
    td = str(signal_date).strip()[:10]
    code = str(row.get("stock_code", "")).strip().zfill(6)
    buy_price = float(row.get("close_price") or 0.0)
    t1_date, t2_date = resolve_t1_t2_dates(td, conn)
    bars = fetch_post_signal_ohlc(conn, code, td)
    exit_info = evaluate_daily_exit(
        buy_price,
        t1_bar=bars["t1"],
        t2_bar=bars["t2"],
        t1_date=t1_date,
        t2_date=t2_date,
        sell_offset=sell_offset,
    )
    return {
        "signal_date": td,
        "stock_code": code,
        "stock_name": row.get("stock_name"),
        "rank": row.get("rank"),
        "rule_score": row.get("rule_score"),
        "buy_price": buy_price,
        "sell_date": exit_info.get("sell_date"),
        "sell_price": exit_info.get("sell_price"),
        "hold_days": exit_info.get("hold_days"),
        "pnl_ratio": exit_info.get("pnl_ratio"),
        "status": exit_info.get("status"),
        "stop_loss_triggered": exit_info.get("stop_loss_triggered"),
        "exit_reason": exit_info.get("exit_reason"),
    }


def _max_drawdown(nav_series: list[float]) -> float | None:
    if len(nav_series) < 2:
        return None
    peak = nav_series[0]
    max_dd = 0.0
    for nav in nav_series:
        peak = max(peak, nav)
        if peak > 0:
            max_dd = min(max_dd, (nav - peak) / peak)
    return float(max_dd)


def summarize_backtest(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """由交易明细与日汇总生成摘要指标。"""
    closed = trades_df[
        trades_df["status"].astype(str).str.upper() == ORDER_STATUS_CLOSED
    ].copy()
    pnls = pd.to_numeric(closed["pnl_ratio"], errors="coerce").dropna()
    win_rate = float((pnls > 0).mean()) if len(pnls) else None

    nav = 1.0
    nav_points: list[float] = [nav]
    for ret in pd.to_numeric(daily_df.get("day_return"), errors="coerce"):
        if pd.isna(ret):
            continue
        nav *= 1.0 + float(ret)
        nav_points.append(nav)

    exit_counts: dict[str, int] = {}
    if not closed.empty and "exit_reason" in closed.columns:
        exit_counts = (
            closed["exit_reason"].fillna("unknown").astype(str).value_counts().to_dict()
        )

    summary = {
        **meta,
        "signal_days": int(
            daily_df.loc[daily_df["day_type"] == "signal", "signal_date"].nunique()
        )
        if not daily_df.empty
        else 0,
        "fused_days": int((daily_df["day_type"] == "fused").sum())
        if not daily_df.empty
        else 0,
        "empty_days": int((daily_df["day_type"] == "empty").sum())
        if not daily_df.empty
        else 0,
        "total_trades": int(len(trades_df)),
        "closed_trades": int(len(closed)),
        "holding_trades": int(
            (trades_df["status"].astype(str).str.upper() == ORDER_STATUS_HOLDING).sum()
        )
        if not trades_df.empty
        else 0,
        "win_rate": win_rate,
        "avg_pnl_pct": float(pnls.mean() * 100.0) if len(pnls) else None,
        "median_pnl_pct": float(pnls.median() * 100.0) if len(pnls) else None,
        "cum_return_pct": float((nav - 1.0) * 100.0) if nav_points else None,
        "max_drawdown_pct": (
            float(_max_drawdown(nav_points) * 100.0)
            if _max_drawdown(nav_points) is not None
            else None
        ),
        "exit_reason_counts": exit_counts,
    }
    return summary


def save_backtest_outputs(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    summary: dict[str, Any],
    *,
    trades_path: Path | None = None,
    daily_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, str]:
    """落盘 CSV 与摘要 JSON。"""
    tp = trades_path or SHORT_TERM_BACKTEST_TRADES_CSV
    dp = daily_path or SHORT_TERM_BACKTEST_DAILY_CSV
    sp = summary_path or SHORT_TERM_BACKTEST_SUMMARY_JSON
    tp.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(tp, index=False, encoding="utf-8-sig")
    daily_df.to_csv(dp, index=False, encoding="utf-8-sig")
    sp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "trades_csv": str(tp),
        "daily_csv": str(dp),
        "summary_json": str(sp),
    }


def run_short_term_rolling_backtest(
    conn: sqlite3.Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    top_n: int | None = None,
    include_300: bool = False,
    include_688: bool = False,
    max_scan_stocks: int | None = None,
    sell_offset: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    滚动回放短线规则并返回 trades / daily / summary。

    ``start_date`` / ``end_date`` 缺省时取库内 MIN/MAX(date)。
    """
    top_n = int(top_n if top_n is not None else SHORT_TOP_N)
    offset = int(sell_offset if sell_offset is not None else SHORT_SELL_OFFSET)
    offset = max(1, min(2, offset))

    all_dates = list_kline_trade_dates(conn, start_date, end_date)
    if not all_dates:
        raise ValueError("本地 stock_daily_kline 无可用交易日，请先同步行情。")

    lo = all_dates[0]
    hi = all_dates[-1]
    scan_end = resolve_scan_end_date(all_dates, sell_offset=offset)
    if not scan_end:
        raise ValueError(
            f"交易日不足 {offset + 1} 天，无法完成 T+{offset} 平仓模拟。"
        )

    scan_dates = [d for d in all_dates if d <= scan_end]
    if not scan_dates:
        raise ValueError("可扫描信号日为空，请扩大日期区间或同步更多 K 线。")

    if verbose:
        print(
            f"[短线回测] 区间 {lo} ~ {hi}，信号日 {scan_dates[0]} ~ {scan_end}，"
            f"共 {len(scan_dates)} 个交易日 · Top {top_n} · T+{offset} 平仓",
            flush=True,
        )

    engine = ShortTermRuleStrategy(conn)
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    for i, td in enumerate(scan_dates):
        if verbose and (i % 10 == 0 or i == len(scan_dates) - 1):
            print(f"[短线回测] {i + 1}/{len(scan_dates)} {td} …", flush=True)

        _df, _, mkt_score = engine.scan(
            td,
            top_n=top_n,
            max_scan_stocks=max_scan_stocks,
            include_300=include_300,
            include_688=include_688,
        )
        mkt_score = int(mkt_score or 0)

        if mkt_score < SHORT_MIN_MARKET_SCORE:
            daily_rows.append(
                {
                    "signal_date": td,
                    "market_score": mkt_score,
                    "pick_count": 0,
                    "closed_count": 0,
                    "day_return": None,
                    "cum_nav": None,
                    "day_type": "fused",
                }
            )
            continue

        picks = engine.get_last_persist_rows()
        if not picks:
            daily_rows.append(
                {
                    "signal_date": td,
                    "market_score": mkt_score,
                    "pick_count": 0,
                    "closed_count": 0,
                    "day_return": None,
                    "cum_nav": None,
                    "day_type": "empty",
                }
            )
            continue

        day_pnls: list[float] = []
        closed_n = 0
        for row in picks:
            row = dict(row)
            row["market_score"] = mkt_score
            tr = simulate_short_trade(conn, td, row, sell_offset=offset)
            tr["market_score"] = mkt_score
            trade_rows.append(tr)
            if tr.get("status") == ORDER_STATUS_CLOSED and tr.get("pnl_ratio") is not None:
                day_pnls.append(float(tr["pnl_ratio"]))
                closed_n += 1

        day_ret = float(np.mean(day_pnls)) if day_pnls else None
        daily_rows.append(
            {
                "signal_date": td,
                "market_score": mkt_score,
                "pick_count": len(picks),
                "closed_count": closed_n,
                "day_return": day_ret,
                "cum_nav": None,
                "day_type": "signal",
            }
        )

    trades_df = pd.DataFrame(trade_rows)
    daily_df = pd.DataFrame(daily_rows)

    nav = 1.0
    cum_navs: list[float | None] = []
    for _, r in daily_df.iterrows():
        ret = r.get("day_return")
        if ret is not None and pd.notna(ret):
            nav *= 1.0 + float(ret)
            cum_navs.append(nav)
        else:
            cum_navs.append(None)
    daily_df["cum_nav"] = cum_navs

    meta = {
        "start_date": scan_dates[0],
        "end_date": hi,
        "scan_end_date": scan_end,
        "top_n": top_n,
        "sell_offset": offset,
        "stop_loss_ratio": SHORT_STOP_LOSS_RATIO,
        "min_market_score": SHORT_MIN_MARKET_SCORE,
        "include_300": include_300,
        "include_688": include_688,
        "max_scan_stocks": max_scan_stocks,
        "calendar_days_scanned": len(scan_dates),
    }
    summary = summarize_backtest(trades_df, daily_df, meta=meta)
    paths = save_backtest_outputs(trades_df, daily_df, summary)

    if verbose:
        print(
            "[短线回测] 完成:",
            {
                "cum_return_pct": summary.get("cum_return_pct"),
                "win_rate": summary.get("win_rate"),
                "closed_trades": summary.get("closed_trades"),
                **paths,
            },
            flush=True,
        )

    return {
        "trades": trades_df,
        "daily": daily_df,
        "summary": summary,
        "paths": paths,
    }
