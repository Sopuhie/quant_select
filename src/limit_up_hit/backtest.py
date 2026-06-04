# -*- coding: utf-8 -*-
"""
打板历史滚动回测（纯日线，不写库）。

规则
----
- T 日：``LimitUpHitStrategy.scan`` 确认涨停封板信号；
- T+1：若非一字涨停，按 **开盘价** 买入；
- T+2 起：若 **开盘价为一字涨停** 则连板骑乘；**开板** 当日按 **开盘价** 卖出。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.market_regime import short_term_market_allows_trading
from src.short_term.backtest import (
    list_kline_trade_dates,
    resolve_scan_end_date,
    summarize_backtest,
)

from .config import (
    LUH_BACKTEST_LEGACY,
    LUH_BACKTEST_MAX_RIDE_BARS,
    LUH_BT_MIN_CONCEPT_LIMIT_UP,
    LUH_BT_MIN_INDEX_RET,
    LUH_BT_MIN_MARKET_LIMIT_UP,
    LUH_BT_REQUIRE_T1_CLOSE_LIMIT,
    LUH_BT_SLIPPAGE,
    LUH_BT_STOP_LOSS,
    LUH_BT_STRONG_BOARD_TURNOVER,
    LUH_BT_T1_LOW_CLOSE_MIN,
    LUH_BT_TP_CLOSE_TIER1,
    LUH_BT_TP_CLOSE_TIER2,
    LUH_MARKET_INDEX_CODE,
    LUH_MARKET_MOMENTUM_DAYS,
    LUH_MARKET_MOMENTUM_MIN,
    LUH_MIN_MARKET_SCORE,
    LUH_TOP_N,
)
from .backtest_filters import (
    backtest_concept_allows,
    backtest_market_allows_trading,
    clear_backtest_filter_cache,
    is_limit_up_close,
)
from .execution import (
    ORDER_STATUS_CLOSED,
    ORDER_STATUS_HOLDING,
    ORDER_STATUS_SKIPPED,
    fetch_post_signal_ohlc,
    is_one_word_limit_down,
    is_one_word_limit_up,
)
from src.factor_calculator import is_bar_limit_up
from .review_prices import resolve_t1_t2_dates
from .strategy import LimitUpHitStrategy

LUH_BACKTEST_TRADES_CSV = DATA_DIR / "limit_up_backtest_trades.csv"
LUH_BACKTEST_DAILY_CSV = DATA_DIR / "limit_up_backtest_daily.csv"
LUH_BACKTEST_SUMMARY_JSON = DATA_DIR / "limit_up_backtest_summary.json"

# T 信号 → T+1 买 → 最早 T+2 可卖
_BACKTEST_MIN_LAG = 2


def fetch_post_t1_bars(
    conn: sqlite3.Connection,
    stock_code: str,
    t1_date: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """T+1 之后逐日 K 线（含 T+2 起），用于连板骑乘退出判定。"""
    code = str(stock_code).strip().zfill(6)
    td = str(t1_date).strip()[:10]
    cap = int(limit if limit is not None else LUH_BACKTEST_MAX_RIDE_BARS)
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(stock_daily_kline)").fetchall()
    }
    to_col = "turnover_rate" if "turnover_rate" in cols else "NULL AS turnover_rate"
    cur = conn.execute(
        f"""
        SELECT date, open, high, low, close, {to_col}
        FROM stock_daily_kline
        WHERE stock_code = ? AND date > ?
        ORDER BY date ASC
        LIMIT ?
        """,
        (code, td, cap),
    )
    out: list[dict[str, Any]] = []
    for row in cur.fetchall():
        d = str(row[0]).strip()[:10]
        try:
            o, hi, lo, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            tr = float(row[5]) if row[5] is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if all(np.isfinite(x) for x in (o, hi, lo, c)):
            item: dict[str, Any] = {
                "date": d,
                "open": o,
                "high": hi,
                "low": lo,
                "close": c,
            }
            if np.isfinite(tr):
                item["turnover"] = tr
            out.append(item)
    return out


def _bar_ohlc_dict(bar: dict[str, Any]) -> dict[str, float | None]:
    return {
        "open": bar.get("open"),
        "high": bar.get("high"),
        "low": bar.get("low"),
        "close": bar.get("close"),
    }


def evaluate_optimized_backtest_trade(
    signal_close: float,
    stock_code: str,
    *,
    t1_bar: dict[str, float | None],
    t1_date: str | None,
    post_t1_bars: list[dict[str, Any]] | None = None,
    t2_date: str | None = None,
) -> dict[str, Any]:
    """
    优化回测：T+1 非一字开盘价+滑点买入；收盘止盈/止损；T+2 强板续骑。
    """
    sig = float(signal_close)
    code = str(stock_code).strip().zfill(6)
    base_out: dict[str, Any] = {"t1_date": t1_date, "t2_date": t2_date, "ride_days": 0}
    slip = float(LUH_BT_SLIPPAGE)

    if sig <= 0:
        return {**base_out, "status": ORDER_STATUS_SKIPPED, "buy_price": None,
                "sell_date": None, "sell_price": None, "hold_days": 0, "pnl_ratio": None,
                "exit_reason": "invalid_signal_close"}

    if is_one_word_limit_up(t1_bar, sig, code):
        return {**base_out, "status": ORDER_STATUS_SKIPPED, "buy_price": None,
                "sell_date": None, "sell_price": None, "hold_days": 0, "pnl_ratio": None,
                "exit_reason": "t1_one_word_limit_up"}

    t1_close = t1_bar.get("close")
    t1_open = t1_bar.get("open")
    t1_low = t1_bar.get("low")
    if t1_close is None or t1_open is None:
        return {**base_out, "status": ORDER_STATUS_HOLDING, "buy_price": None,
                "sell_date": None, "sell_price": None, "hold_days": 0, "pnl_ratio": None,
                "exit_reason": "await_t1_kline"}

    t1_close_f = float(t1_close)
    t1_open_f = float(t1_open)
    if LUH_BT_REQUIRE_T1_CLOSE_LIMIT and not is_bar_limit_up(t1_close_f, sig, code):
        return {**base_out, "status": ORDER_STATUS_SKIPPED, "buy_price": None,
                "sell_date": None, "sell_price": None, "hold_days": 0, "pnl_ratio": None,
                "exit_reason": "t1_not_limit_up_close"}

    if (
        LUH_BT_REQUIRE_T1_CLOSE_LIMIT
        and t1_low is not None
        and np.isfinite(float(t1_low))
        and t1_close_f > 0
    ):
        if float(t1_low) / t1_close_f < float(LUH_BT_T1_LOW_CLOSE_MIN):
            return {**base_out, "status": ORDER_STATUS_SKIPPED, "buy_price": None,
                    "sell_date": None, "sell_price": None, "hold_days": 0, "pnl_ratio": None,
                    "exit_reason": "t1_intraday_open_board"}
    elif (
        not LUH_BT_REQUIRE_T1_CLOSE_LIMIT
        and t1_low is not None
        and np.isfinite(float(t1_low))
        and t1_open_f > 0
    ):
        # 追高过滤：T+1 低开过大则放弃（相对信号收盘 -5% 以下）
        if float(t1_open_f) < sig * 0.95:
            return {**base_out, "status": ORDER_STATUS_SKIPPED, "buy_price": None,
                    "sell_date": None, "sell_price": None, "hold_days": 0, "pnl_ratio": None,
                    "exit_reason": "t1_open_gap_down"}

    if t1_open_f <= 0:
        return {**base_out, "status": ORDER_STATUS_HOLDING, "buy_price": None,
                "sell_date": None, "sell_price": None, "hold_days": 0, "pnl_ratio": None,
                "exit_reason": "await_t1_kline"}

    buy_price = t1_open_f * (1.0 + slip)
    exit_bars = list(post_t1_bars or [])
    if not exit_bars:
        return {**base_out, "status": ORDER_STATUS_HOLDING, "buy_price": buy_price,
                "sell_date": None, "sell_price": None, "hold_days": 1, "pnl_ratio": None,
                "exit_reason": "await_t2_kline"}

    prev_close = t1_close_f
    ride_days = 0
    tier1 = float(LUH_BT_TP_CLOSE_TIER1)
    tier2 = float(LUH_BT_TP_CLOSE_TIER2)
    stop = float(LUH_BT_STOP_LOSS)
    strong_to = float(LUH_BT_STRONG_BOARD_TURNOVER)

    for i, bar in enumerate(exit_bars):
        sell_date = str(bar["date"]).strip()[:10]
        ohlc = _bar_ohlc_dict(bar)
        close_f = float(bar["close"])
        turnover = float(bar.get("turnover") or 0.0)
        close_pnl = (close_f - buy_price) / buy_price if buy_price > 0 else 0.0

        if i == 0:
            if is_limit_up_close(close_f, prev_close, code) and turnover >= strong_to:
                ride_days += 1
                prev_close = close_f
                continue
            pnl = (close_f - buy_price) / buy_price
            return {**base_out, "status": ORDER_STATUS_CLOSED, "buy_price": buy_price,
                    "sell_date": sell_date, "sell_price": close_f, "hold_days": 2,
                    "pnl_ratio": pnl, "exit_reason": "t2_close_exit", "ride_days": ride_days,
                    "t2_date": sell_date}

        if is_one_word_limit_up(ohlc, prev_close, code):
            ride_days += 1
            prev_close = close_f
            continue
        if is_one_word_limit_down(ohlc, prev_close, code):
            ride_days += 1
            prev_close = close_f
            continue

        sell_open = bar.get("open")
        if (
            sell_open is not None
            and np.isfinite(float(sell_open))
            and float(sell_open) > 0
        ):
            sell_price = float(sell_open)
            pnl = (sell_price - buy_price) / buy_price
            return {**base_out, "status": ORDER_STATUS_CLOSED, "buy_price": buy_price,
                    "sell_date": sell_date, "sell_price": sell_price, "hold_days": i + 2,
                    "pnl_ratio": pnl, "exit_reason": "ride_open_exit", "ride_days": ride_days}

        if close_pnl <= -stop:
            pnl = (close_f - buy_price) / buy_price
            return {**base_out, "status": ORDER_STATUS_CLOSED, "buy_price": buy_price,
                    "sell_date": sell_date, "sell_price": close_f, "hold_days": i + 2,
                    "pnl_ratio": pnl, "exit_reason": "stop_loss_close", "ride_days": ride_days}

        if close_pnl >= tier1:
            pnl = (close_f - buy_price) / buy_price
            return {**base_out, "status": ORDER_STATUS_CLOSED, "buy_price": buy_price,
                    "sell_date": sell_date, "sell_price": close_f, "hold_days": i + 2,
                    "pnl_ratio": pnl, "exit_reason": "tp_close_tier1", "ride_days": ride_days}

        if close_pnl >= tier2:
            pnl = (close_f - buy_price) / buy_price
            return {**base_out, "status": ORDER_STATUS_CLOSED, "buy_price": buy_price,
                    "sell_date": sell_date, "sell_price": close_f, "hold_days": i + 2,
                    "pnl_ratio": pnl, "exit_reason": "tp_close_tier2", "ride_days": ride_days}

        ride_days += 1
        prev_close = close_f

    return {**base_out, "status": ORDER_STATUS_HOLDING, "buy_price": buy_price,
            "sell_date": None, "sell_price": None, "hold_days": len(exit_bars) + 1,
            "pnl_ratio": None, "exit_reason": "await_exit_kline", "ride_days": ride_days}


def evaluate_open_open_trade(
    signal_close: float,
    stock_code: str,
    *,
    t1_bar: dict[str, float | None],
    t1_date: str | None,
    post_t1_bars: list[dict[str, Any]] | None = None,
    t2_date: str | None = None,
) -> dict[str, Any]:
    """
    回测专用：T+1 开盘价买入；T+2 起一字涨停续骑，开板日开盘价卖出。
    """
    sig = float(signal_close)
    code = str(stock_code).strip().zfill(6)
    base_out: dict[str, Any] = {
        "t1_date": t1_date,
        "t2_date": t2_date,
        "ride_days": 0,
    }

    if sig <= 0:
        return {
            **base_out,
            "status": ORDER_STATUS_SKIPPED,
            "buy_price": None,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "exit_reason": "invalid_signal_close",
        }

    if is_one_word_limit_up(t1_bar, sig, code):
        return {
            **base_out,
            "status": ORDER_STATUS_SKIPPED,
            "buy_price": None,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "exit_reason": "t1_one_word_limit_up",
        }

    t1_open = t1_bar.get("open")
    if t1_open is None or not np.isfinite(float(t1_open)) or float(t1_open) <= 0:
        return {
            **base_out,
            "status": ORDER_STATUS_HOLDING,
            "buy_price": None,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 0,
            "pnl_ratio": None,
            "exit_reason": "await_t1_kline",
        }

    buy_price = float(t1_open)
    t1_close = t1_bar.get("close")
    if t1_close is None or not np.isfinite(float(t1_close)):
        return {
            **base_out,
            "status": ORDER_STATUS_HOLDING,
            "buy_price": buy_price,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 1,
            "pnl_ratio": None,
            "exit_reason": "await_t1_kline",
        }

    exit_bars = list(post_t1_bars or [])
    if not exit_bars and t2_date:
        return {
            **base_out,
            "status": ORDER_STATUS_HOLDING,
            "buy_price": buy_price,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 1,
            "pnl_ratio": None,
            "exit_reason": "await_exit_kline",
        }
    if not exit_bars:
        return {
            **base_out,
            "status": ORDER_STATUS_HOLDING,
            "buy_price": buy_price,
            "sell_date": None,
            "sell_price": None,
            "hold_days": 1,
            "pnl_ratio": None,
            "exit_reason": "await_t2_kline",
        }

    prev_close = float(t1_close)
    ride_days = 0

    for i, bar in enumerate(exit_bars):
        sell_date = str(bar["date"]).strip()[:10]
        ohlc = _bar_ohlc_dict(bar)

        if is_one_word_limit_up(ohlc, prev_close, code):
            ride_days += 1
            prev_close = float(bar["close"])
            continue

        if is_one_word_limit_down(ohlc, prev_close, code):
            ride_days += 1
            prev_close = float(bar["close"])
            continue

        sell_open = bar.get("open")
        if sell_open is None or not np.isfinite(float(sell_open)) or float(sell_open) <= 0:
            ride_days += 1
            prev_close = float(bar["close"])
            continue

        sell_price = float(sell_open)
        pnl = (sell_price - buy_price) / buy_price
        exit_reason = "t2_open_exit" if i == 0 else "ride_open_exit"
        hold_days = i + 2  # T+1 买，T+2 起为第 1 个可卖日

        return {
            **base_out,
            "status": ORDER_STATUS_CLOSED,
            "buy_price": buy_price,
            "sell_date": sell_date,
            "sell_price": sell_price,
            "hold_days": hold_days,
            "pnl_ratio": pnl,
            "exit_reason": exit_reason,
            "t2_date": sell_date if i == 0 else base_out.get("t2_date"),
            "ride_days": ride_days,
        }

    return {
        **base_out,
        "status": ORDER_STATUS_HOLDING,
        "buy_price": buy_price,
        "sell_date": None,
        "sell_price": None,
        "hold_days": len(exit_bars) + 1,
        "pnl_ratio": None,
        "exit_reason": "await_exit_kline",
        "ride_days": ride_days,
    }


def simulate_limit_up_backtest_trade(
    conn: sqlite3.Connection,
    signal_date: str,
    row: dict[str, Any],
    *,
    legacy: bool | None = None,
) -> dict[str, Any]:
    """单笔打板回测（不写订单表）。"""
    use_legacy = LUH_BACKTEST_LEGACY if legacy is None else legacy
    td = str(signal_date).strip()[:10]
    code = str(row.get("stock_code", "")).strip().zfill(6)
    signal_close = float(row.get("close_price") or 0.0)
    t1_date, t2_date = resolve_t1_t2_dates(td, conn)
    bars = fetch_post_signal_ohlc(conn, code, td)
    post_t1 = fetch_post_t1_bars(conn, code, t1_date) if t1_date else []
    evaluator = evaluate_open_open_trade if use_legacy else evaluate_optimized_backtest_trade
    exit_info = evaluator(
        signal_close,
        code,
        t1_bar=bars["t1"],
        t1_date=t1_date,
        post_t1_bars=post_t1,
        t2_date=t2_date,
    )
    return {
        "signal_date": td,
        "stock_code": code,
        "stock_name": row.get("stock_name"),
        "rank": row.get("rank"),
        "board_score": row.get("board_score"),
        "board_streak": row.get("board_streak"),
        "signal_close": signal_close,
        "t1_date": exit_info.get("t1_date"),
        "t2_date": exit_info.get("t2_date"),
        "buy_price": exit_info.get("buy_price"),
        "sell_date": exit_info.get("sell_date"),
        "sell_price": exit_info.get("sell_price"),
        "hold_days": exit_info.get("hold_days"),
        "ride_days": exit_info.get("ride_days"),
        "pnl_ratio": exit_info.get("pnl_ratio"),
        "status": exit_info.get("status"),
        "exit_reason": exit_info.get("exit_reason"),
    }


def save_luh_backtest_outputs(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    summary: dict[str, Any],
    *,
    trades_path: Path | None = None,
    daily_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, str]:
    tp = trades_path or LUH_BACKTEST_TRADES_CSV
    dp = daily_path or LUH_BACKTEST_DAILY_CSV
    sp = summary_path or LUH_BACKTEST_SUMMARY_JSON
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


def run_limit_up_rolling_backtest(
    conn: sqlite3.Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    top_n: int | None = None,
    include_300: bool = False,
    include_688: bool = False,
    max_scan_stocks: int | None = None,
    legacy: bool | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """滚动回放打板规则并返回 trades / daily / summary。"""
    use_legacy = LUH_BACKTEST_LEGACY if legacy is None else legacy
    top_n = int(top_n if top_n is not None else LUH_TOP_N)
    clear_backtest_filter_cache()

    all_dates = list_kline_trade_dates(conn, start_date, end_date)
    if not all_dates:
        raise ValueError("本地 stock_daily_kline 无可用交易日，请先同步行情。")

    lo = all_dates[0]
    hi = all_dates[-1]
    scan_end = resolve_scan_end_date(all_dates, sell_offset=_BACKTEST_MIN_LAG)
    if not scan_end:
        raise ValueError(
            f"交易日不足 {_BACKTEST_MIN_LAG + 1} 天，无法完成 T+2 起退出模拟。"
        )

    scan_dates = [d for d in all_dates if d <= scan_end]
    if not scan_dates:
        raise ValueError("可扫描信号日为空，请扩大日期区间或同步更多 K 线。")

    if verbose:
        mode = "legacy" if use_legacy else "optimized"
        print(
            f"[打板回测/{mode}] 区间 {lo} ~ {hi}，信号日 {scan_dates[0]} ~ {scan_end}，"
            f"共 {len(scan_dates)} 个交易日 · Top {top_n}",
            flush=True,
        )

    from .backtest_filters import limit_up_codes_on_date

    engine = LimitUpHitStrategy(conn)
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    for i, td in enumerate(scan_dates):
        if verbose and (i % 10 == 0 or i == len(scan_dates) - 1):
            print(f"[打板回测] {i + 1}/{len(scan_dates)} {td} …", flush=True)

        if use_legacy:
            allows, mkt_score, _mom = short_term_market_allows_trading(
                td,
                min_score=LUH_MIN_MARKET_SCORE,
                index_code=LUH_MARKET_INDEX_CODE,
                momentum_days=LUH_MARKET_MOMENTUM_DAYS,
                momentum_min=LUH_MARKET_MOMENTUM_MIN,
            )
            mkt_detail: dict[str, Any] = {}
            mkt_score = int(mkt_score or 0)
        else:
            allows, mkt_detail = backtest_market_allows_trading(conn, td)
            mkt_score = int(mkt_detail.get("market_limit_up_count") or 0)

        if not allows:
            daily_rows.append(
                {
                    "signal_date": td,
                    "market_score": mkt_score,
                    "pick_count": 0,
                    "closed_count": 0,
                    "day_return": None,
                    "cum_nav": None,
                    "day_type": "fused",
                    **({} if use_legacy else {"filter_detail": mkt_detail}),
                }
            )
            continue

        _df, _, _ = engine.scan(
            td,
            top_n=top_n,
            max_scan_stocks=max_scan_stocks,
            include_300=include_300,
            include_688=include_688,
        )
        picks = engine.get_last_persist_rows()
        if not use_legacy and picks:
            lu_codes = limit_up_codes_on_date(conn, td)
            filtered: list[dict[str, Any]] = []
            for row in picks:
                ok, concept_n = backtest_concept_allows(
                    conn, td, str(row.get("stock_code", "")), limit_up_codes=lu_codes
                )
                if ok:
                    row = dict(row)
                    row["concept_limit_up_count"] = concept_n
                    filtered.append(row)
            picks = filtered
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
        skipped_n = 0
        for row in picks:
            row = dict(row)
            tr = simulate_limit_up_backtest_trade(conn, td, row, legacy=use_legacy)
            tr["market_score"] = mkt_score
            if not use_legacy:
                tr["index_ret_20d"] = mkt_detail.get("index_ret_20d")
                tr["market_limit_up_count"] = mkt_detail.get("market_limit_up_count")
                tr["concept_limit_up_count"] = row.get("concept_limit_up_count")
            trade_rows.append(tr)
            if tr.get("status") == ORDER_STATUS_SKIPPED:
                skipped_n += 1
                continue
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
                "skipped_count": skipped_n,
                "day_return": day_ret,
                "cum_nav": None,
                "day_type": "signal" if closed_n > 0 else "empty",
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
        "backtest_mode": "legacy" if use_legacy else "optimized",
        "entry_rule": (
            "T+1开盘价买入（非一字涨停）"
            if use_legacy
            else "T+1非一字开盘价+滑点买入（信号日已涨停）"
        ),
        "exit_rule": (
            "T+2起一字涨停续骑，开板日开盘价卖出"
            if use_legacy
            else "收盘止盈10%/6%+T+2弱板收盘卖+强板续骑"
        ),
        "min_market_score": LUH_MIN_MARKET_SCORE,
        "market_index": LUH_MARKET_INDEX_CODE,
        "include_300": include_300,
        "include_688": include_688,
        "max_scan_stocks": max_scan_stocks,
        "calendar_days_scanned": len(scan_dates),
    }
    if not use_legacy:
        meta.update(
            {
                "bt_min_index_ret_20d": LUH_BT_MIN_INDEX_RET,
                "bt_min_market_limit_up": LUH_BT_MIN_MARKET_LIMIT_UP,
                "bt_min_concept_limit_up": LUH_BT_MIN_CONCEPT_LIMIT_UP,
                "bt_slippage": LUH_BT_SLIPPAGE,
            }
        )
    summary = summarize_backtest(trades_df, daily_df, meta=meta)
    paths = save_luh_backtest_outputs(trades_df, daily_df, summary)

    if verbose:
        print(
            "[打板回测] 完成:",
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
