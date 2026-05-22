# -*- coding: utf-8 -*-
"""短线选股跑批：扫描、落库、写 short_today.json。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DB_PATH
from src.database import get_connection, init_db, insert_system_log

from .config import (
    SHORT_HOLD_PLAN,
    SHORT_HOLDING_DAYS,
    SHORT_SELL_OFFSET,
    SHORT_STOP_LOSS_RATIO,
    SHORT_TODAY_JSON,
    SHORT_TOP_N,
)
from .db import (
    delete_short_orders_for_buy_date,
    delete_short_selections_for_date,
    ensure_short_term_tables,
    insert_short_daily_selections,
    load_short_orders_for_buy_date,
    mark_selections_executed_for_buy_date,
    short_selection_exists,
)
from .execution import refresh_holding_short_orders, sync_short_orders_for_signal_day
from .review_prices import auto_fill_review_prices
from .strategy import ShortTermRuleStrategy


def run_short_daily_pipeline(
    trade_date: str | None = None,
    *,
    force: bool = False,
    top_n: int | None = None,
    max_scan_stocks: int | None = None,
    include_300: bool = False,
    include_688: bool = False,
    write_json: bool = True,
    skip_dingtalk: bool = False,
) -> dict[str, Any]:
    """
    执行短线规则扫描并写入 ``short_daily_selections``。

    返回摘要 dict（trade_date、count、market_score、skipped 等）。
    """
    init_db(DB_PATH)
    top_n = int(top_n if top_n is not None else SHORT_TOP_N)

    with get_connection(DB_PATH) as conn:
        ensure_short_term_tables(conn)

        engine = ShortTermRuleStrategy(conn)
        df, td, mkt_score = engine.scan(
            trade_date,
            top_n=top_n,
            max_scan_stocks=max_scan_stocks,
            include_300=include_300,
            include_688=include_688,
        )
        if not td:
            return {
                "ok": False,
                "error": "本地 stock_daily_kline 无可用交易日",
                "count": 0,
            }

        if not force and short_selection_exists(conn, td):
            return {
                "ok": True,
                "skipped": True,
                "trade_date": td,
                "market_score": mkt_score,
                "count": 0,
                "message": f"{td} 已有短线记录，使用 --force 覆盖",
            }

        rows = engine.get_last_persist_rows()
        n_written = 0

        execution_summary: dict[str, Any] = {}
        try:
            conn.execute("BEGIN IMMEDIATE")
            if force:
                delete_short_selections_for_date(conn, td, commit=False)
                delete_short_orders_for_buy_date(conn, td, commit=False)
            if rows:
                execution_summary = sync_short_orders_for_signal_day(
                    conn, td, rows, commit=False
                )
                n_written = insert_short_daily_selections(
                    conn, td, rows, commit=False
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        mark_selections_executed_for_buy_date(conn, td, commit=True)
        review_fill = auto_fill_review_prices(conn, td, commit=False)
        review_fill_all = auto_fill_review_prices(conn, None, commit=True)
        holding_refreshed = refresh_holding_short_orders(conn, commit=True)
        orders_db = load_short_orders_for_buy_date(conn, td)

        summary = {
            "ok": True,
            "skipped": False,
            "trade_date": td,
            "market_score": mkt_score,
            "count": n_written,
            "review_prices_updated": review_fill.get("rows", 0),
            "review_prices_fill_all": review_fill_all,
            "holding_orders_refreshed": holding_refreshed,
            "holding_days": SHORT_HOLDING_DAYS,
            "hold_plan": SHORT_HOLD_PLAN,
            "sell_offset": SHORT_SELL_OFFSET,
            "stop_loss_ratio": SHORT_STOP_LOSS_RATIO,
            "top_n": top_n,
            "include_300": include_300,
            "include_688": include_688,
            "execution": execution_summary,
            "orders": orders_db,
            "signals": df.to_dict(orient="records") if not df.empty else [],
        }

        if write_json:
            _write_short_today_json(summary, conn=conn)

        ding_ok = False
        if not skip_dingtalk and n_written > 0:
            from .dingtalk import maybe_push_short_selections

            ding_ok = maybe_push_short_selections(td, market_score=mkt_score)
        summary["dingtalk_pushed"] = ding_ok

        insert_system_log(
            "short_daily",
            "success" if n_written or mkt_score else "info",
            json.dumps(
                {
                    "trade_date": td,
                    "written": n_written,
                    "market_score": mkt_score,
                    "force": force,
                    "include_300": include_300,
                    "include_688": include_688,
                },
                ensure_ascii=False,
            ),
            None,
        )
        return summary


def _write_short_today_json(
    summary: dict[str, Any],
    *,
    conn: Any | None = None,
) -> None:
    path = Path(SHORT_TODAY_JSON)
    orders = summary.get("orders") or []
    if not orders and conn is not None and summary.get("trade_date"):
        orders = load_short_orders_for_buy_date(conn, str(summary["trade_date"]))

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": summary.get("trade_date"),
        "market_score": summary.get("market_score"),
        "holding_days": summary.get("holding_days"),
        "hold_plan": summary.get("hold_plan") or SHORT_HOLD_PLAN,
        "sell_offset": summary.get("sell_offset", SHORT_SELL_OFFSET),
        "stop_loss_ratio": summary.get("stop_loss_ratio", SHORT_STOP_LOSS_RATIO),
        "count": summary.get("count"),
        "signals": summary.get("signals") or [],
        "orders": orders,
        "execution": summary.get("execution") or {},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
