# -*- coding: utf-8 -*-
"""打板选股跑批：扫描、落库、写 limit_up_today.json。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DB_PATH
from src.database import get_connection, init_db, insert_system_log

from .config import (
    LUH_HOLDING_DAYS,
    LUH_HOLD_PLAN,
    LUH_SELL_OFFSET,
    LUH_TODAY_JSON,
    LUH_TOP_N,
)
from .db import (
    delete_luh_orders_for_buy_date,
    delete_luh_selections_for_date,
    ensure_luh_tables,
    insert_luh_daily_selections,
    load_luh_orders_for_buy_date,
    luh_selection_exists,
)
from .execution import sync_luh_orders_for_signal_day
from .review_prices import auto_fill_review_prices
from .strategy import LimitUpHitStrategy


def run_limit_up_daily_pipeline(
    trade_date: str | None = None,
    *,
    force: bool = False,
    top_n: int | None = None,
    max_scan_stocks: int | None = None,
    include_300: bool = False,
    include_688: bool = False,
    write_json: bool = True,
) -> dict[str, Any]:
    """执行打板扫描并写入 ``luh_daily_selections`` / ``luh_order_tracker``。"""
    init_db(DB_PATH)
    top_n = int(top_n if top_n is not None else LUH_TOP_N)

    with get_connection(DB_PATH) as conn:
        ensure_luh_tables(conn)

        engine = LimitUpHitStrategy(conn)
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

        if not force and luh_selection_exists(conn, td):
            return {
                "ok": True,
                "skipped": True,
                "trade_date": td,
                "market_score": mkt_score,
                "count": 0,
                "message": f"{td} 已有打板记录，使用 --force 覆盖",
            }

        rows = engine.get_last_persist_rows()
        n_written = 0
        execution_summary: dict[str, Any] = {}

        try:
            conn.execute("BEGIN IMMEDIATE")
            if force:
                delete_luh_selections_for_date(conn, td, commit=False)
                delete_luh_orders_for_buy_date(conn, td, commit=False)
            if rows:
                execution_summary = sync_luh_orders_for_signal_day(
                    conn, td, rows, commit=False
                )
                n_written = insert_luh_daily_selections(conn, td, rows, commit=False)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        auto_fill_review_prices(conn, td, commit=False)
        auto_fill_review_prices(conn, None, commit=True)
        orders_db = load_luh_orders_for_buy_date(conn, td)

        summary = {
            "ok": True,
            "skipped": False,
            "trade_date": td,
            "market_score": mkt_score,
            "count": n_written,
            "holding_days": LUH_HOLDING_DAYS,
            "hold_plan": LUH_HOLD_PLAN,
            "sell_offset": LUH_SELL_OFFSET,
            "top_n": top_n,
            "include_300": include_300,
            "include_688": include_688,
            "execution": execution_summary,
            "orders": orders_db,
            "signals": df.to_dict(orient="records") if not df.empty else [],
        }

        if write_json:
            _write_limit_up_today_json(summary, conn=conn)

        insert_system_log(
            "limit_up_daily",
            "success" if n_written or mkt_score else "info",
            json.dumps(
                {
                    "trade_date": td,
                    "written": n_written,
                    "market_score": mkt_score,
                    "force": force,
                },
                ensure_ascii=False,
            ),
            None,
        )
        return summary


def _write_limit_up_today_json(
    summary: dict[str, Any],
    *,
    conn: Any | None = None,
) -> None:
    path = Path(LUH_TODAY_JSON)
    orders = summary.get("orders") or []
    if not orders and conn is not None and summary.get("trade_date"):
        orders = load_luh_orders_for_buy_date(conn, str(summary["trade_date"]))

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": summary.get("trade_date"),
        "market_score": summary.get("market_score"),
        "holding_days": summary.get("holding_days"),
        "hold_plan": summary.get("hold_plan") or LUH_HOLD_PLAN,
        "sell_offset": summary.get("sell_offset", LUH_SELL_OFFSET),
        "count": summary.get("count"),
        "signals": summary.get("signals") or [],
        "orders": orders,
        "execution": summary.get("execution") or {},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
