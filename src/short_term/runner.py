# -*- coding: utf-8 -*-
"""短线选股跑批：扫描、落库、写 short_today.json。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DB_PATH
from src.database import get_connection, init_db, insert_system_log

from .config import SHORT_HOLDING_DAYS, SHORT_TODAY_JSON, SHORT_TOP_N
from .db import (
    delete_short_selections_for_date,
    ensure_short_term_tables,
    insert_short_daily_selections,
    refresh_short_review_prices,
    short_selection_exists,
)
from .strategy import ShortTermRuleStrategy


def run_short_daily_pipeline(
    trade_date: str | None = None,
    *,
    force: bool = False,
    top_n: int | None = None,
    max_scan_stocks: int | None = None,
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

        try:
            conn.execute("BEGIN IMMEDIATE")
            if force:
                delete_short_selections_for_date(conn, td, commit=False)
            if rows:
                n_written = insert_short_daily_selections(
                    conn, td, rows, commit=False
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        review_updated = refresh_short_review_prices(conn, td, commit=True)

        summary = {
            "ok": True,
            "skipped": False,
            "trade_date": td,
            "market_score": mkt_score,
            "count": n_written,
            "review_prices_updated": review_updated,
            "holding_days": SHORT_HOLDING_DAYS,
            "top_n": top_n,
            "signals": df.to_dict(orient="records") if not df.empty else [],
        }

        if write_json:
            _write_short_today_json(summary)

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
                },
                ensure_ascii=False,
            ),
            None,
        )
        return summary


def _write_short_today_json(summary: dict[str, Any]) -> None:
    path = Path(SHORT_TODAY_JSON)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": summary.get("trade_date"),
        "market_score": summary.get("market_score"),
        "holding_days": summary.get("holding_days"),
        "count": summary.get("count"),
        "signals": summary.get("signals") or [],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
