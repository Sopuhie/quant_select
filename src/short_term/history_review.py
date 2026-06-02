# -*- coding: utf-8 -*-
"""短线历史选股记录复盘：查询、汇总与展示用 DataFrame 构建。"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd

from .db import (
    ensure_short_term_tables,
    load_short_orders_for_buy_date,
    load_short_selections_df,
)
from .review_prices import calc_t1_buy_t2_sell_return
from .rules_doc import CHECK_LABELS
from .trade_guide import build_trade_action_guide


def list_short_selection_trade_dates(conn: sqlite3.Connection) -> list[str]:
    """库内已有短线记录的信号日列表（降序，最新在前）。"""
    ensure_short_term_tables(conn)
    rows = conn.execute(
        """
        SELECT DISTINCT trade_date
        FROM short_daily_selections
        ORDER BY trade_date DESC
        """
    ).fetchall()
    return [str(r[0]).strip()[:10] for r in rows if r[0]]


def load_short_history_calendar(conn: sqlite3.Connection) -> pd.DataFrame:
    """各信号日汇总（入选数、得分、模拟盘平仓与盈亏）。"""
    ensure_short_term_tables(conn)
    return pd.read_sql_query(
        """
        SELECT
            s.trade_date AS 信号日,
            COUNT(*) AS 入选数,
            ROUND(AVG(COALESCE(s.final_score, s.rule_score)), 2) AS 平均规则得分,
            SUM(CASE WHEN COALESCE(s.is_executed, 0) = 1 THEN 1 ELSE 0 END) AS 已执行数,
            SUM(CASE WHEN o.status = 'CLOSED' THEN 1 ELSE 0 END) AS 已平仓笔数,
            SUM(CASE WHEN o.status = 'HOLDING' THEN 1 ELSE 0 END) AS 持仓中笔数,
            ROUND(AVG(CASE WHEN o.pnl_ratio IS NOT NULL THEN o.pnl_ratio END) * 100, 2) AS 平均盈亏pct,
            SUM(CASE WHEN s.t1_close IS NOT NULL THEN 1 ELSE 0 END) AS T1收盘已填,
            SUM(CASE WHEN s.t2_close IS NOT NULL THEN 1 ELSE 0 END) AS T2收盘已填
        FROM short_daily_selections s
        LEFT JOIN short_order_tracker o
          ON s.trade_date = o.buy_date AND s.stock_code = o.stock_code
        GROUP BY s.trade_date
        ORDER BY s.trade_date DESC
        """,
        conn,
    )


def load_short_checks_recap_df(
    conn: sqlite3.Connection,
    trade_date: str,
) -> pd.DataFrame:
    """入选明细：各票规则校验项（复盘对照）。"""
    ensure_short_term_tables(conn)
    td = str(trade_date).strip()[:10]
    rows = conn.execute(
        """
        SELECT stock_code, stock_name, rank, detail_json
        FROM short_daily_selections
        WHERE trade_date = ?
        ORDER BY COALESCE(rank, 999), stock_code
        """,
        (td,),
    ).fetchall()

    recap: list[dict[str, Any]] = []
    for code, name, rank, dj in rows:
        checks: dict[str, bool] = {}
        if dj:
            try:
                payload = json.loads(dj)
                checks = payload.get("checks") or {}
            except json.JSONDecodeError:
                checks = {}
        row_out: dict[str, Any] = {
            "排名": rank,
            "代码": str(code).zfill(6),
            "名称": name,
        }
        for key, label in CHECK_LABELS.items():
            if not checks:
                row_out[label] = "—"
            else:
                row_out[label] = "✓" if checks.get(key) else "✗"
        recap.append(row_out)
    return pd.DataFrame(recap)


def load_short_execution_recap_df(
    conn: sqlite3.Connection,
    trade_date: str,
) -> pd.DataFrame:
    """从 detail_json.execution 与订单表合并的模拟执行复盘。"""
    ensure_short_term_tables(conn)
    td = str(trade_date).strip()[:10]
    orders = {str(o["stock_code"]).zfill(6): o for o in load_short_orders_for_buy_date(conn, td)}
    rows = conn.execute(
        """
        SELECT stock_code, stock_name, rank, close_price,
               t1_open, t1_close, t2_close, detail_json
        FROM short_daily_selections
        WHERE trade_date = ?
        ORDER BY rank
        """,
        (td,),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for code, name, rank, close_px, t1_o, t1_c, t2_c, dj in rows:
        code = str(code).zfill(6)
        exec_d: dict[str, Any] = {}
        if dj:
            try:
                exec_d = (json.loads(dj).get("execution") or {})
            except json.JSONDecodeError:
                exec_d = {}
        order = orders.get(code, {})
        pnl = order.get("pnl_ratio")
        if pnl is None:
            pnl = exec_d.get("pnl_ratio")
        pnl_disp = "—"
        if pnl is not None:
            try:
                pnl_disp = f"{float(pnl) * 100:.2f}%"
            except (TypeError, ValueError):
                pnl_disp = str(pnl)

        ret_t1t2 = calc_t1_buy_t2_sell_return(t1_o, t1_c, t2_c)
        ret_t1t2_disp = "—"
        if ret_t1t2 is not None:
            ret_t1t2_disp = f"{float(ret_t1t2) * 100:.2f}%"

        out.append(
            {
                "排名": rank,
                "代码": code,
                "名称": name,
                "买入价": close_px,
                "T1买T2卖涨跌幅": ret_t1t2_disp,
                "订单状态": order.get("status") or exec_d.get("status") or "—",
                "卖出日": order.get("sell_date") or exec_d.get("sell_date") or "—",
                "卖出价": order.get("sell_price") or exec_d.get("sell_price"),
                "盈亏": pnl_disp,
                "触发止损": "是"
                if int(order.get("stop_loss_triggered") or 0)
                else ("否" if order else "—"),
                "退出原因": order.get("exit_reason") or exec_d.get("exit_reason") or "—",
            }
        )
    return pd.DataFrame(out)


def load_short_action_guide_df(
    conn: sqlite3.Connection,
    trade_date: str,
) -> pd.DataFrame:
    """入选票 T+1 买入 / T+2 卖出·持有操作价位（可直接对照挂单）。"""
    ensure_short_term_tables(conn)
    td = str(trade_date).strip()[:10]
    rows = conn.execute(
        """
        SELECT stock_code, stock_name, rank, close_price, detail_json
        FROM short_daily_selections
        WHERE trade_date = ?
        ORDER BY rank
        """,
        (td,),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for code, name, rank, close_px, dj in rows:
        guide: dict[str, Any] | None = None
        if dj:
            try:
                payload = json.loads(dj)
                guide = payload.get("trade_guide")
            except json.JSONDecodeError:
                guide = None
        if not guide and close_px is not None:
            try:
                guide = build_trade_action_guide(float(close_px))
            except (TypeError, ValueError):
                guide = None
        disp = (guide or {}).get("display") or {}
        out.append(
            {
                "排名": rank,
                "代码": str(code).zfill(6),
                "名称": name,
                "信号收盘": close_px,
                "T+1开盘区间": disp.get("T+1开盘区间", "—"),
                "T+1放弃": disp.get("T+1放弃条件", "—"),
                "T+2止盈": disp.get("T+2止盈触发", "—"),
                "T+2止损": disp.get("T+2止损线", "—"),
                "T+2持有": disp.get("T+2持有条件", "—"),
                "操作要点": disp.get("操作要点", "—"),
                "详细说明": (guide or {}).get("summary_text", "—"),
            }
        )
    df = pd.DataFrame(out)
    if not df.empty and "信号收盘" in df.columns:
        df["信号收盘"] = pd.to_numeric(df["信号收盘"], errors="coerce").round(2)
    return df


def enrich_selections_with_action_guide(df: pd.DataFrame) -> pd.DataFrame:
    """为选股表追加操作价位列（无 detail 时按收盘价重算）。"""
    if df.empty:
        return df
    out = df.copy()
    guide_cols = [
        "T+1开盘区间",
        "T+2止盈触发",
        "T+2止损线",
        "T+2持有条件",
        "操作要点",
    ]
    for col in guide_cols:
        if col not in out.columns:
            out[col] = "—"

    close_col = None
    for c in ("信号日收盘价", "close_price", "收盘价"):
        if c in out.columns:
            close_col = c
            break
    if close_col is None:
        return out

    for idx, row in out.iterrows():
        if all(str(row.get(c, "—")) not in ("—", "", "nan") for c in guide_cols[:4]):
            continue
        try:
            px = float(row[close_col])
        except (TypeError, ValueError):
            continue
        disp = build_trade_action_guide(px).get("display") or {}
        for col in guide_cols:
            if str(out.at[idx, col]) in ("—", "", "nan"):
                out.at[idx, col] = disp.get(col, "—")
    return out


def format_selections_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """界面展示用格式化（百分比、小数位）。"""
    if df.empty:
        return df
    out = enrich_selections_with_action_guide(df.copy())
    if "日涨幅" in out.columns:
        out["日涨幅"] = out["日涨幅"].apply(
            lambda x: f"{float(x) * 100:.2f}%"
            if pd.notna(x) and abs(float(x)) < 2
            else (f"{float(x):.2f}%" if pd.notna(x) else "—")
        )
    for pct_col in ("T1日涨幅", "T2日涨幅", "T1买T2卖涨跌幅"):
        if pct_col in out.columns:
            out[pct_col] = out[pct_col].apply(
                lambda x: f"{float(x) * 100:.2f}%"
                if pd.notna(x) and abs(float(x)) < 2
                else (f"{float(x):.2f}%" if pd.notna(x) else "—")
            )
    for col in ("规则得分", "五日量比", "MACD柱改善", "J斜率", "KDJ_J", "MACD柱"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    if "已执行" in out.columns:
        out["已执行"] = out["已执行"].map(
            lambda x: "是" if x in (1, "1", True) else ("否" if x in (0, "0", False) else "—")
        )
    for col in ("信号日收盘价", "T1开盘价", "T1收盘价", "T2开盘价", "T2收盘价"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    order = [
        "信号日",
        "排名",
        "股票代码",
        "股票名称",
        "规则得分",
        "日涨幅",
        "五日量比",
        "MACD柱改善",
        "J斜率",
        "KDJ_J",
        "MACD柱",
        "T+1开盘区间",
        "T+2止盈触发",
        "T+2止损线",
        "T+2持有条件",
        "操作要点",
        "已执行",
        "信号日收盘价",
        "T1开盘价",
        "T1收盘价",
        "T1日涨幅",
        "T2开盘价",
        "T2收盘价",
        "T2日涨幅",
        "T1买T2卖涨跌幅",
        "持仓计划",
        "实盘建议",
    ]
    return out[[c for c in order if c in out.columns]]


def load_short_review_bundle(
    conn: sqlite3.Connection,
    trade_date: str,
    *,
    auto_fill_t1_t2: bool = True,
) -> dict[str, Any]:
    """单日复盘数据包（选股表、校验表、执行表、订单列表）。"""
    td = str(trade_date).strip()[:10]
    fill_stats: dict[str, int] = {}
    if auto_fill_t1_t2:
        from .review_prices import auto_fill_review_prices

        fill_stats = auto_fill_review_prices(conn, td, commit=True)
    sel = load_short_selections_df(conn, td)
    return {
        "trade_date": td,
        "selections": sel,
        "selections_display": format_selections_display_df(sel),
        "action_guide": load_short_action_guide_df(conn, td),
        "count": len(sel),
        "t1_t2_fill": fill_stats,
    }
