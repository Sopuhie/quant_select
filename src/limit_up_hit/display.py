# -*- coding: utf-8 -*-
"""打板模块表格列名中文化（界面展示）。"""
from __future__ import annotations

from typing import Any

import pandas as pd

_SELECTION_COL_ZH: dict[str, str] = {
    "trade_date": "信号日",
    "stock_code": "股票代码",
    "stock_name": "股票名称",
    "close_price": "收盘价",
    "pct_change": "日涨幅",
    "board_streak": "连板数",
    "seal_strength": "封板强度",
    "turnover": "换手率(%)",
    "board_score": "打板得分",
    "is_executed": "已执行",
    "rank": "排名",
    "hold_plan": "持仓计划",
    "advice_text": "操作建议",
    "detail": "明细",
    "detail_json": "明细",
    "t1_open": "T+1开盘",
    "t1_close": "T+1收盘",
    "t2_open": "T+2开盘",
    "t2_close": "T+2收盘",
    "created_at": "创建时间",
}

_ORDER_COL_ZH: dict[str, str] = {
    "stock_code": "股票代码",
    "stock_name": "股票名称",
    "buy_date": "买入日",
    "buy_price": "买入价",
    "sell_date": "卖出日",
    "sell_price": "卖出价",
    "hold_days": "持有天数",
    "pnl_ratio": "盈亏比例",
    "status": "状态",
    "stop_loss_triggered": "止损触发",
    "signal_rank": "排名",
    "board_score": "打板得分",
    "board_streak": "连板数",
    "exit_reason": "退出原因",
    "hold_plan": "持仓计划",
    "order_id": "订单号",
    "created_at": "创建时间",
    "updated_at": "更新时间",
}

_TRADE_COL_ZH: dict[str, str] = {
    "signal_date": "信号日",
    "t1_date": "T+1日期",
    "t2_date": "T+2日期",
    "stock_code": "股票代码",
    "stock_name": "股票名称",
    "rank": "排名",
    "board_score": "打板得分",
    "board_streak": "连板数",
    "signal_close": "信号收盘价",
    "buy_price": "买入价(T+1开)",
    "sell_price": "卖出价(开板日开)",
    "sell_date": "卖出日",
    "hold_days": "持有天数",
    "pnl_ratio": "盈亏比例",
    "status": "状态",
    "exit_reason": "退出原因",
    "ride_days": "骑乘天数",
    "market_score": "大盘环境分",
}

_STATUS_ZH: dict[str, str] = {
    "HOLDING": "持有中",
    "CLOSED": "已平仓",
    "SKIPPED": "已跳过",
}

_EXIT_REASON_ZH: dict[str, str] = {
    "t1_one_word_limit_up": "T+1一字涨停无法买入",
    "await_t1_kline": "待T+1 K线",
    "await_t2_kline": "待T+2 K线",
    "t2_close_exit": "T+2收盘卖出（弱板）",
    "tp_close_tier1": "收盘止盈10%",
    "tp_close_tier2": "收盘止盈6%",
    "stop_loss_close": "收盘止损",
    "t1_not_limit_up_close": "T+1未收盘涨停",
    "t1_intraday_open_board": "T+1盘中开板过多",
    "ride_open_exit": "连板骑乘后开板卖出",
    "await_exit_kline": "待后续K线（仍在骑乘）",
    "invalid_signal_close": "信号收盘价无效",
    "invalid_t1_open": "T+1开盘价无效",
    "invalid_t2_open": "T+2开盘价无效",
}


def _format_pct_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—")


def format_luh_dataframe(
    df: pd.DataFrame,
    *,
    table: str = "selection",
) -> pd.DataFrame:
    """将打板相关 DataFrame 列名与部分字段转为中文展示。"""
    if df is None or df.empty:
        return df

    out = df.copy()
    mapping = {
        "selection": _SELECTION_COL_ZH,
        "order": _ORDER_COL_ZH,
        "trade": _TRADE_COL_ZH,
    }.get(table, _SELECTION_COL_ZH)

    if "status" in out.columns:
        out["status"] = (
            out["status"]
            .astype(str)
            .str.upper()
            .map(lambda x: _STATUS_ZH.get(x, x))
        )
    if "exit_reason" in out.columns:
        out["exit_reason"] = out["exit_reason"].astype(str).map(
            lambda x: _EXIT_REASON_ZH.get(x, x)
        )
    if "is_executed" in out.columns:
        out["is_executed"] = out["is_executed"].map(
            lambda x: "是" if int(x or 0) == 1 else "否"
        )
    if "stop_loss_triggered" in out.columns:
        out["stop_loss_triggered"] = out["stop_loss_triggered"].map(
            lambda x: "是" if int(x or 0) == 1 else "否"
        )
    for col in ("pct_change", "pnl_ratio"):
        if col in out.columns:
            out[col] = _format_pct_series(out[col])

    rename = {k: v for k, v in mapping.items() if k in out.columns}
    return out.rename(columns=rename)


def format_luh_records(
    records: list[dict[str, Any]],
    *,
    table: str = "selection",
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return format_luh_dataframe(pd.DataFrame(records), table=table)
