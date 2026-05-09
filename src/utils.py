"""日期与交易日工具。"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd


def to_date_str(d: date | datetime | str) -> str:
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    return d.isoformat()


def latest_trade_date_from_series(dates: pd.Series) -> Optional[str]:
    if dates is None or len(dates) == 0:
        return None
    s = pd.to_datetime(dates).dt.strftime("%Y-%m-%d")
    return str(s.max())


def add_calendar_days(d: str, n: int) -> str:
    base = datetime.strptime(d[:10], "%Y-%m-%d").date()
    return (base + timedelta(days=n)).isoformat()


def next_trade_day_after(d: str) -> str | None:
    """
    给定YYYYMMDD或YYYY-MM-DD的「当前选股所属交易日」d，返回其后的下一个A股交易日（YYYY-MM-DD）。

    优先使用 akshare 新浪交易日历（含休市）；网络异常或日历无更晚日期时，退回为「跳过周六日」的近似工作日。
    d 无法解析为日期时返回 None。
    """
    try:
        base = datetime.strptime(d[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

    try:
        from akshare.tool.trade_date_hist import tool_trade_date_hist_sina

        df = tool_trade_date_hist_sina()
        trade_dates = sorted(df["trade_date"].tolist())
        for td in trade_dates:
            if isinstance(td, date) and td > base:
                return td.isoformat()
    except Exception:
        pass

    nxt = base + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.isoformat()
