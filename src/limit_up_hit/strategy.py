# -*- coding: utf-8 -*-
"""
打板选股（纯日线 K 线）。

信号日 T：收盘涨停且封板质量达标。
排序：连板高度、封板强度、换手/成交额。
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from src.factor_calculator import is_bar_limit_up, is_bar_suspended
from src.market_regime import short_term_market_allows_trading
from src.short_term.board_filter import board_allowed

from .config import (
    LUH_ENTITY_RATIO_MIN,
    LUH_EXCLUDE_BJ,
    LUH_EXCLUDE_ST,
    LUH_HOLD_PLAN,
    LUH_MARKET_INDEX_CODE,
    LUH_MARKET_MOMENTUM_DAYS,
    LUH_MARKET_MOMENTUM_MIN,
    LUH_MIN_AMOUNT,
    LUH_MIN_HISTORY_BARS,
    LUH_MIN_MARKET_SCORE,
    LUH_MIN_TURNOVER,
    LUH_SEAL_CLOSE_HIGH_MIN,
    LUH_TOP_N,
    LUH_TURNOVER_GOLDEN_MAX,
    LUH_TURNOVER_GOLDEN_MIN,
)

RESULT_COLUMNS = [
    "股票代码",
    "股票名称",
    "信号日",
    "收盘价",
    "连板数",
    "封板强度",
    "换手率",
    "打板得分",
    "T+1打板条件",
    "T+2止盈",
    "T+2止损",
    "操作要点",
    "持仓计划",
]


def _is_st_name(name: str) -> bool:
    n = str(name or "").strip().upper()
    return "ST" in n or "*ST" in n


def _exclude_code_prefix(code: str) -> bool:
    c = str(code).strip().zfill(6)
    if LUH_EXCLUDE_BJ and c.startswith(("8", "4", "83", "87", "92")):
        return True
    return False


def count_board_streak(
    close_arr: pd.Series,
    code: str,
    *,
    end_idx: int | None = None,
) -> int:
    """截至 end_idx（含）的连续涨停天数。"""
    idx = int(end_idx if end_idx is not None else len(close_arr) - 1)
    streak = 0
    for i in range(idx, 0, -1):
        c_now = float(close_arr.iloc[i])
        c_prev = float(close_arr.iloc[i - 1])
        if is_bar_limit_up(c_now, c_prev, code):
            streak += 1
        else:
            break
    return streak


def seal_strength(close: float, high: float, low: float) -> float:
    """封板强度 0~1：收盘贴近最高价且实体饱满。"""
    if not all(np.isfinite(x) for x in (close, high, low)) or high <= low:
        return 0.0
    close_high = close / high if high > 0 else 0.0
    body = abs(close - low) / (high - low)
    return float(min(1.0, close_high * 0.6 + body * 0.4))


def compute_board_score(
    *,
    streak: int,
    seal: float,
    turnover: float,
    amount: float,
) -> float:
    """打板综合得分。"""
    streak_score = min(100.0, 40.0 + streak * 25.0)
    seal_score = seal * 100.0
    to_score = 0.0
    if LUH_TURNOVER_GOLDEN_MIN <= turnover <= LUH_TURNOVER_GOLDEN_MAX:
        mid = (LUH_TURNOVER_GOLDEN_MIN + LUH_TURNOVER_GOLDEN_MAX) / 2.0
        to_score = max(0.0, 100.0 - abs(turnover - mid) * 4.0)
    amt_score = min(100.0, amount / LUH_MIN_AMOUNT * 40.0)
    return float(
        streak_score * 0.35
        + seal_score * 0.30
        + to_score * 0.20
        + amt_score * 0.15
    )


def board_advice_text(streak: int, seal: float, turnover: float) -> str:
    if streak >= 3:
        return "⚠️ 高位连板：仅小仓博弈，T+1 一字板无法买入则放弃"
    if seal < 0.95:
        return "⚠️ 封板偏弱：次日炸板概率高，严格执行 T+2 止损"
    if turnover > 25:
        return "⚠️ 换手偏高：注意主力出货，不追一字"
    return "✅ 首板/弱转强：T+1 非一字再次封板以涨停价模拟买入"


class LimitUpHitStrategy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._last_persist_rows: list[dict[str, Any]] = []

    def get_last_persist_rows(self) -> list[dict[str, Any]]:
        return list(self._last_persist_rows)

    def scan(
        self,
        target_date: str | None = None,
        *,
        top_n: int | None = None,
        max_scan_stocks: int | None = None,
        include_300: bool = False,
        include_688: bool = False,
    ) -> tuple[pd.DataFrame, str, int]:
        top_n = int(top_n if top_n is not None else LUH_TOP_N)
        cur = self.conn.cursor()
        if target_date is None:
            res_date = cur.execute(
                "SELECT MAX(date) FROM stock_daily_kline"
            ).fetchone()
            if res_date is None or res_date[0] is None:
                return (pd.DataFrame(columns=RESULT_COLUMNS), "", 0)
            target_date = str(res_date[0]).strip()[:10]
        else:
            target_date = str(target_date).strip()[:10]

        allows, mkt_score, _mom = short_term_market_allows_trading(
            target_date,
            min_score=LUH_MIN_MARKET_SCORE,
            index_code=LUH_MARKET_INDEX_CODE,
            momentum_days=LUH_MARKET_MOMENTUM_DAYS,
            momentum_min=LUH_MARKET_MOMENTUM_MIN,
        )
        if not allows:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        cur_cols_query = cur.execute("PRAGMA table_info(stock_daily_kline)").fetchall()
        cur_cols = {str(row[1]) for row in cur_cols_query}
        has_turnover_col = "turnover_rate" in cur_cols
        amount_col = "volume * close" if "amount" not in cur_cols else "amount"
        turnover_col = "turnover_rate" if has_turnover_col else "NULL"

        df_target = pd.read_sql_query(
            f"""
            SELECT stock_code, stock_name, close, volume,
                   {amount_col} AS amount, {turnover_col} AS turnover
            FROM stock_daily_kline WHERE date = ?
            """,
            self.conn,
            params=[target_date],
        )
        if df_target.empty:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        df_target["stock_code"] = (
            df_target["stock_code"].astype(str).str.strip().str.zfill(6)
        )
        df_target = df_target[
            df_target["stock_code"].map(
                lambda c: board_allowed(
                    c, include_300=include_300, include_688=include_688
                )
            )
        ].copy()
        if max_scan_stocks is not None and int(max_scan_stocks) > 0:
            df_target = df_target.head(int(max_scan_stocks))

        panel_df = pd.read_sql_query(
            """
            SELECT stock_code, date, open, high, low, close, volume
            FROM stock_daily_kline WHERE date <= ?
            ORDER BY stock_code, date ASC
            """,
            self.conn,
            params=[target_date],
        )
        if panel_df.empty:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        panel_df["stock_code"] = (
            panel_df["stock_code"].astype(str).str.strip().str.zfill(6)
        )

        target_meta_map: dict[str, dict[str, Any]] = {}
        for _, r in df_target.iterrows():
            amt = float(r["amount"]) if r["amount"] is not None else 0.0
            to_val = float(r["turnover"]) if r["turnover"] is not None else 0.0
            target_meta_map[r["stock_code"]] = {
                "name": str(r["stock_name"] or "未知").strip(),
                "close": float(r["close"]),
                "amount": amt,
                "turnover": to_val,
            }

        candidates: list[dict[str, Any]] = []
        min_bars = int(LUH_MIN_HISTORY_BARS)

        for code, sub_df in panel_df.groupby("stock_code", sort=False):
            if code not in target_meta_map:
                continue
            meta = target_meta_map[code]
            name = meta["name"]

            if _exclude_code_prefix(code):
                continue
            if LUH_EXCLUDE_ST and _is_st_name(name):
                continue

            amt = float(meta["amount"])
            tr = float(meta["turnover"]) if meta["turnover"] is not None else 0.0
            if amt < float(LUH_MIN_AMOUNT) and tr < float(LUH_MIN_TURNOVER):
                continue
            if has_turnover_col and (
                tr < float(LUH_TURNOVER_GOLDEN_MIN)
                or tr > float(LUH_TURNOVER_GOLDEN_MAX)
            ):
                continue

            if len(sub_df) < min_bars:
                continue

            last_row = sub_df.iloc[-1]
            if str(last_row["date"])[:10] != target_date:
                continue
            if is_bar_suspended(last_row):
                continue

            open_arr = pd.to_numeric(sub_df["open"], errors="coerce")
            close_arr = pd.to_numeric(sub_df["close"], errors="coerce")
            high_arr = pd.to_numeric(sub_df["high"], errors="coerce")
            low_arr = pd.to_numeric(sub_df["low"], errors="coerce")

            prev_close = float(close_arr.iloc[-2])
            c_close = float(close_arr.iloc[-1])
            c_open = float(open_arr.iloc[-1])
            c_high = float(high_arr.iloc[-1])
            c_low = float(low_arr.iloc[-1])

            if not is_bar_limit_up(c_close, prev_close, code):
                continue

            if c_high <= 0 or c_close / c_high < float(LUH_SEAL_CLOSE_HIGH_MIN):
                continue

            body_ratio = (
                abs(c_close - c_open) / (c_high - c_low + 1e-6)
                if c_high > c_low
                else 0.0
            )
            if body_ratio < float(LUH_ENTITY_RATIO_MIN):
                continue

            streak = count_board_streak(close_arr, code)
            seal = seal_strength(c_close, c_high, c_low)
            score = compute_board_score(
                streak=streak,
                seal=seal,
                turnover=tr,
                amount=amt,
            )
            chg = (c_close - prev_close) / prev_close if prev_close > 0 else 0.0

            candidates.append(
                {
                    "stock_code": code,
                    "stock_name": name,
                    "trade_date": target_date,
                    "close_price": c_close,
                    "pct_change": chg,
                    "board_streak": streak,
                    "seal_strength": seal,
                    "turnover": tr,
                    "amount": amt,
                    "board_score": score,
                    "entity_ratio": body_ratio,
                    "advice_text": board_advice_text(streak, seal, tr),
                    "hold_plan": LUH_HOLD_PLAN,
                    "detail": {
                        "board_streak": streak,
                        "seal_strength": seal,
                        "entity_ratio": body_ratio,
                        "turnover": tr,
                        "amount": amt,
                    },
                }
            )

        candidates.sort(
            key=lambda x: (float(x["board_score"]), float(x["board_streak"])),
            reverse=True,
        )
        picked = candidates[:top_n]

        display_rows: list[dict[str, Any]] = []
        persist_rows: list[dict[str, Any]] = []
        for rank, item in enumerate(picked, start=1):
            item["rank"] = rank
            persist_rows.append(dict(item))
            display_rows.append(
                {
                    "股票代码": item["stock_code"],
                    "股票名称": item["stock_name"],
                    "信号日": target_date,
                    "收盘价": round(item["close_price"], 2),
                    "连板数": item["board_streak"],
                    "封板强度": round(item["seal_strength"], 3),
                    "换手率": round(item["turnover"], 2),
                    "打板得分": round(item["board_score"], 1),
                    "T+1打板条件": "非一字 + T+1 收盘涨停 → 涨停价买入",
                    "T+2止盈": "冲高 10%/6% 双阶梯",
                    "T+2止损": "收盘 -7%",
                    "操作要点": item["advice_text"],
                    "持仓计划": item["hold_plan"],
                }
            )

        self._last_persist_rows = persist_rows
        df_out = pd.DataFrame(display_rows, columns=RESULT_COLUMNS)
        return df_out, target_date, mkt_score
