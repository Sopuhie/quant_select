# -*- coding: utf-8 -*-
"""
短线规则选股（持有 1 个交易日）。

信号日 T 收盘后筛选 → 计划 T+1 开盘买入 → T+2 开盘卖出（A 股 T+1，不可当日回转）。
与中长线 LightGBM、题材 v2.0 策略独立；全市场面板向量化扫描，避免逐股 SQL 与指标前视。
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from src.factor_calculator import is_bar_limit_up, is_bar_suspended
from src.market_regime import market_environment_allows_trading

from .config import (
    SHORT_EXCLUDE_BJ,
    SHORT_EXCLUDE_NEAR_LIMIT,
    SHORT_EXCLUDE_ST,
    SHORT_HOLDING_DAYS,
    SHORT_KDJ_J_MAX,
    SHORT_KDJ_J_MIN,
    SHORT_KDJ_J_SLOPE_MIN,
    SHORT_MA_FAST,
    SHORT_MA_SLOW,
    SHORT_MAX_5D_RETURN,
    SHORT_MAX_DAY_RETURN,
    SHORT_MIN_DAY_RETURN,
    SHORT_MIN_HISTORY_BARS,
    SHORT_MIN_MARKET_SCORE,
    SHORT_NEAR_LIMIT_PCT,
    SHORT_TOP_N,
    SHORT_VOL_RATIO_1D_MIN,
    SHORT_VOL_RATIO_5D_MIN,
)

RESULT_COLUMNS = [
    "股票代码",
    "股票名称",
    "信号日",
    "收盘价",
    "日涨幅",
    "五日量比",
    "KDJ_J",
    "MACD柱",
    "规则得分",
    "持仓计划",
    "实盘建议",
]

_HOLD_PLAN = (
    f"T 日收盘确认信号 → T+1 开盘买入 → T+{1 + SHORT_HOLDING_DAYS} 开盘卖出"
    f"（持有 {SHORT_HOLDING_DAYS} 个交易日）"
)


def _is_st_name(name: str) -> bool:
    n = str(name or "").strip().upper()
    return "ST" in n or "*ST" in n


def _exclude_code_prefix(code: str) -> bool:
    c = str(code).strip().zfill(6)
    if SHORT_EXCLUDE_BJ and c.startswith(("8", "4")):
        return True
    return False


def _near_limit_up_signal(close: float, prev_close: float, code: str) -> bool:
    if not SHORT_EXCLUDE_NEAR_LIMIT:
        return False
    try:
        c, p = float(close), float(prev_close)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(c) and np.isfinite(p) and p > 0):
        return False
    pct = (c - p) / p
    return pct >= float(SHORT_NEAR_LIMIT_PCT)


class ShortTermRuleStrategy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._last_persist_rows: list[dict[str, Any]] = []

    @staticmethod
    def _rule_score(
        chg: float, vr5: float, bar_now: float, bar_prev: float, j_slope: float
    ) -> float:
        bar_delta = bar_now - bar_prev
        return float(
            40.0 * min(chg / 0.05, 1.5)
            + 25.0 * min(vr5 / 3.0, 2.0)
            + 20.0 * min(max(bar_delta, 0) * 50.0, 2.0)
            + 15.0 * min(max(j_slope, 0) / 15.0, 2.0)
        )

    @staticmethod
    def advice_text(j_now: float, chg: float) -> str:
        if j_now >= 88:
            return "⚠️ J 偏高：仅适合极小仓博弈，严格执行 T+2 开盘离场"
        if chg >= 0.06:
            return "⚠️ 当日涨幅偏大：次日高开若不及预期应果断止损"
        return "✅ 短线共振成立：次日开盘按计划买入，T+2 开盘无条件卖出"

    def scan(
        self,
        target_date: str | None = None,
        *,
        top_n: int | None = None,
        max_scan_stocks: int | None = None,
    ) -> tuple[pd.DataFrame, str, int]:
        top_n = int(top_n if top_n is not None else SHORT_TOP_N)
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

        allows, mkt_score = market_environment_allows_trading(
            target_date, min_score=SHORT_MIN_MARKET_SCORE
        )
        if not allows:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        df_target = pd.read_sql_query(
            "SELECT stock_code, stock_name, close, volume FROM stock_daily_kline WHERE date = ?",
            self.conn,
            params=[target_date],
        )
        if df_target.empty:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        df_target["stock_code"] = (
            df_target["stock_code"].astype(str).str.strip().str.zfill(6)
        )
        if max_scan_stocks is not None and int(max_scan_stocks) > 0:
            df_target = df_target.head(int(max_scan_stocks))

        panel_df = pd.read_sql_query(
            """
            SELECT stock_code, date, open, high, low, close, volume
            FROM stock_daily_kline
            WHERE date <= ?
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
            target_meta_map[r["stock_code"]] = {
                "name": str(r["stock_name"] or "未知").strip(),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }

        candidates: list[dict[str, Any]] = []
        min_bars = int(SHORT_MIN_HISTORY_BARS)

        for code, sub_df in panel_df.groupby("stock_code", sort=False):
            if code not in target_meta_map:
                continue

            meta = target_meta_map[code]
            name = meta["name"]

            if _exclude_code_prefix(code):
                continue
            if SHORT_EXCLUDE_ST and _is_st_name(name):
                continue

            if len(sub_df) < min_bars:
                continue

            last_row = sub_df.iloc[-1]
            last_date = str(last_row["date"]).strip()[:10]
            if last_date != target_date:
                continue
            if is_bar_suspended(last_row):
                continue

            close_arr = pd.to_numeric(sub_df["close"], errors="coerce")
            high_arr = pd.to_numeric(sub_df["high"], errors="coerce")
            low_arr = pd.to_numeric(sub_df["low"], errors="coerce")
            vol_arr = pd.to_numeric(sub_df["volume"], errors="coerce")
            open_arr = pd.to_numeric(sub_df["open"], errors="coerce")

            ma_f = close_arr.rolling(SHORT_MA_FAST).mean()
            ma_s = close_arr.rolling(SHORT_MA_SLOW).mean()

            ema12 = close_arr.ewm(span=12, adjust=False).mean()
            ema26 = close_arr.ewm(span=26, adjust=False).mean()
            diff = ema12 - ema26
            dea = diff.ewm(span=9, adjust=False).mean()
            macd_bar = (diff - dea) * 2.0

            low_min = low_arr.rolling(9).min()
            high_max = high_arr.rolling(9).max()
            rsv = (close_arr - low_min) / (high_max - low_min + 1e-6) * 100.0
            k = rsv.ewm(com=2, adjust=False).mean()
            d = k.ewm(com=2, adjust=False).mean()
            j = 3.0 * k - 2.0 * d

            vol_ratio_5d = vol_arr / (vol_arr.rolling(5).mean() + 1e-6)
            vol_ratio_1d = vol_arr / (vol_arr.shift(1) + 1e-6)
            change_pct = close_arr.pct_change()
            return_5d = close_arr.pct_change(5)

            c_close = float(close_arr.iloc[-1])
            prev_close = float(close_arr.iloc[-2])
            c_open = float(open_arr.iloc[-1])

            if not (
                np.isfinite(c_close)
                and np.isfinite(prev_close)
                and np.isfinite(c_open)
            ):
                continue

            if is_bar_limit_up(c_close, prev_close, code) or _near_limit_up_signal(
                c_close, prev_close, code
            ):
                continue

            c_ma_f = float(ma_f.iloc[-1])
            c_ma_s = float(ma_s.iloc[-1])
            chg = float(change_pct.iloc[-1])
            ret5 = float(return_5d.iloc[-1])
            vr5 = float(vol_ratio_5d.iloc[-1])
            vr1 = float(vol_ratio_1d.iloc[-1])
            j_now = float(j.iloc[-1])
            j_prev = float(j.iloc[-2])
            k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])
            cd, ca = float(diff.iloc[-1]), float(dea.iloc[-1])
            cbar, pbar = float(macd_bar.iloc[-1]), float(macd_bar.iloc[-2])

            if not all(
                np.isfinite(x)
                for x in (
                    c_ma_f,
                    c_ma_s,
                    chg,
                    ret5,
                    vr5,
                    vr1,
                    j_now,
                    j_prev,
                    k_now,
                    d_now,
                    cd,
                    ca,
                    cbar,
                    pbar,
                )
            ):
                continue

            checks = {
                "trend_ma": c_close > c_ma_f and c_ma_f >= c_ma_s,
                "bullish_bar": c_close >= c_open,
                "day_return_band": SHORT_MIN_DAY_RETURN <= chg <= SHORT_MAX_DAY_RETURN,
                "momentum_5d_cap": ret5 <= SHORT_MAX_5D_RETURN,
                "volume": vr5 >= SHORT_VOL_RATIO_5D_MIN
                and vr1 >= SHORT_VOL_RATIO_1D_MIN,
                "macd": cd > ca and cbar >= pbar,
                "kdj": k_now > d_now
                and SHORT_KDJ_J_MIN <= j_now <= SHORT_KDJ_J_MAX
                and (j_now - j_prev) >= SHORT_KDJ_J_SLOPE_MIN,
            }

            if not all(checks.values()):
                continue

            score = self._rule_score(chg, vr5, cbar, pbar, j_now - j_prev)
            detail = {
                "checks": checks,
                "change_pct": chg,
                "return_5d": ret5,
                "vol_ratio_5d": vr5,
                "vol_ratio_1d": vr1,
                "kdj_j": j_now,
                "macd_bar": cbar,
            }

            candidates.append(
                {
                    "stock_code": code,
                    "stock_name": name,
                    "rule_score": score,
                    "close_price": c_close,
                    "day_change_pct": chg,
                    "vol_ratio_5d": vr5,
                    "kdj_j": j_now,
                    "macd_bar": cbar,
                    "detail": detail,
                    "display": {
                        "股票代码": code,
                        "股票名称": name,
                        "信号日": target_date,
                        "收盘价": f"{c_close:.2f} 元",
                        "日涨幅": f"{chg * 100:.2f}%",
                        "五日量比": f"{vr5:.2f} 倍",
                        "KDJ_J": round(j_now, 2),
                        "MACD柱": round(cbar, 4),
                        "规则得分": round(score, 2),
                        "持仓计划": _HOLD_PLAN,
                        "实盘建议": self.advice_text(j_now, chg),
                    },
                }
            )

        if not candidates:
            self._last_persist_rows = []
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        candidates.sort(key=lambda x: float(x["rule_score"]), reverse=True)
        picked = candidates[:top_n]

        rows = []
        for rank, item in enumerate(picked, start=1):
            d = dict(item["display"])
            rows.append(d)
            item["rank"] = rank
            item["advice_text"] = d["实盘建议"]
            item["hold_plan"] = _HOLD_PLAN

        out_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        self._last_persist_rows = picked
        return out_df, target_date, mkt_score

    def get_last_persist_rows(self) -> list[dict[str, Any]]:
        return list(self._last_persist_rows)


def run_short_term_scan(
    connection: sqlite3.Connection,
    target_date: str | None = None,
    *,
    top_n: int | None = None,
    max_scan_stocks: int | None = None,
) -> tuple[pd.DataFrame, str, int]:
    engine = ShortTermRuleStrategy(connection)
    return engine.scan(
        target_date,
        top_n=top_n,
        max_scan_stocks=max_scan_stocks,
    )
