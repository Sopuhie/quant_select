# -*- coding: utf-8 -*-
"""
短线规则选股（持有 1 个交易日）。

信号日 T 收盘后筛选 → 计划 T+1 开盘买入 → T+2 开盘卖出。
多板块动态涨跌幅/近涨停阈值；面板向量化扫描。
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
    SHORT_MIN_AMOUNT,
    SHORT_MIN_HISTORY_BARS,
    SHORT_MIN_MARKET_SCORE,
    SHORT_MIN_TURNOVER,
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


def _get_sector_limits(code: str) -> tuple[float, float, float]:
    """
    返回 (最小日涨幅, 最大日涨幅, 近涨停剔除阈值)。
    主板约 ±7.5% / 9.5%；创业板/科创板 ±15% / 19.2%；北交所 ±25% / 29.2%。
    """
    c = str(code).strip().zfill(6)
    if c.startswith(("300", "301", "688")):
        return 0.005, 0.15, 0.192
    if c.startswith(("8", "4", "83", "87", "92")):
        return 0.005, 0.25, 0.292
    return 0.005, 0.075, 0.095


def _exclude_code_prefix(code: str) -> bool:
    c = str(code).strip().zfill(6)
    if SHORT_EXCLUDE_BJ and c.startswith(("8", "4", "83", "87", "92")):
        return True
    return False


def _passes_liquidity_filter(amount: float, turnover: float) -> bool:
    """成交额或换手率至少满足一项（均为信号日截面）。"""
    if amount >= float(SHORT_MIN_AMOUNT):
        return True
    if turnover >= float(SHORT_MIN_TURNOVER):
        return True
    return False


class ShortTermRuleStrategy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._last_persist_rows: list[dict[str, Any]] = []

    @staticmethod
    def _rule_score(
        chg: float, vr5: float, bar_now: float, bar_prev: float, j_slope: float
    ) -> float:
        bar_delta = bar_now - bar_prev
        bounded_j_slope = max(0.0, min(float(j_slope), 35.0))
        return float(
            20.0 * min(chg / 0.05, 1.5)
            + 25.0 * min(vr5 / 3.0, 2.0)
            + 25.0 * min(max(bar_delta, 0) * 50.0, 2.0)
            + 15.0 * min(bounded_j_slope / 15.0, 2.0)
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

        cur_cols_query = cur.execute("PRAGMA table_info(stock_daily_kline)").fetchall()
        cur_cols = {str(row[1]) for row in cur_cols_query}
        amount_col = "volume * close" if "amount" not in cur_cols else "amount"
        turnover_col = (
            "turnover_rate" if "turnover_rate" in cur_cols else "NULL"
        )

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
            try:
                amt = float(r["amount"]) if pd.notna(r.get("amount")) else 0.0
            except (TypeError, ValueError):
                amt = 0.0
            try:
                to_val = float(r["turnover"]) if pd.notna(r.get("turnover")) else 0.0
            except (TypeError, ValueError):
                to_val = 0.0
            target_meta_map[r["stock_code"]] = {
                "name": str(r["stock_name"] or "未知").strip(),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
                "amount": amt,
                "turnover": to_val,
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

            if not _passes_liquidity_filter(
                float(meta["amount"]), float(meta["turnover"])
            ):
                continue

            if len(sub_df) < min_bars:
                continue

            last_row = sub_df.iloc[-1]
            if str(last_row["date"]).strip()[:10] != target_date:
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

            vol_prev = vol_arr.shift(1)
            vol_ratio_1d = np.where(vol_prev > 0, vol_arr / (vol_prev + 1e-6), 1.0)
            vol_ratio_1d = pd.Series(vol_ratio_1d, index=sub_df.index)

            change_pct = close_arr.pct_change()
            return_5d = close_arr.pct_change(5)

            c_close = float(close_arr.iloc[-1])
            prev_close = float(close_arr.iloc[-2])

            if not (np.isfinite(c_close) and np.isfinite(prev_close)):
                continue

            if is_bar_limit_up(c_close, prev_close, code):
                continue

            min_ret, max_ret, near_limit_thr = _get_sector_limits(code)
            chg = float(change_pct.iloc[-1])
            if not np.isfinite(chg):
                continue
            if SHORT_EXCLUDE_NEAR_LIMIT and chg >= near_limit_thr:
                continue

            c_ma_f = float(ma_f.iloc[-1])
            c_ma_s = float(ma_s.iloc[-1])
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
                "bullish_return": c_close > prev_close,
                "day_return_band": min_ret <= chg <= max_ret,
                "momentum_5d_cap": ret5 <= SHORT_MAX_5D_RETURN,
                "volume": vr5 >= SHORT_VOL_RATIO_5D_MIN
                and vr1 >= SHORT_VOL_RATIO_1D_MIN,
                "macd_cross": cd > ca,
                "macd_slope_relaxed": cbar >= (pbar * 0.8)
                if pbar > 0
                else cbar >= pbar,
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
                "sector_limits": {
                    "min_day_ret": min_ret,
                    "max_day_ret": max_ret,
                    "near_limit": near_limit_thr,
                },
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
