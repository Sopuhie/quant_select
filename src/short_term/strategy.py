"""
短线规则选股（持有 1 个交易日）。

信号日 T 收盘后筛选 → 计划 T+1 开盘买入 → T+2 开盘卖出（A 股 T+1，不可当日回转）。
与中长线 LightGBM、题材 v2.0 策略独立；仅读 SQLite 日线与指数表。
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
    SHORT_HIST_LIMIT,
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
    """1 日持有规则引擎。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def compute_technical_row(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series] | None:
        if df is None or len(df) < SHORT_MIN_HISTORY_BARS:
            return None
        work = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(work["close"], errors="coerce")
        high = pd.to_numeric(work["high"], errors="coerce")
        low = pd.to_numeric(work["low"], errors="coerce")
        volume = pd.to_numeric(work["volume"], errors="coerce")
        open_ = pd.to_numeric(work["open"], errors="coerce")

        ma_f = close.rolling(SHORT_MA_FAST, min_periods=SHORT_MA_FAST).mean()
        ma_s = close.rolling(SHORT_MA_SLOW, min_periods=SHORT_MA_SLOW).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_bar = (diff - dea) * 2.0

        low_min = low.rolling(9, min_periods=9).min()
        high_max = high.rolling(9, min_periods=9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-6) * 100.0
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3.0 * k - 2.0 * d

        out = work.copy()
        out["ma_fast"] = ma_f
        out["ma_slow"] = ma_s
        out["macd_diff"] = diff
        out["macd_dea"] = dea
        out["macd_bar"] = macd_bar
        out["kdj_k"] = k
        out["kdj_d"] = d
        out["kdj_j"] = j
        out["vol_ratio_5d"] = volume / (volume.rolling(5, min_periods=5).mean() + 1e-6)
        out["vol_ratio_1d"] = volume / (volume.shift(1) + 1e-6)
        out["change_pct"] = close.pct_change()
        out["return_5d"] = close.pct_change(5)
        out["open"] = open_

        curr = out.iloc[-1]
        prev = out.iloc[-2]

        def _ok(x: object) -> bool:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return False
            return bool(np.isfinite(v))

        need = (
            "ma_fast",
            "ma_slow",
            "close",
            "open",
            "vol_ratio_5d",
            "vol_ratio_1d",
            "change_pct",
            "return_5d",
            "macd_diff",
            "macd_dea",
            "macd_bar",
            "kdj_k",
            "kdj_d",
            "kdj_j",
        )
        for col in need:
            if not _ok(curr.get(col)) or (
                col in ("macd_diff", "macd_dea", "macd_bar", "kdj_k", "kdj_d", "kdj_j")
                and not _ok(prev.get(col))
            ):
                return None
        return curr, prev

    @staticmethod
    def _rule_score(
        chg: float,
        vr5: float,
        bar_now: float,
        bar_prev: float,
        j_slope: float,
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

    def evaluate_rules(
        self,
        code: str,
        name: str,
        curr: pd.Series,
        prev: pd.Series,
    ) -> tuple[bool, float, dict[str, Any]]:
        c_close = float(curr["close"])
        c_open = float(curr["open"])
        c_ma_f = float(curr["ma_fast"])
        c_ma_s = float(curr["ma_slow"])
        chg = float(curr["change_pct"])
        ret5 = float(curr["return_5d"])
        vr5 = float(curr["vol_ratio_5d"])
        vr1 = float(curr["vol_ratio_1d"])
        j_now = float(curr["kdj_j"])
        j_prev = float(prev["kdj_j"])
        k_now, d_now = float(curr["kdj_k"]), float(curr["kdj_d"])
        k_prev, d_prev = float(prev["kdj_k"]), float(prev["kdj_d"])
        cd, ca = float(curr["macd_diff"]), float(curr["macd_dea"])
        cbar, pbar = float(curr["macd_bar"]), float(prev["macd_bar"])

        checks: dict[str, bool] = {
            "trend_ma": c_close > c_ma_f and c_ma_f >= c_ma_s,
            "bullish_bar": c_close >= c_open,
            "day_return_band": SHORT_MIN_DAY_RETURN <= chg <= SHORT_MAX_DAY_RETURN,
            "momentum_5d_cap": ret5 <= SHORT_MAX_5D_RETURN,
            "volume": vr5 >= SHORT_VOL_RATIO_5D_MIN and vr1 >= SHORT_VOL_RATIO_1D_MIN,
            "macd": cd > ca and cbar >= pbar,
            "kdj": k_now > d_now
            and j_now >= SHORT_KDJ_J_MIN
            and j_now <= SHORT_KDJ_J_MAX
            and (j_now - j_prev) >= SHORT_KDJ_J_SLOPE_MIN,
        }
        passed = all(checks.values())
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
        _ = name
        return passed, score, detail

    def scan(
        self,
        target_date: str | None = None,
        *,
        top_n: int | None = None,
        max_scan_stocks: int | None = None,
    ) -> tuple[pd.DataFrame, str, int]:
        """
        返回 ``(signals_df, target_date, market_score)``。
        """
        top_n = int(top_n if top_n is not None else SHORT_TOP_N)
        cur = self.conn.cursor()
        if target_date is None:
            res_date = cur.execute("SELECT MAX(date) FROM stock_daily_kline").fetchone()
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

        df_all = pd.read_sql_query(
            "SELECT * FROM stock_daily_kline WHERE date = ?",
            self.conn,
            params=[target_date],
        )
        if df_all.empty:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        df_all["stock_code"] = df_all["stock_code"].astype(str).str.strip().str.zfill(6)
        if max_scan_stocks is not None and int(max_scan_stocks) > 0:
            df_all = df_all.head(int(max_scan_stocks))

        hist_sql = """
            SELECT * FROM stock_daily_kline
            WHERE stock_code = ? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
        """

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        for _, row in df_all.iterrows():
            code = str(row["stock_code"]).strip().zfill(6)
            if code in seen:
                continue
            seen.add(code)

            if _exclude_code_prefix(code):
                continue

            raw_name = row.get("stock_name", "未知")
            if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
                name = "未知"
            else:
                name = str(raw_name)

            if SHORT_EXCLUDE_ST and _is_st_name(name):
                continue

            df_hist = pd.read_sql_query(
                hist_sql,
                self.conn,
                params=[code, target_date, SHORT_HIST_LIMIT],
            )
            if df_hist.empty or len(df_hist) < SHORT_MIN_HISTORY_BARS:
                continue
            df_hist["date"] = pd.to_datetime(
                df_hist["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            df_hist = df_hist.sort_values("date").reset_index(drop=True)

            if is_bar_suspended(df_hist.iloc[-1]):
                continue

            sig = self.compute_technical_row(df_hist)
            if sig is None:
                continue
            curr, prev = sig

            prev_close = float(df_hist.iloc[-2]["close"])
            c_close = float(curr["close"])
            if is_bar_limit_up(c_close, prev_close, code):
                continue
            if _near_limit_up_signal(c_close, prev_close, code):
                continue

            passed, score, detail = self.evaluate_rules(code, name, curr, prev)
            if not passed:
                continue

            chg = float(curr["change_pct"])
            vr5 = float(curr["vol_ratio_5d"])
            j_now = float(curr["kdj_j"])
            cbar = float(curr["macd_bar"])

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
        return getattr(self, "_last_persist_rows", [])


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
