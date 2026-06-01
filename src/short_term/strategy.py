# -*- coding: utf-8 -*-
"""
短线规则选股（纯日线 K 线）。

硬性门槛：趋势均线 + 当日收阳 + 流动性/板块近涨停/5日动量等。
指标共振：量比、MACD、KDJ 共 4 项，至少满足 3 项（防止纯日线下条件过苛频繁空仓）。
排序：``_rule_score`` 非线性涨幅评分 + 量比/MACD/J 斜率加权，J≥88 超买扣分。
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from src.factor_calculator import is_bar_limit_up, is_bar_suspended
from src.market_regime import short_term_market_allows_trading

from .board_filter import board_allowed
from .config import (
    SHORT_HOLD_PLAN,
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
    SHORT_MARKET_INDEX_CODE,
    SHORT_MARKET_MOMENTUM_DAYS,
    SHORT_MARKET_MOMENTUM_MIN,
    SHORT_ENTITY_RATIO_MIN,
    SHORT_PCT_SCORE_MAX,
    SHORT_TOP_N,
    SHORT_VOL_RATIO_1D_MIN,
    SHORT_VOL_RATIO_5D_MIN,
    SHORT_VOL_RATIO_CLIP_MAX,
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

_HOLD_PLAN = SHORT_HOLD_PLAN

# 共振项最少通过数量（4 项中至少 3 项）
_RESONANCE_MIN_PASS = 3
# J 超买惩罚阈值与扣分
_J_OVERBOUGHT_THRESHOLD = 88.0
_J_OVERBOUGHT_PENALTY = 30.0


def _is_st_name(name: str) -> bool:
    n = str(name or "").strip().upper()
    return "ST" in n or "*ST" in n


def get_sector_near_limit_threshold(code: str) -> float:
    """
    按代码前缀返回「近涨停」剔除阈值（小数，如 0.095 表示 9.5%）。

    主板 >= 9.5%；创业板/科创板 >= 19.2%；北交所 >= 29.2%。
    """
    c = str(code).strip().zfill(6)
    if c.startswith(("300", "301", "688")):
        return 0.192
    if c.startswith(("8", "4", "83", "87", "92")):
        return 0.292
    return 0.095


def _exclude_code_prefix(code: str) -> bool:
    """北交所等前缀（与 config.SHORT_EXCLUDE_BJ 联动）。"""
    c = str(code).strip().zfill(6)
    if SHORT_EXCLUDE_BJ and c.startswith(("8", "4", "83", "87", "92")):
        return True
    return False


def hard_trend_pass(
    close: float,
    open_price: float,
    ma_fast: float,
    ma_slow: float,
) -> tuple[bool, bool]:
    """
    不可违背的趋势硬性门槛。

    Returns:
        (均线趋势成立, 当日收阳): 收盘>MA5 且 MA5>=MA10；收盘>开盘为收阳。
    """
    trend_ma = close > ma_fast and ma_fast >= ma_slow
    bullish_candle = close > open_price
    return trend_ma, bullish_candle


def count_resonance_items(
    *,
    vr5: float,
    vr1: float,
    dif: float,
    dea: float,
    macd_bar: float,
    macd_bar_prev: float,
    k: float,
    d: float,
    j: float,
    j_prev: float,
) -> tuple[int, dict[str, bool]]:
    """
    温和共振：量比 / MACD / KDJ 共 4 子项，统计满足个数。

    - 5 日量比 >= SHORT_VOL_RATIO_5D_MIN
    - 1 日量比 >= SHORT_VOL_RATIO_1D_MIN
    - DIF > DEA 且 MACD 柱 >= 前柱 × 0.8（红柱不显著缩短）
    - K > D，J 在 [SHORT_KDJ_J_MIN, SHORT_KDJ_J_MAX]，J 上升 >= SHORT_KDJ_J_SLOPE_MIN
    """
    macd_ok = dif > dea and macd_bar >= (macd_bar_prev * 0.8)
    kdj_ok = (
        k > d
        and SHORT_KDJ_J_MIN <= j <= SHORT_KDJ_J_MAX
        and (j - j_prev) >= SHORT_KDJ_J_SLOPE_MIN
    )
    flags = {
        "resonance_vr5": vr5 >= SHORT_VOL_RATIO_5D_MIN,
        "resonance_vr1": vr1 >= SHORT_VOL_RATIO_1D_MIN,
        "resonance_macd": macd_ok,
        "resonance_kdj": kdj_ok,
    }
    return sum(1 for v in flags.values() if v), flags


def score_pct_nonlinear(chg_decimal: float) -> float:
    """
    当日涨幅非线性区间得分（0~100）。

    ``chg_decimal`` 为小数涨幅（0.03 = 3%）。
    - [2.0%, SHORT_PCT_SCORE_MAX]：满分 100（温和启动右侧）
    - > SHORT_PCT_SCORE_MAX：每多 1 个百分点扣 25 分（防高位接盘）
    - < 2.0%：按 pct×15 线性缩水
    """
    pct = float(chg_decimal) * 100.0
    hi = float(SHORT_PCT_SCORE_MAX)
    if 2.0 <= pct <= hi:
        return 100.0
    if pct > hi:
        return max(0.0, 100.0 - (pct - hi) * 25.0)
    return max(0.0, pct * 15.0)


def compute_rule_score(
    chg_decimal: float,
    vr5: float,
    macd_bar: float,
    macd_bar_prev: float,
    j_slope: float,
    j_now: float,
) -> float:
    """
    综合规则得分（加权后可为负，最终截断为 >= 0）。

    权重：当日涨幅 20%、5 日量比 25%、MACD 柱改善 25%、J 斜率 15%。
    J >= 88 时额外扣 30 分（超买惩罚，降低 Top 排名）。
    """
    pct_score = score_pct_nonlinear(chg_decimal)
    vr5_score = min(100.0, (vr5 / SHORT_VOL_RATIO_5D_MIN) * 60.0)
    bar_diff = macd_bar - macd_bar_prev
    macd_score = max(0.0, min(100.0, 50.0 + bar_diff * 200.0))
    j_score = min(100.0, max(0.0, 50.0 + j_slope * 5.0))

    total = (
        pct_score * 0.20
        + vr5_score * 0.25
        + macd_score * 0.25
        + j_score * 0.15
    )
    if j_now >= _J_OVERBOUGHT_THRESHOLD:
        total -= _J_OVERBOUGHT_PENALTY
    return max(0.0, float(total))


class ShortTermRuleStrategy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._last_persist_rows: list[dict[str, Any]] = []

    @staticmethod
    def _rule_score(
        chg: float,
        vr5: float,
        bar_now: float,
        bar_prev: float,
        j_slope: float,
        j_now: float,
    ) -> float:
        """兼容旧调用入口，内部转调 ``compute_rule_score``。"""
        return compute_rule_score(chg, vr5, bar_now, bar_prev, j_slope, j_now)

    @staticmethod
    def advice_text(j_now: float, chg: float) -> str:
        if j_now >= _J_OVERBOUGHT_THRESHOLD:
            return "⚠️ J 偏高：仅适合极小仓博弈，严格执行 T+2 开盘离场"
        if chg >= 0.06:
            return "⚠️ 当日涨幅偏大：次日高开若不及预期应果断止损"
        return "✅ 短线共振成立：T+1 开盘限价买入，T+1 收盘 -5% 破位止损，否则 T+2 收盘离场"

    def scan(
        self,
        target_date: str | None = None,
        *,
        top_n: int | None = None,
        max_scan_stocks: int | None = None,
        include_300: bool = False,
        include_688: bool = False,
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

        allows, mkt_score, _mom = short_term_market_allows_trading(
            target_date,
            min_score=SHORT_MIN_MARKET_SCORE,
            index_code=SHORT_MARKET_INDEX_CODE,
            momentum_days=SHORT_MARKET_MOMENTUM_DAYS,
            momentum_min=SHORT_MARKET_MOMENTUM_MIN,
        )
        if not allows:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)

        cur_cols_query = cur.execute("PRAGMA table_info(stock_daily_kline)").fetchall()
        cur_cols = {str(row[1]) for row in cur_cols_query}
        amount_col = "volume * close" if "amount" not in cur_cols else "amount"
        turnover_col = "turnover_rate" if "turnover_rate" in cur_cols else "NULL"

        df_target = pd.read_sql_query(
            f"SELECT stock_code, stock_name, close, volume, {amount_col} AS amount, {turnover_col} AS turnover FROM stock_daily_kline WHERE date = ?",
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
        if df_target.empty:
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date, mkt_score)
        if max_scan_stocks is not None and int(max_scan_stocks) > 0:
            df_target = df_target.head(int(max_scan_stocks))

        panel_df = pd.read_sql_query(
            "SELECT stock_code, date, open, high, low, close, volume FROM stock_daily_kline WHERE date <= ? ORDER BY stock_code, date ASC",
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

            amt = float(meta["amount"])
            tr = float(meta["turnover"]) if meta["turnover"] is not None else 0.0
            if amt < float(SHORT_MIN_AMOUNT) and tr < float(SHORT_MIN_TURNOVER):
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
            vol_arr = pd.to_numeric(sub_df["volume"], errors="coerce")

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

            vol_ratio_5d = (
                vol_arr / (vol_arr.rolling(5).mean() + 1e-6)
            ).clip(upper=SHORT_VOL_RATIO_CLIP_MAX)
            vol_prev = vol_arr.shift(1)
            vol_ratio_1d = np.where(vol_prev > 0, vol_arr / (vol_prev + 1e-6), 1.0)
            vol_ratio_1d = pd.Series(vol_ratio_1d, index=sub_df.index).clip(
                upper=SHORT_VOL_RATIO_CLIP_MAX
            )

            change_pct = close_arr.pct_change()
            return_5d = close_arr.pct_change(5)

            c_close = float(close_arr.iloc[-1])
            c_open = float(open_arr.iloc[-1])
            prev_close = float(close_arr.iloc[-2])

            if is_bar_limit_up(c_close, prev_close, code):
                continue

            near_limit_thr = get_sector_near_limit_threshold(code)
            chg = float(change_pct.iloc[-1])
            if SHORT_EXCLUDE_NEAR_LIMIT and chg >= near_limit_thr:
                continue

            c_ma_f = float(ma_f.iloc[-1])
            c_ma_s = float(ma_s.iloc[-1])
            ret5 = float(return_5d.iloc[-1])
            if ret5 > SHORT_MAX_5D_RETURN:
                continue

            trend_ok, bullish_ok = hard_trend_pass(c_close, c_open, c_ma_f, c_ma_s)
            if not trend_ok or not bullish_ok:
                continue

            c_low = float(low_arr.iloc[-1])
            c_high = float(high_arr.iloc[-1])
            entity_ratio = (c_close - c_low) / (c_high - c_low + 1e-6)
            if entity_ratio < SHORT_ENTITY_RATIO_MIN:
                continue

            vr5 = float(vol_ratio_5d.iloc[-1])
            vr1 = float(vol_ratio_1d.iloc[-1])
            j_now = float(j.iloc[-1])
            j_prev = float(j.iloc[-2])
            k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])
            cd, ca = float(diff.iloc[-1]), float(dea.iloc[-1])
            cbar, pbar = float(macd_bar.iloc[-1]), float(macd_bar.iloc[-2])

            resonance_n, resonance_flags = count_resonance_items(
                vr5=vr5,
                vr1=vr1,
                dif=cd,
                dea=ca,
                macd_bar=cbar,
                macd_bar_prev=pbar,
                k=k_now,
                d=d_now,
                j=j_now,
                j_prev=j_prev,
            )
            if resonance_n < _RESONANCE_MIN_PASS:
                continue

            j_slope = j_now - j_prev
            score = compute_rule_score(chg, vr5, cbar, pbar, j_slope, j_now)

            checks: dict[str, bool] = {
                "trend_ma": trend_ok,
                "bullish_candle": bullish_ok,
                "momentum_5d_cap": ret5 <= SHORT_MAX_5D_RETURN,
                "near_limit_ok": not (
                    SHORT_EXCLUDE_NEAR_LIMIT and chg >= near_limit_thr
                ),
                "resonance_pass": resonance_n >= _RESONANCE_MIN_PASS,
                **resonance_flags,
            }
            detail = {
                "checks": checks,
                "resonance_count": resonance_n,
                "resonance_min": _RESONANCE_MIN_PASS,
                "change_pct": chg,
                "return_5d": ret5,
                "vol_ratio_5d": vr5,
                "vol_ratio_1d": vr1,
                "kdj_j": j_now,
                "macd_bar": cbar,
                "macd_bar_improve": cbar - pbar,
                "j_slope": j_slope,
                "near_limit_threshold": near_limit_thr,
                "score_breakdown": {
                    "pct_score": score_pct_nonlinear(chg),
                    "vr5_score": min(100.0, (vr5 / SHORT_VOL_RATIO_5D_MIN) * 60.0),
                    "j_overbought_penalty": (
                        _J_OVERBOUGHT_PENALTY if j_now >= _J_OVERBOUGHT_THRESHOLD else 0.0
                    ),
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
    include_300: bool = False,
    include_688: bool = False,
) -> tuple[pd.DataFrame, str, int]:
    engine = ShortTermRuleStrategy(connection)
    return engine.scan(
        target_date,
        top_n=top_n,
        max_scan_stocks=max_scan_stocks,
        include_300=include_300,
        include_688=include_688,
    )
