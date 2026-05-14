"""
热门题材高爆选股 — 规则 **v2.1**（买入/卖出分表；阈值来自 ``config``）。

在 SQLite 截面 + 单股历史（倒序取、升序算）上计算 MA/MACD/KDJ/量比；
保留 SQL 关键词预筛与 ``run_theme_alpha_scan`` 的 ``theme_keywords`` 兼容。
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import (
    MIN_HISTORY_BARS,
    THEME_KDJ_J_SLOPE_MIN,
    THEME_KDJ_LEVEL_1,
    THEME_KDJ_LEVEL_2,
    THEME_MA_LONG,
    THEME_MA_SHORT,
    THEME_VOL_RATIO_MIN_1D,
    THEME_VOL_RATIO_MIN_5D,
)

THEME_HIST_LIMIT = 250
THEME_MIN_BARS = MIN_HISTORY_BARS

BUY_RESULT_COLUMNS = [
    "股票代码",
    "股票名称",
    "最新价格",
    "当前量比",
    "KDJ_J值",
    "MACD红柱",
    "信号类型",
    "建议",
    "触发信号",
]

# 热门题材打分权重
THEME_SCORE_TREND = 30
THEME_SCORE_VOLUME = 25
THEME_SCORE_MACD = 25
THEME_SCORE_KDJ = 20
THEME_SCORE_BUY_THRESHOLD = 50  # 总分超过此阈值即推荐买入

SELL_RESULT_COLUMNS = [
    "股票代码",
    "股票名称",
    "最新价格",
    "KDJ_J值",
    "MACD红柱",
    "信号类型",
    "建议",
    "详情",
]


class ThemeAlphaStrategy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.market_score_cache: dict[str, int] = {}

    def get_market_score(self, target_date: str) -> int:
        """
        返回 0~100 的大盘环境分。
        数据缺失或表不存在时默认 60 分（允许交易），不阻断策略。
        """
        td = str(target_date).strip()[:10]
        if td in self.market_score_cache:
            return int(self.market_score_cache[td])
        score_out = 60
        try:
            df_idx = pd.read_sql_query(
                """
                SELECT date, close FROM index_daily
                WHERE index_code = '000300' AND date <= ?
                ORDER BY date DESC
                LIMIT 200
                """,
                self.conn,
                params=[td],
            )
            if len(df_idx) >= 20:
                df_idx = df_idx.sort_values("date")
                ma20 = df_idx["close"].rolling(20).mean().iloc[-1]
                cond = float(df_idx["close"].iloc[-1]) > float(ma20)
                score = 20 if cond else 0
                score_out = int(score + 40)
        except Exception:
            score_out = 60
        self.market_score_cache[td] = score_out
        return score_out

    def check_market_environment(self, target_date: str) -> bool:
        return self.get_market_score(target_date) >= 60

    def compute_technical_signals(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series] | None:
        if df is None or len(df) < THEME_MIN_BARS:
            return None
        df = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")

        ma_s = int(THEME_MA_SHORT)
        ma_l = int(THEME_MA_LONG)
        ma20 = close.rolling(ma_s, min_periods=ma_s).mean()
        ma60 = close.rolling(ma_l, min_periods=ma_l).mean()

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

        out = df.copy()
        out["ma20"] = ma20
        out["ma60"] = ma60
        out["macd_diff"] = diff
        out["macd_dea"] = dea
        out["macd_bar"] = macd_bar
        out["kdj_k"] = k
        out["kdj_d"] = d
        out["kdj_j"] = j
        out["vol_ratio_5d"] = volume / (volume.rolling(5, min_periods=5).mean() + 1e-6)
        out["vol_ratio_1d"] = volume / (volume.shift(1) + 1e-6)
        out["change_pct"] = close.pct_change()

        curr = out.iloc[-1]
        prev = out.iloc[-2]

        def _ok(x: object) -> bool:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return False
            return bool(np.isfinite(v))

        for col in ("ma20", "ma60", "close", "vol_ratio_5d", "vol_ratio_1d", "change_pct"):
            if not _ok(curr.get(col)):
                return None
        for col in ("macd_diff", "macd_dea", "macd_bar", "kdj_k", "kdj_d", "kdj_j"):
            if not _ok(curr.get(col)) or not _ok(prev.get(col)):
                return None
        return curr, prev

    def detect_macd_divergence(self, df_hist: pd.DataFrame) -> bool:
        """近 20 日：价格新高而 DIF 不创新高（简化顶背离）。"""
        if len(df_hist) < 20:
            return False
        close = pd.to_numeric(df_hist["close"], errors="coerce")
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        prices = close.to_numpy(dtype=float)[-20:]
        difs = diff.to_numpy(dtype=float)[-20:]
        price_peaks: list[tuple[int, float]] = []
        dif_peaks: list[tuple[int, float]] = []
        n = len(prices)
        for i in range(5, n - 5):
            window = prices[i - 5 : i + 6]
            if prices[i] == float(np.max(window)):
                price_peaks.append((i, float(prices[i])))
                dif_peaks.append((i, float(difs[i])))
        if len(price_peaks) < 2:
            return False
        last_price = price_peaks[-1][1]
        prev_price = price_peaks[-2][1]
        last_dif = dif_peaks[-1][1]
        prev_dif = dif_peaks[-2][1]
        return last_price > prev_price and last_dif < prev_dif

    def scan_hot_themes(
        self,
        target_date: str | None = None,
        keyword: str | None = None,
        *,
        theme_keywords: str | Iterable[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        返回: ``(buy_signals, sell_signals, target_date)``。

        buy_signals 列: 股票代码, 股票名称, 最新价格, 当前量比, KDJ_J值, MACD红柱, 信号类型, 建议
        sell_signals 列: 股票代码, 股票名称, 最新价格, KDJ_J值, MACD红柱, 信号类型, 建议, 详情
        """
        cur = self.conn.cursor()
        if target_date is None:
            res_date = cur.execute("SELECT MAX(date) FROM stock_daily_kline").fetchone()
            if res_date is None or res_date[0] is None:
                return (
                    pd.DataFrame(columns=BUY_RESULT_COLUMNS),
                    pd.DataFrame(columns=SELL_RESULT_COLUMNS),
                    "",
                )
            target_date = str(res_date[0]).strip()[:10]
        else:
            target_date = str(target_date).strip()[:10]

        if not self.check_market_environment(target_date):
            return (
                pd.DataFrame(columns=BUY_RESULT_COLUMNS),
                pd.DataFrame(columns=SELL_RESULT_COLUMNS),
                target_date,
            )

        eff_kw = self._effective_keyword(keyword, theme_keywords)

        if eff_kw:
            like_term = f"%{eff_kw}%"
            df_all = pd.read_sql_query(
                """
                SELECT * FROM stock_daily_kline
                WHERE date = ? AND (stock_name LIKE ? OR stock_code LIKE ?)
                """,
                self.conn,
                params=[target_date, like_term, like_term],
            )
        else:
            df_all = pd.read_sql_query(
                "SELECT * FROM stock_daily_kline WHERE date = ?",
                self.conn,
                params=[target_date],
            )
        if df_all.empty:
            return (
                pd.DataFrame(columns=BUY_RESULT_COLUMNS),
                pd.DataFrame(columns=SELL_RESULT_COLUMNS),
                target_date,
            )

        df_all["stock_code"] = df_all["stock_code"].astype(str).str.strip().str.zfill(6)
        if "stock_name" not in df_all.columns:
            df_all["stock_name"] = ""

        hist_sql = """
            SELECT * FROM stock_daily_kline
            WHERE stock_code = ? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
        """

        seen: set[str] = set()
        buy_rows: list[dict[str, Any]] = []
        sell_rows: list[dict[str, Any]] = []

        vr5_min = float(THEME_VOL_RATIO_MIN_5D)
        vr1_min = float(THEME_VOL_RATIO_MIN_1D)
        j_slope_min = float(THEME_KDJ_J_SLOPE_MIN)
        j_lv1 = float(THEME_KDJ_LEVEL_1)
        j_lv2 = float(THEME_KDJ_LEVEL_2)

        for _, row in df_all.iterrows():
            code = str(row["stock_code"]).strip().zfill(6)
            if code in seen:
                continue
            seen.add(code)

            raw_name = row.get("stock_name", "未知")
            if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
                name = "未知"
            else:
                name = str(raw_name)

            df_hist = pd.read_sql_query(
                hist_sql,
                self.conn,
                params=[code, target_date, THEME_HIST_LIMIT],
            )
            if df_hist.empty or len(df_hist) < THEME_MIN_BARS:
                continue
            df_hist["date"] = pd.to_datetime(
                df_hist["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            df_hist = df_hist.sort_values("date").reset_index(drop=True)

            sig = self.compute_technical_signals(df_hist)
            if sig is None:
                continue
            curr, prev = sig

            c_close = float(curr["close"])
            c_ma20 = float(curr["ma20"])
            c_ma60 = float(curr["ma60"])
            cond_trend = c_close > c_ma20 and c_close > c_ma60 and c_ma20 > c_ma60

            vr5 = float(curr["vol_ratio_5d"])
            vr1 = float(curr["vol_ratio_1d"])
            chg = float(curr["change_pct"])
            cond_volume = vr5 >= vr5_min and vr1 >= vr1_min
            if chg < 0.02 and vr5 >= 2.0:
                cond_volume = False

            cd, ca = float(curr["macd_diff"]), float(curr["macd_dea"])
            pdiff, pdea = float(prev["macd_diff"]), float(prev["macd_dea"])
            cbar, pbar = float(curr["macd_bar"]), float(prev["macd_bar"])
            cond_macd_buy = (cd > ca) and (
                cbar > pbar or (pdiff <= pdea and cbar > 0)
            )

            j_now = float(curr["kdj_j"])
            j_prev = float(prev["kdj_j"])
            j_slope = j_now - j_prev
            k_now, d_now = float(curr["kdj_k"]), float(curr["kdj_d"])
            k_prev, d_prev = float(prev["kdj_k"]), float(prev["kdj_d"])
            k_cross_d = (k_prev <= d_prev) and (k_now > d_now)
            cond_kdj_buy = (
                k_cross_d
                and j_now > 40.0
                and j_slope > j_slope_min
            )

            # 涨跌停过滤：涨停/跌停日无法交易，跳过
            if abs(chg) >= 0.095:
                continue

            # 打分制买入评估（替代原 AND 逻辑）
            score = 0
            signals = []
            if cond_trend:
                score += THEME_SCORE_TREND
                signals.append("趋势多头")
            if cond_volume:
                score += THEME_SCORE_VOLUME
                signals.append("放量启动")
            if cond_macd_buy:
                score += THEME_SCORE_MACD
                signals.append("MACD多头")
            if cond_kdj_buy:
                score += THEME_SCORE_KDJ
                signals.append("KDJ金叉")

            if score >= THEME_SCORE_BUY_THRESHOLD:
                if score >= 80:
                    suggestion = "买入共振成立，建议建仓30%"
                elif score >= 65:
                    suggestion = "强信号，建议建仓20%"
                else:
                    suggestion = "偏多信号，建议轻仓试探15%"
                buy_rows.append(
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{c_close:.2f} 元",
                        "当前量比": f"{vr5:.2f} 倍",
                        "KDJ_J值": round(j_now, 2),
                        "MACD红柱": round(cbar, 4),
                        "信号类型": "BUY",
                        "建议": suggestion,
                        "触发信号": " + ".join(signals),
                    }
                )

            sells_here: list[dict[str, Any]] = []
            if j_now < 80.0 and j_prev >= 80.0:
                sells_here.append(
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{c_close:.2f} 元",
                        "KDJ_J值": round(j_now, 2),
                        "MACD红柱": round(cbar, 4),
                        "信号类型": "SELL_ALL",
                        "建议": "KDJ 跌破 80，短线强势结束，建议清仓",
                        "详情": f"J值从 {j_prev:.1f} 下穿 80",
                    }
                )
            elif j_now >= j_lv1 and j_now < j_lv2:
                if not any(x.get("信号类型") == "SELL_ALL" for x in sells_here):
                    sells_here.append(
                        {
                            "股票代码": code,
                            "股票名称": name,
                            "最新价格": f"{c_close:.2f} 元",
                            "KDJ_J值": round(j_now, 2),
                            "MACD红柱": round(cbar, 4),
                            "信号类型": "SELL_HALF",
                            "建议": f"J值≥{j_lv1:.0f}，短线超买，建议卖出50%仓位",
                            "详情": f"J值={j_now:.1f}",
                        }
                    )
            elif j_now >= j_lv2:
                sells_here.append(
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{c_close:.2f} 元",
                        "KDJ_J值": round(j_now, 2),
                        "MACD红柱": round(cbar, 4),
                        "信号类型": "SELL_ALL",
                        "建议": f"J值≥{j_lv2:.0f}，极度超买，建议清仓",
                        "详情": f"J值={j_now:.1f}",
                    }
                )

            if cd < ca and pdiff >= pdea and cd > 0.0:
                sells_here.append(
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{c_close:.2f} 元",
                        "KDJ_J值": round(j_now, 2),
                        "MACD红柱": round(cbar, 4),
                        "信号类型": "SELL_ALL",
                        "建议": "MACD 零轴上死叉，主升浪结束，建议立即清仓",
                        "详情": f"DIF={cd:.3f} DEA={ca:.3f}",
                    }
                )
            if self.detect_macd_divergence(df_hist):
                sells_here.append(
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{c_close:.2f} 元",
                        "KDJ_J值": round(j_now, 2),
                        "MACD红柱": round(cbar, 4),
                        "信号类型": "SELL_HALF",
                        "建议": "MACD 顶背离，价格新高但动能不足，建议减仓50%",
                        "详情": "顶背离信号",
                    }
                )
            sell_rows.extend(sells_here)

        buy_df = (
            pd.DataFrame(buy_rows)
            if buy_rows
            else pd.DataFrame(columns=BUY_RESULT_COLUMNS)
        )
        if not buy_df.empty and "当前量比" in buy_df.columns:
            def _vr5_key(s: object) -> float:
                try:
                    t = str(s).replace("倍", "").strip()
                    return float(t)
                except (TypeError, ValueError):
                    return 0.0

            buy_df = buy_df.copy()
            buy_df["_sort_vr5"] = buy_df["当前量比"].map(_vr5_key)
            buy_df = buy_df.sort_values("_sort_vr5", ascending=False).drop(
                columns=["_sort_vr5"]
            )

        sell_df = (
            pd.DataFrame(sell_rows)
            if sell_rows
            else pd.DataFrame(columns=SELL_RESULT_COLUMNS)
        )
        return buy_df, sell_df, target_date

    def _effective_keyword(
        self,
        keyword: str | None,
        theme_keywords: str | Iterable[str] | None,
    ) -> str | None:
        if keyword is not None and str(keyword).strip():
            return str(keyword).strip()
        if theme_keywords is None:
            return None
        if isinstance(theme_keywords, (list, tuple)):
            for x in theme_keywords:
                s = str(x).strip()
                if s:
                    return s
            return None
        s = str(theme_keywords).strip()
        if not s:
            return None
        parts = re.split(r"[,，;；\s]+", s)
        return parts[0].strip() if parts else None
