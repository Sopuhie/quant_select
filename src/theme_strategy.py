"""
热门题材高爆选股 — 规则 **v2.0**（单表：买入共振 + 「实盘决策建议结论」按 J 分层；阈值来自 ``config``）。

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

RESULT_COLUMNS = [
    "股票代码",
    "股票名称",
    "最新价格",
    "当前量比",
    "KDJ_J值",
    "MACD红柱",
    "实盘决策建议结论",
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

    @staticmethod
    def decision_conclusion_v2(j_now: float) -> str:
        """按 J 值输出「实盘决策建议结论」文案（与界面 v2.0 一致）。"""
        if j_now >= float(THEME_KDJ_LEVEL_2):
            return "⚠️ KDJ高位脉冲J≥110(v2规则: 逢高分步派发减仓)"
        if j_now >= float(THEME_KDJ_LEVEL_1):
            return "⚠️ KDJ探顶J≥100(v2规则: 立即减仓当前1/3)"
        return "✅ 三位一体共振买点成立(v2规则: 可按计划分批建仓)"

    def scan_hot_themes(
        self,
        target_date: str | None = None,
        keyword: str | None = None,
        *,
        theme_keywords: str | Iterable[str] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """
        返回: ``(signals_df, target_date)``。

        signals_df 列: 股票代码, 股票名称, 最新价格, 当前量比, KDJ_J值, MACD红柱, 实盘决策建议结论
        （仅保留通过趋势+量能+MACD/KDJ 买入共振筛选的标的；结论列按 J 分层提示减仓或持仓）。
        """
        cur = self.conn.cursor()
        if target_date is None:
            res_date = cur.execute("SELECT MAX(date) FROM stock_daily_kline").fetchone()
            if res_date is None or res_date[0] is None:
                return (pd.DataFrame(columns=RESULT_COLUMNS), "")
            target_date = str(res_date[0]).strip()[:10]
        else:
            target_date = str(target_date).strip()[:10]

        if not self.check_market_environment(target_date):
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date)

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
            return (pd.DataFrame(columns=RESULT_COLUMNS), target_date)

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
        result_rows: list[dict[str, Any]] = []

        vr5_min = float(THEME_VOL_RATIO_MIN_5D)
        vr1_min = float(THEME_VOL_RATIO_MIN_1D)
        j_slope_min = float(THEME_KDJ_J_SLOPE_MIN)

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

            if cond_trend and cond_volume and cond_macd_buy and cond_kdj_buy:
                result_rows.append(
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{c_close:.2f} 元",
                        "当前量比": f"{vr5:.2f} 倍",
                        "KDJ_J值": round(j_now, 2),
                        "MACD红柱": round(cbar, 4),
                        "实盘决策建议结论": self.decision_conclusion_v2(j_now),
                    }
                )

        out_df = (
            pd.DataFrame(result_rows)
            if result_rows
            else pd.DataFrame(columns=RESULT_COLUMNS)
        )
        if not out_df.empty and "当前量比" in out_df.columns:
            def _vr5_key(s: object) -> float:
                try:
                    t = str(s).replace("倍", "").strip()
                    return float(t)
                except (TypeError, ValueError):
                    return 0.0

            out_df = out_df.copy()
            out_df["_sort_vr5"] = out_df["当前量比"].map(_vr5_key)
            out_df = out_df.sort_values("_sort_vr5", ascending=False).drop(
                columns=["_sort_vr5"]
            )

        return out_df, target_date

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
