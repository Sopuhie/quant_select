"""
热门题材高爆选股 — 规则 **v2.0**（见项目 ``upgrade_to_theme_v2.txt``）。

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
    THEME_KDJ_J_SLOPE_MIN,
    THEME_KDJ_LEVEL_1,
    THEME_KDJ_LEVEL_2,
    THEME_MA_LONG,
    THEME_MA_SHORT,
    THEME_VOL_RATIO_MIN_1D,
    THEME_VOL_RATIO_MIN_5D,
)

# 历史根数：至少满足 MA60 + 指标稳定（与 v2 文档 65 根一致，且与 MA_LONG 对齐）
THEME_MIN_BARS = max(65, THEME_MA_LONG + 5)
THEME_HIST_LIMIT = 90

STATUS_DEFAULT_V2 = "🔵 横盘蓄势(静待突破)"
STATUS_BUY_V2 = "🟢 三点指标强力共振(v2规则：最佳买入拐点)"
STATUS_KDJ_100_V2 = "⚠️ KDJ探顶J≥100(v2规则：立即减仓当前1/3)"
STATUS_KDJ_110_V2 = "⚠️ KDJ高位脉冲J≥110(v2规则：逢高分步派发减仓)"
STATUS_MACD_EXIT_V2 = "🚨 MACD高位死叉(v2规则：坚决清仓跑路)"

EMPTY_RESULT_COLUMNS = [
    "股票代码",
    "股票名称",
    "最新价格",
    "实盘决策建议结论",
]


def _effective_keyword(
    keyword: str | None,
    theme_keywords: str | list[str] | None,
) -> str | None:
    k = (keyword or "").strip()
    if k:
        return k
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
    return (parts[0].strip() if parts else None) or None


class ThemeAlphaStrategy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def check_market_environment(self, target_date: str) -> bool:
        """
        市场环境评估桩：v2.0 预留；数据源未接入指数表时默认放行。
        """
        _ = target_date
        return True

    def compute_technical_signals(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series] | None:
        """
        v2.0 指标：MA20/MA60、MACD(12,26,9)、KDJ(9,3,3)、5 日量比、相对昨量、日涨跌幅。
        返回 (curr, prev)；不足 ``THEME_MIN_BARS`` 根返回 None。
        """
        if df is None or len(df) < THEME_MIN_BARS:
            return None
        df = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")

        ma20 = close.rolling(int(THEME_MA_SHORT), min_periods=int(THEME_MA_SHORT)).mean()
        ma60 = close.rolling(int(THEME_MA_LONG), min_periods=int(THEME_MA_LONG)).mean()

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
            return np.isfinite(v)

        for col in ("ma20", "ma60", "close", "vol_ratio_5d", "vol_ratio_1d", "change_pct"):
            if not _ok(curr.get(col)):
                return None
        for col in ("macd_diff", "macd_dea", "macd_bar", "kdj_k", "kdj_d", "kdj_j"):
            if not _ok(curr.get(col)) or not _ok(prev.get(col)):
                return None
        return curr, prev

    def scan_hot_themes(
        self,
        target_date: str | None = None,
        keyword: str | None = None,
        *,
        theme_keywords: str | Iterable[str] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        cur = self.conn.cursor()
        if target_date is None:
            res_date = cur.execute(
                "SELECT MAX(date) FROM stock_daily_kline"
            ).fetchone()
            if res_date is None or res_date[0] is None:
                return pd.DataFrame(), ""
            target_date = str(res_date[0]).strip()[:10]
        else:
            target_date = str(target_date).strip()[:10]

        if not self.check_market_environment(target_date):
            return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS), target_date

        eff_kw = _effective_keyword(keyword, theme_keywords)

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
            return pd.DataFrame(), target_date

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
        scored: list[tuple[float, dict[str, Any]]] = []

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
            cond_volume = vr5 >= float(THEME_VOL_RATIO_MIN_5D) and vr1 >= float(
                THEME_VOL_RATIO_MIN_1D
            )
            if chg < 0.02 and vr5 >= 2.0:
                cond_volume = False

            cd, ca = float(curr["macd_diff"]), float(curr["macd_dea"])
            pdiff, pdea = float(prev["macd_diff"]), float(prev["macd_dea"])
            cbar, pbar = float(curr["macd_bar"]), float(prev["macd_bar"])
            cond_macd_buy = (cd > ca) and (
                (pdiff <= pdea and cd >= -0.1)
                or (cbar > 0 and cbar > pbar)
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
                and j_slope > float(THEME_KDJ_J_SLOPE_MIN)
            )

            j110 = float(THEME_KDJ_LEVEL_2)
            j100 = float(THEME_KDJ_LEVEL_1)
            is_kdj_t2 = j_now >= j110
            is_kdj_t1 = j_now >= j100 and j_now < j110
            is_macd_dead = (
                cd < ca and pdiff >= pdea and cd > 0.0
            )

            status = STATUS_DEFAULT_V2
            if is_macd_dead:
                status = STATUS_MACD_EXIT_V2
            elif is_kdj_t2:
                status = STATUS_KDJ_110_V2
            elif is_kdj_t1:
                status = STATUS_KDJ_100_V2
            elif cond_trend and cond_volume and cond_macd_buy and cond_kdj_buy:
                status = STATUS_BUY_V2

            if status == STATUS_DEFAULT_V2:
                continue

            scored.append(
                (
                    vr5,
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{c_close:.2f} 元",
                        "当前量比": f"{vr5:.2f} 倍",
                        "KDJ_J值": round(j_now, 2),
                        "MACD红柱": round(cbar, 4),
                        "实盘决策建议结论": status,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        rows = [t[1] for t in scored]
        return pd.DataFrame(rows), target_date
