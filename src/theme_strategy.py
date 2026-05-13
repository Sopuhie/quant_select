"""
热门题材高爆选股 — 实盘 K 线经验扫描（严格按 ``integrate_trader_experience.txt`` Part 1）。

四类状态（仅非默认状态会出现在结果表中）：
  🟢 买入共振 | ⚠️ KDJ 超买减仓 | 🚨 MACD 0 轴上死叉清仓 | 🔵 横盘蓄势（默认，不输出）
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable

import numpy as np
import pandas as pd

MIN_THEME_BARS = 35

STATUS_BUY = "🟢 三大指标金叉共振(买入拐点)"
STATUS_KDJ_EXIT = "⚠️ KDJ极度超买(实盘经验：先减仓一半)"
STATUS_MACD_EXIT = "🚨 MACD高位死叉(0轴上死叉：坚决清仓跑路)"
# Part 1 代码块 lines 117–128：与 ``if status != ...`` 比较用的默认文案
STATUS_DEFAULT = "🔵 横盘蓄势(静待突破)"


def _normalize_keywords(theme_keywords: str | Iterable[str] | None) -> list[str]:
    if theme_keywords is None:
        return []
    if isinstance(theme_keywords, str):
        raw = re.split(r"[,，;；\s]+", theme_keywords.strip())
        return [x.strip() for x in raw if x.strip()]
    return [str(x).strip() for x in theme_keywords if str(x).strip()]


class ThemeAlphaStrategy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def compute_technical_signals(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series] | None:
        """
        与文档一致：MACD(12,26,9)、KDJ(9,3,3)、量比=当日量/(5日均量+1e-6)；
        返回 (当前 bar, 昨日 bar)；不足 35 根返回 None。
        """
        if df is None or len(df) < MIN_THEME_BARS:
            return None
        df = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_bar = (diff - dea) * 2.0

        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-6) * 100.0
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3.0 * k - 2.0 * d

        df = df.copy()
        df["macd_diff"] = diff
        df["macd_dea"] = dea
        df["macd_bar"] = macd_bar
        df["kdj_k"] = k
        df["kdj_d"] = d
        df["kdj_j"] = j
        df["vol_ratio"] = volume / (volume.rolling(5).mean() + 1e-6)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        for key in ("vol_ratio", "macd_bar", "macd_diff", "macd_dea", "kdj_j", "kdj_k"):
            try:
                cv = float(curr[key])
                pv = float(prev[key])
            except (TypeError, ValueError):
                return None
            if not np.isfinite(cv) or not np.isfinite(pv):
                return None
        return curr, prev

    def scan_hot_themes(
        self,
        target_date: str | None = None,
        theme_keywords: str | list[str] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """
        全市场截面扫描（``WHERE date = ?`` 与文档一致）；仅输出触发买入/减仓/清仓的标的。
        ``theme_keywords`` 可选：在扫描前按名称/行业子串预筛（非文档必选，供 ``run_theme_alpha_scan``）。
        """
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
        if "industry" not in df_all.columns:
            df_all["industry"] = ""
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )

        kws = _normalize_keywords(theme_keywords)

        hist_sql = """
            SELECT * FROM stock_daily_kline
            WHERE stock_code = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 45
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
            ind = row.get("industry", "") or ""
            ind = "" if ind is None or (isinstance(ind, float) and pd.isna(ind)) else str(ind)

            if kws:
                blob = name + ind
                if not any(k in blob for k in kws):
                    continue

            df_hist = pd.read_sql_query(
                hist_sql, self.conn, params=[code, target_date]
            )
            if df_hist.empty or len(df_hist) < MIN_THEME_BARS:
                continue
            df_hist["date"] = pd.to_datetime(
                df_hist["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            df_hist = df_hist.sort_values("date").reset_index(drop=True)

            signals = self.compute_technical_signals(df_hist)
            if signals is None:
                continue
            curr, prev = signals

            cond_volume = float(curr["vol_ratio"]) >= 1.3
            cond_macd_buy = float(curr["macd_bar"]) > 0 and float(curr["macd_bar"]) > float(
                prev["macd_bar"]
            )
            cond_kdj_buy = (
                float(curr["kdj_j"]) > float(curr["kdj_k"])
                and float(curr["kdj_j"]) < 95.0
                and float(prev["kdj_j"]) <= float(curr["kdj_j"])
            )

            is_kdj_overbought = float(curr["kdj_j"]) >= 98.0
            is_macd_dead_cross = (
                float(curr["macd_diff"]) < float(curr["macd_dea"])
                and float(prev["macd_diff"]) >= float(prev["macd_dea"])
                and float(curr["macd_diff"]) > 0.0
            )

            status = STATUS_DEFAULT
            if cond_volume and cond_macd_buy and cond_kdj_buy:
                status = STATUS_BUY
            elif is_kdj_overbought:
                status = STATUS_KDJ_EXIT
            elif is_macd_dead_cross:
                status = STATUS_MACD_EXIT

            if status == STATUS_DEFAULT:
                continue

            vr = float(curr["vol_ratio"])
            scored.append(
                (
                    vr,
                    {
                        "股票代码": code,
                        "股票名称": name,
                        "最新价格": f"{float(curr['close']):.2f} 元",
                        "最新量比": f"{vr:.2f} 倍",
                        "KDJ_J值": round(float(curr["kdj_j"]), 2),
                        "MACD红柱": round(float(curr["macd_bar"]), 4),
                        "实盘决策建议结论": status,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        rows = [t[1] for t in scored]
        return pd.DataFrame(rows), target_date
