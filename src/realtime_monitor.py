"""
Top3 标的分钟级监控：拉取 1 分钟线、VWAP/技术信号判定、去重写入 signal_history。
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    REALTIME_MONITOR_MAX_REQUESTS_PER_MINUTE,
    REALTIME_MONITOR_PAUSE_AFTER_FAILS,
    REALTIME_MONITOR_PAUSE_SECONDS,
    REALTIME_MONITOR_SLEEP_SEC_MAX,
    REALTIME_MONITOR_SLEEP_SEC_MIN,
    REALTIME_SIGNAL_DEDUP_WINDOW_MINUTES,
    TOP_N_SELECTION,
)
from .database import (
    fetch_signal_history_for_stock_on_date,
    fetch_top3_selections_for_monitor,
    init_db,
    insert_signal_record,
    signal_exists_in_minute,
    signal_exists_within_minutes,
)
from .kline_chart import get_realtime_min_data
from .utils import is_a_share_intraday_session

logger = logging.getLogger(__name__)

SIGNAL_VWAP_SUPPORT = "VWAP_Support"
SIGNAL_MACD_GOLDEN = "MACD_Golden_Cross"
SIGNAL_VOL_PRICE = "Volume_Price_Resonance"

_COLOR_BUY = "#16a34a"
_COLOR_VOL_UP = "rgba(220, 38, 38, 0.72)"  # 上涨红
_COLOR_VOL_DN = "rgba(22, 163, 74, 0.72)"  # 下跌绿

# 分钟级请求时间戳（monotonic），用于 RPM 限流
_request_ts_mono: deque[float] = deque()
_consecutive_api_fails = 0
_pause_until_wall: float = 0.0


def _sleep_inter_request_jitter() -> None:
    lo = float(REALTIME_MONITOR_SLEEP_SEC_MIN)
    hi = float(REALTIME_MONITOR_SLEEP_SEC_MAX)
    if hi < lo:
        lo, hi = hi, lo
    if hi <= 0:
        return
    time.sleep(random.uniform(max(0.0, lo), hi))


def _rpm_guard_before_request() -> None:
    cap = int(REALTIME_MONITOR_MAX_REQUESTS_PER_MINUTE)
    if cap <= 0:
        return
    now = time.monotonic()
    while _request_ts_mono and (now - _request_ts_mono[0]) > 60.0:
        _request_ts_mono.popleft()
    if len(_request_ts_mono) >= cap:
        wait = 60.0 - (now - _request_ts_mono[0]) + 0.05
        if wait > 0:
            logger.info(
                "实时监控 RPM 达上限 %s，休眠 %.2fs",
                cap,
                wait,
            )
            time.sleep(wait)
        now = time.monotonic()
        while _request_ts_mono and (now - _request_ts_mono[0]) > 60.0:
            _request_ts_mono.popleft()
    _request_ts_mono.append(time.monotonic())


def _monitor_cooling() -> bool:
    return time.time() < float(_pause_until_wall)


def _register_api_failure() -> None:
    global _consecutive_api_fails, _pause_until_wall
    _consecutive_api_fails += 1
    n = int(REALTIME_MONITOR_PAUSE_AFTER_FAILS)
    if n > 0 and _consecutive_api_fails >= n:
        pause_sec = float(REALTIME_MONITOR_PAUSE_SECONDS)
        _pause_until_wall = time.time() + pause_sec
        logger.error(
            "实时监控连续失败 %s 次，暂停 %.0f 秒（数据源不稳定冷却）",
            _consecutive_api_fails,
            pause_sec,
        )
        _consecutive_api_fails = 0


def _register_api_success() -> None:
    global _consecutive_api_fails
    _consecutive_api_fails = 0


def compute_td_sequential_labels(close: np.ndarray | pd.Series) -> tuple[list[str | None], list[str | None]]:
    """
    分时九转（Tom DeMark Setup）：相对 4 根 K 线前的收盘比较，连续满足则在当前根标注 1–9。
    返回 (低位序列标注, 高位序列标注)，元素为 ``None`` 或 ``'1'``…``'9'`` 字符串。
    """
    c = np.asarray(pd.to_numeric(close, errors="coerce"), dtype=float)
    n = len(c)
    buy_l: list[str | None] = [None] * n
    sell_l: list[str | None] = [None] * n
    bs = 0
    ss = 0
    for i in range(4, n):
        if np.isfinite(c[i]) and np.isfinite(c[i - 4]) and c[i] < c[i - 4]:
            bs += 1
            buy_l[i] = str(min(bs, 9))
            if bs >= 9:
                bs = 0
        else:
            bs = 0
        if np.isfinite(c[i]) and np.isfinite(c[i - 4]) and c[i] > c[i - 4]:
            ss += 1
            sell_l[i] = str(min(ss, 9))
            if ss >= 9:
                ss = 0
        else:
            ss = 0
    return buy_l, sell_l


def _minute_macd_kdj(close: pd.Series) -> pd.DataFrame:
    """1 分钟收盘价序列上的 MACD / KDJ（与 factor_calculator 参数一致）。"""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    low9 = c.rolling(9, min_periods=5).min()
    high9 = c.rolling(9, min_periods=5).max()
    rsv = ((c - low9) / (high9 - low9 + 1e-9) * 100.0).clip(0.0, 100.0)
    k = rsv.rolling(3, min_periods=1).mean()
    d = k.rolling(3, min_periods=1).mean()
    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist, "k": k, "d": d, "j": j})


def enrich_minute_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    补充 VWAP、成交额列；VWAP = 累计成交额 / 累计成交量。
    若无成交额列则用 typical_price * volume 近似。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    vol = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    if "amount" in out.columns:
        amt = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    else:
        typical = (
            pd.to_numeric(out.get("high", out["close"]), errors="coerce")
            + pd.to_numeric(out.get("low", out["close"]), errors="coerce")
            + pd.to_numeric(out["close"], errors="coerce")
        ) / 3.0
        amt = typical * vol
    out["amount"] = amt
    out["volume"] = vol
    cum_amt = out["amount"].cumsum()
    cum_vol = out["volume"].cumsum().replace(0, np.nan)
    out["vwap"] = cum_amt / cum_vol
    out["vwap"] = out["vwap"].ffill().bfill()
    tk = _minute_macd_kdj(out["close"])
    for col in tk.columns:
        out[col] = tk[col].values
    return out


def fetch_today_minute_bars(stock_code: str) -> pd.DataFrame | None:
    """拉取当日 1 分钟线并 enrich；失败返回 None。"""
    raw = get_realtime_min_data(stock_code)
    if raw is None or raw.empty:
        return None
    # 仅保留当天开盘以来的分钟线（或最多保留最近 240 个分钟节点）
    raw = raw.tail(240).reset_index(drop=True)
    rename_extra = {}
    for c in raw.columns:
        sc = str(c)
        if sc in ("成交额", "amount"):
            rename_extra[c] = "amount"
    if rename_extra:
        raw = raw.rename(columns=rename_extra)
    enriched = enrich_minute_bars(raw)
    return enriched if not enriched.empty else None


def _detect_vwap_support(df: pd.DataFrame) -> tuple[str, float, str] | None:
    """判定 A：回踩 VWAP 不破、缩量后放量起跳。"""
    if len(df) < 8:
        return None
    sub = df.tail(8).reset_index(drop=True)
    vol = sub["volume"].astype(float)
    close = sub["close"].astype(float)
    vwap = sub["vwap"].astype(float)
    tol = 0.0025
    touch_idx = None
    for i in range(len(sub) - 3):
        if not np.isfinite(vwap.iloc[i]):
            continue
        near = abs(close.iloc[i] - vwap.iloc[i]) / max(vwap.iloc[i], 1e-6) <= tol
        hold = close.iloc[i] >= vwap.iloc[i] * (1.0 - tol)
        if near and hold:
            touch_idx = i
            break
    if touch_idx is None:
        return None
    shrink = vol.iloc[touch_idx + 1 : touch_idx + 3]
    if len(shrink) < 2 or not (shrink.iloc[0] > 0 and shrink.iloc[-1] <= shrink.iloc[0] * 0.85):
        return None
    last = sub.iloc[-1]
    prev = sub.iloc[-2]
    if vol.iloc[-1] < vol.iloc[-3] * 1.25:
        return None
    if close.iloc[-1] <= close.iloc[-2]:
        return None
    if close.iloc[-1] < vwap.iloc[-1] * (1.0 - tol):
        return None
    price = float(last["close"])
    ts = last["time"]
    sig_time = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
    reason = (
        f"回踩 VWAP({float(vwap.iloc[-1]):.2f}) 不破，缩量后放量起跳 "
        f"({float(prev['close']):.2f}→{price:.2f})"
    )
    return SIGNAL_VWAP_SUPPORT, price, reason


def _detect_macd_kdj(df: pd.DataFrame) -> tuple[str, float, str] | None:
    """判定 B：1 分钟 MACD 金叉且 KDJ 超卖区回升。"""
    if len(df) < 30:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    for col in ("dif", "dea", "k", "d", "j"):
        if not np.isfinite(float(curr[col])) or not np.isfinite(float(prev[col])):
            return None
    golden = float(prev["dif"]) <= float(prev["dea"]) and float(curr["dif"]) > float(curr["dea"])
    if not golden:
        return None
    j_prev, j_now = float(prev["j"]), float(curr["j"])
    k_cross = float(prev["k"]) <= float(prev["d"]) and float(curr["k"]) > float(curr["d"])
    oversold_recover = j_prev < 25.0 and j_now > j_prev and (k_cross or j_now > 20.0)
    if not oversold_recover:
        return None
    price = float(curr["close"])
    ts = curr["time"]
    sig_time = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
    reason = (
        f"MACD 金叉(DIF {float(prev['dif']):.3f}→{float(curr['dif']):.3f})，"
        f"KDJ 超卖回升(J {j_prev:.1f}→{j_now:.1f})"
    )
    return SIGNAL_MACD_GOLDEN, price, reason


def _detect_volume_price(df: pd.DataFrame) -> tuple[str, float, str] | None:
    """判定 C：近 3 分钟量持续放大且区间涨幅 > 0.5%。"""
    if len(df) < 3:
        return None
    tail = df.tail(3).reset_index(drop=True)
    vol = tail["volume"].astype(float)
    if not (vol.iloc[1] > vol.iloc[0] and vol.iloc[2] > vol.iloc[1]):
        return None
    p0 = float(tail["close"].iloc[0])
    p2 = float(tail["close"].iloc[-1])
    if p0 <= 0:
        return None
    ret = p2 / p0 - 1.0
    if ret < 0.005:
        return None
    last = tail.iloc[-1]
    price = float(last["close"])
    ts = last["time"]
    sig_time = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
    reason = f"近 3 分钟量能递增，涨幅 {ret * 100:.2f}%"
    return SIGNAL_VOL_PRICE, price, reason


def detect_intraday_signals(df: pd.DataFrame) -> list[dict[str, Any]]:
    """对 enrich 后的分钟线运行全部检测器。"""
    out: list[dict[str, Any]] = []
    if df is None or df.empty:
        return out
    detectors = (_detect_vwap_support, _detect_macd_kdj, _detect_volume_price)
    for fn in detectors:
        hit = fn(df)
        if hit is None:
            continue
        sig_type, price, reason = hit
        ts = df.iloc[-1]["time"]
        sig_time = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        out.append(
            {
                "signal_type": sig_type,
                "signal_time": sig_time,
                "signal_price": price,
                "reason": reason,
            }
        )
    return out


def _signal_time_to_minute_bucket(signal_time: str) -> str:
    s = str(signal_time).strip()
    if len(s) >= 16:
        return s[:16]
    return s


def persist_signals_for_stock(
    stock_code: str,
    stock_name: str,
    signals: list[dict[str, Any]],
    *,
    realtime_score: float | None = None,
) -> int:
    """去重入库；返回新写入条数。"""
    code = str(stock_code).strip().zfill(6)
    written = 0
    win = int(REALTIME_SIGNAL_DEDUP_WINDOW_MINUTES)
    for sig in signals:
        sig_type = str(sig["signal_type"])
        sig_time = str(sig["signal_time"])
        if win > 0 and signal_exists_within_minutes(
            code, sig_time, sig_type, win, db_path=None
        ):
            logger.info(
                "跳过重复信号入库 %s %s（%s 分钟内已有同类型）",
                code,
                sig_type,
                win,
            )
            continue
        bucket = _signal_time_to_minute_bucket(sig_time)
        if signal_exists_in_minute(code, bucket, sig_type):
            continue
        insert_signal_record(
            stock_code=code,
            stock_name=stock_name,
            signal_time=bucket + ":00" if len(bucket) == 16 else sig_time,
            signal_price=float(sig["signal_price"]),
            signal_type=sig_type,
            reason=str(sig.get("reason") or ""),
            realtime_score=realtime_score,
        )
        written += 1
    return written



def _push_signal_to_dingtalk(
    stock_code: str, stock_name: str,
    signals: list[dict[str, Any]],
    score: float | None,
) -> None:
    """Push buy signal alert to DingTalk."""
    buy_sigs = [s for s in signals if str(s.get("signal_type", "")).upper() == "BUY"]
    if not buy_sigs:
        return
    from .config_manager import config_manager
    from .dingtalk_notifier import DingTalkNotifier
    config_manager.reload()
    wh = config_manager.get_dingtalk_webhook_url()
    if not wh:
        return
    try:
        notifier = DingTalkNotifier(wh, config_manager.get_dingtalk_secret() or None)
    except Exception:
        return
    code = str(stock_code).zfill(6)
    name = str(stock_name)
    now_s = time.strftime("%H:%M:%S")
    lines = [f"## 🟢 买入信号\n\n**{name}** ({code})\n\n"]
    for s in buy_sigs:
        price = s.get("signal_price", "?")
        reason = str(s.get("reason", "")).strip()
        t = str(s.get("signal_time", ""))
        lines.append(f"- ⏱ {t}  |  💰 {price}  |  {reason}\n")
    if score is not None:
        lines.append(f"\n📊 量化评分: {score:.1f}\n")
    lines.append("\n⚠️ 信号仅供参考，不构成投资建议")
    text = "".join(lines)
    notifier.send_markdown(f"买入信号 {name}", text)


def run_monitor_cycle_for_targets(
    targets: list[dict[str, Any]],
    *,
    persist: bool = True,
    allow_off_session_display: bool = False,
) -> list[dict[str, Any]]:
    """
    对 Top3 列表执行一轮监控；返回每只股票的面板数据（含分钟线、指标、信号）。
    非交易时段默认不拉取；``allow_off_session_display=True`` 时仅展示、不写库。
    """
    in_session = is_a_share_intraday_session()
    if not in_session and not allow_off_session_display:
        return []
    if not in_session:
        persist = False
    init_db()
    if _monitor_cooling():
        logger.warning(
            "实时监控处于冷却期（至 %.0f），本轮不拉取行情",
            _pause_until_wall,
        )
        return []
    panels: list[dict[str, Any]] = []
    for ti, t in enumerate(targets[:TOP_N_SELECTION]):
        if ti > 0:
            _sleep_inter_request_jitter()
        code = str(t.get("stock_code", "")).strip().zfill(6)
        name = str(t.get("stock_name") or "").strip()
        score_raw = t.get("score")
        try:
            rt_score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            rt_score = None
        try:
            _rpm_guard_before_request()
            df = fetch_today_minute_bars(code)
            _register_api_success()
        except Exception as exc:
            logger.warning("分钟线拉取失败 %s: %s", code, exc)
            _register_api_failure()
            df = None
        if df is None or df.empty:
            panels.append(
                {
                    "stock_code": code,
                    "stock_name": name,
                    "rank": t.get("rank"),
                    "realtime_score": rt_score,
                    "minute_df": None,
                    "signals_today": fetch_signal_history_for_stock_on_date(code),
                    "error": "empty",
                }
            )
            continue
        new_sigs = detect_intraday_signals(df)
        if persist and new_sigs:
            n_written = persist_signals_for_stock(code, name, new_sigs, realtime_score=rt_score)
            if n_written > 0:
                _push_signal_to_dingtalk(code, name, new_sigs, rt_score)
        today_sigs = fetch_signal_history_for_stock_on_date(code)
        last = df.iloc[-1]
        open_px = float(df["open"].iloc[0])
        last_px = float(last["close"])
        pct = (last_px / open_px - 1.0) * 100.0 if open_px > 0 else 0.0
        panels.append(
            {
                "stock_code": code,
                "stock_name": name,
                "rank": t.get("rank"),
                "realtime_score": rt_score,
                "minute_df": df,
                "latest_price": last_px,
                "pct_chg": pct,
                "volume": float(last["volume"]),
                "signals_today": today_sigs,
                "new_signals": new_sigs,
                "error": None,
            }
        )
    return panels


def run_top3_monitor_cycle(
    *,
    persist: bool = True,
    allow_off_session_display: bool = False,
) -> list[dict[str, Any]]:
    """从 daily_selections 取当日 Top3 并执行监控（默认仅交易时段写库）。"""
    targets = fetch_top3_selections_for_monitor()
    if not targets:
        return []
    return run_monitor_cycle_for_targets(
        targets,
        persist=persist,
        allow_off_session_display=allow_off_session_display,
    )


def signals_to_display_dataframe(signals: list[dict[str, Any]] | None) -> pd.DataFrame:
    """signal_history 行列表 → 表格展示用 DataFrame（中文列名）。"""
    if not signals:
        return pd.DataFrame(
            columns=["触发时间", "价格", "信号类型", "触发理由"],
        )
    rows = []
    for s in signals:
        rows.append(
            {
                "触发时间": str(s.get("signal_time") or ""),
                "价格": float(s["signal_price"])
                if s.get("signal_price") is not None
                and str(s.get("signal_price")).strip() != ""
                else float("nan"),
                "信号类型": str(s.get("signal_type") or ""),
                "触发理由": str(s.get("reason") or ""),
            }
        )
    return pd.DataFrame(rows)


def build_intraday_dashboard_figure(
    panel: dict[str, Any],
    *,
    height: int = 520,
):
    """
    分时多指标看板：主图（分时 + VWAP + 九转数字 + 买点）、成交量红绿柱、K/D/J。
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = panel.get("minute_df")
    if df is None or getattr(df, "empty", True):
        return None

    times = df["time"].dt.strftime("%H:%M").tolist()
    close = df["close"].astype(float)
    open_px = df["open"].astype(float) if "open" in df.columns else close.shift(1).fillna(close)
    vol_raw = df["volume"].astype(float)
    vol = vol_raw.diff().fillna(vol_raw.iloc[0])  # per-minute volume
    vwap = df["vwap"].astype(float)
    k_s = df["k"].astype(float)
    d_s = df["d"].astype(float)
    j_s = df["j"].astype(float)

    pct = float(panel.get("pct_chg") or 0.0)
    line_color = "#dc2626" if pct >= 0 else "#16a34a"
    vol_colors = [
        _COLOR_VOL_UP if float(c) >= float(o) else _COLOR_VOL_DN
        for c, o in zip(close.tolist(), open_px.tolist())
    ]

    buy_td, sell_td = compute_td_sequential_labels(close.to_numpy(dtype=float))

    prev_close = panel.get("prev_close")
    if prev_close is None:
        c4 = str(panel.get("stock_code", "")).zfill(6)
        try:
            from .database import fetch_stock_daily_bars_until
            dd = fetch_stock_daily_bars_until(c4, datetime.now().strftime("%Y-%m-%d"))
            if dd is not None and len(dd) >= 2:
                prev_close = float(dd.iloc[-2]["close"])
        except Exception:
            prev_close = None
    if prev_close is None or prev_close <= 0:
        prev_close = float(close.iloc[0])
    cum_amt = (close * vol).cumsum()
    cum_vol = vol.cumsum().replace(0, float("nan"))
    avg_line = (cum_amt / cum_vol).ffill().bfill()
    pct_from_pc = (close - prev_close) / prev_close * 100.0
    pct_rng = max(abs(float(pct_from_pc.min())), abs(float(pct_from_pc.max()))) * 1.15
    pct_rng = max(pct_rng, 1.5)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.22, 0.28],
        subplot_titles=("分时图", "成交量", "KDJ"),
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=close,
            mode="lines",
            name="价格",
            line=dict(color=line_color, width=2.2),
            customdata=pct_from_pc,
            hovertemplate="%{x}<br>价格: %{y:.2f}<br>涨跌: %{customdata:+.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=avg_line,
            mode="lines",
            name="均线",
            line=dict(color="#f59e0b", width=1.3),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=prev_close,
        line=dict(color="#94a3b8", width=1, dash="dash"),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[times[0], times[-1]],
            y=[prev_close, prev_close],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
        secondary_y=True,
    )

    bx_buy, by_buy, bt_buy = [], [], []
    bx_sell, by_sell, bt_sell = [], [], []
    y_low = float(close.min()) if len(close) else 0.0
    y_high = float(close.max()) if len(close) else 0.0
    y_span = max(y_high - y_low, 1e-6)
    for i, t in enumerate(times):
        if buy_td[i] is not None:
            bx_buy.append(t)
            by_buy.append(float(close.iloc[i]) - 0.02 * y_span)
            bt_buy.append(buy_td[i])
        if sell_td[i] is not None:
            bx_sell.append(t)
            by_sell.append(float(close.iloc[i]) + 0.02 * y_span)
            bt_sell.append(sell_td[i])
    if bx_buy:
        fig.add_trace(
            go.Scatter(
                x=bx_buy,
                y=by_buy,
                mode="text",
                text=bt_buy,
                textposition="bottom center",
                textfont=dict(size=11, color="#0f766e", family="Arial Black"),
                name="九转(低)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if bx_sell:
        fig.add_trace(
            go.Scatter(
                x=bx_sell,
                y=by_sell,
                mode="text",
                text=bt_sell,
                textposition="top center",
                textfont=dict(size=11, color="#c2410c", family="Arial Black"),
                name="九转(高)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    sigs = panel.get("signals_today") or []
    if sigs:
        sig_df = pd.DataFrame(sigs)
        sig_df["hm"] = pd.to_datetime(sig_df["signal_time"], errors="coerce").dt.strftime(
            "%H:%M"
        )
        hm_set = set(times)
        xs, ys, texts = [], [], []
        for _, row in sig_df.iterrows():
            hm = str(row.get("hm") or "")
            if hm not in hm_set:
                continue
            idx = times.index(hm)
            xs.append(hm)
            ys.append(float(row.get("signal_price") or close.iloc[idx]))
            texts.append("买")
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=texts,
                    textposition="top center",
                    textfont=dict(size=15, color=_COLOR_BUY, family="Arial Black"),
                    marker=dict(
                        symbol="triangle-up",
                        size=16,
                        color=_COLOR_BUY,
                        line=dict(width=1, color="#14532d"),
                    ),
                    name="买点",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(
            x=times,
            y=vol,
            name="成交量",
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=times, y=k_s, mode="lines", name="K", line=dict(color="#0d9488", width=1.4)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=d_s, mode="lines", name="D", line=dict(color="#ea580c", width=1.4)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=j_s, mode="lines", name="J", line=dict(color="#7c3aed", width=1.2)),
        row=3,
        col=1,
    )

    code = str(panel.get("stock_code", "")).zfill(6)
    name = str(panel.get("stock_name") or "")
    fig.update_layout(
        title_text=f"#{panel.get('rank')} {code} {name} · 分时看板",
        title_font=dict(size=15, color="#0f172a"),
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#f8fafc",
        height=height,
        margin=dict(l=48, r=20, t=56, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#334155", size=11),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9, color="#64748b"))
    for ri in (1, 2, 3):
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(15,23,42,0.06)",
            tickfont=dict(size=9),
            row=ri,
            col=1,
        )
    fig.update_yaxes(
        title_text="%",
        range=[-pct_rng, pct_rng],
        showgrid=False,
        zeroline=True,
        zerolinecolor="#94a3b8",
        zerolinewidth=1,
        tickformat="+.2f",
        ticksuffix="%",
        dtick=pct_rng / 4,
        tickfont=dict(size=9, color="#64748b"),
        row=1, col=1,
        secondary_y=True,
    )
    fig.update_xaxes(showticklabels=True, row=3, col=1)
    return fig


def build_intraday_monitor_figure(
    panel: dict[str, Any],
    *,
    height: int = 520,
):
    """兼容旧名：等价于 ``build_intraday_dashboard_figure``。"""
    return build_intraday_dashboard_figure(panel, height=height)
