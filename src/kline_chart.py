"""
K 线图模块：优先从本地 SQLite 读取日线；缺失时在线拉取并写回缓存。
"""
from __future__ import annotations

import json
import requests
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ensure_eastmoney_no_proxy_if_configured
from .data_fetcher import fetch_daily_hist
from .database import get_connection, upsert_stock_daily_klines
from .utils import get_kline_incremental_end_trade_date


def lookup_stock_display_name(stock_code: object) -> str:
    """从本地 ``stock_daily_kline`` 取该股最新一条记录里的名称（使用统一标准连接池）。"""
    code = str(stock_code).strip().zfill(6)
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT stock_name FROM stock_daily_kline
                WHERE stock_code = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (code,),
            )
            row = cur.fetchone()
            if row and row[0]:
                n = str(row[0]).strip()
                if n and n != "未知":
                    return n
    except Exception as exc:
        print(f"[DB] lookup_stock_display_name 失败: {exc}")
    return ""


def _append_ma(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True)
    out["ma5"] = out["close"].rolling(window=5).mean()
    out["ma10"] = out["close"].rolling(window=10).mean()
    out["ma20"] = out["close"].rolling(window=20).mean()
    return out


def _merge_online_tail_if_local_stale(
    df_local: pd.DataFrame,
    stock_code: str,
    *,
    calendar_days: int,
) -> pd.DataFrame:
    if df_local is None or df_local.empty:
        return df_local
    work = df_local.copy()
    work["date"] = pd.to_datetime(work["date"])
    anchor = str(get_kline_incremental_end_trade_date()).strip()[:10]
    last_s = work["date"].max().strftime("%Y-%m-%d")
    if last_s >= anchor:
        return work

    start_compact = (work["date"].max() - timedelta(days=20)).strftime("%Y%m%d")
    end_compact = datetime.now().strftime("%Y%m%d")
    try:
        tail = fetch_daily_hist(
            stock_code,
            start_date=start_compact,
            end_date=end_compact,
            adjust="qfq",
        )
    except Exception:
        return work

    if tail is None or tail.empty:
        return work

    for col in ("open", "high", "low", "close", "volume"):
        if col in tail.columns:
            tail[col] = pd.to_numeric(tail[col], errors="coerce")
    tail["date"] = pd.to_datetime(tail["date"])
    keep = ["date", "open", "high", "low", "close", "volume"]
    tail = tail[keep].dropna(subset=["open", "high", "low", "close"])
    if tail.empty:
        return work

    merged = pd.concat([work[keep], tail], ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    limit_dt = datetime.now() - timedelta(days=int(calendar_days))
    merged = merged[merged["date"] >= pd.Timestamp(limit_dt)].reset_index(drop=True)

    try:
        disp = lookup_stock_display_name(stock_code) or "未知"
        save = tail.copy()
        save["date"] = save["date"].dt.strftime("%Y-%m-%d")
        recs = [
            {
                "date": r["date"],
                "stock_code": stock_code,
                "stock_name": disp,
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
            }
            for _, r in save.iterrows()
        ]
        upsert_stock_daily_klines(recs)
    except Exception:
        pass

    return merged


def get_stock_kline_data(stock_code: object, days: int = 365):
    """优先读本地 ``stock_daily_kline``；不足则 ``fetch_daily_hist`` 补全并 UPSERT。"""
    stock_code = str(stock_code).strip().zfill(6)

    try:
        limit_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
            SELECT date, open, high, low, close, volume
            FROM stock_daily_kline
            WHERE stock_code = ? AND date >= ?
            ORDER BY date ASC
        """
        with get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=(stock_code, limit_date))

        if df is not None and not df.empty and len(df) >= 30:
            df["date"] = pd.to_datetime(df["date"])
            df = _merge_online_tail_if_local_stale(df, stock_code, calendar_days=days)
            return _append_ma(df)

    except Exception as exc:
        print(f"从本地数据库读取K线失败: {exc}")

    print(f"提示: 本地未检索到 {stock_code} 的完整行情，正在在线抓取补全...")
    end_compact = datetime.now().strftime("%Y%m%d")
    start_compact = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    try:
        df = fetch_daily_hist(
            stock_code,
            start_date=start_compact,
            end_date=end_compact,
            adjust="qfq",
        )
        if df is not None and len(df) > 0:
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]].dropna(
                subset=["open", "high", "low", "close"]
            )

            try:
                save = df.copy()
                save["date"] = save["date"].dt.strftime("%Y-%m-%d")
                recs = [
                    {
                        "date": r["date"],
                        "stock_code": stock_code,
                        "stock_name": "未知",
                        "open": r["open"],
                        "high": r["high"],
                        "low": r["low"],
                        "close": r["close"],
                        "volume": r["volume"],
                    }
                    for _, r in save.iterrows()
                ]
                upsert_stock_daily_klines(recs)
            except Exception:
                pass

            return _append_ma(df)
    except Exception as exc:
        print(f"在线补充拉取失败: {exc}")

    return None


def draw_candlestick(df, stock_code, stock_name):
    """绘制无留白连续 K 线图。"""
    if df is None or len(df) == 0:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    if "ma5" not in df.columns:
        df = _append_ma(df)

    date_strs = df["date"].dt.strftime("%Y-%m-%d").tolist()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    color_up = "#FF3333"
    color_down = "#00CC66"

    ohlc_cd = df[["open", "high", "low", "close"]].to_numpy(dtype=float)

    closes = df["close"].astype(float)
    pct_vs_prev_close: list[str] = []
    for i in range(len(df)):
        if i == 0:
            pct_vs_prev_close.append("—")
            continue
        prev_c = float(closes.iloc[i - 1])
        cur_c = float(closes.iloc[i])
        if prev_c <= 0 or pd.isna(prev_c) or pd.isna(cur_c):
            pct_vs_prev_close.append("—")
            continue
        p = (cur_c / prev_c - 1.0) * 100.0
        pct_vs_prev_close.append(f"{p:+.2f}%")

    customdata = [
        [
            float(ohlc_cd[i, 0]),
            float(ohlc_cd[i, 1]),
            float(ohlc_cd[i, 2]),
            float(ohlc_cd[i, 3]),
            pct_vs_prev_close[i],
        ]
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma5"],
            name="MA5",
            hoverinfo="skip",
            mode="lines",
            line=dict(color="#0d9488", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma10"],
            name="MA10",
            hoverinfo="skip",
            mode="lines",
            line=dict(color="#9333ea", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma20"],
            name="MA20",
            hoverinfo="skip",
            mode="lines",
            line=dict(color="#ca8a04", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=date_strs,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            customdata=customdata,
            hovertemplate=(
                "<b>日期</b>: %{x}<br>"
                "<b>开盘</b>: %{customdata[0]:.2f}<br>"
                "<b>最高</b>: %{customdata[1]:.2f}<br>"
                "<b>最低</b>: %{customdata[2]:.2f}<br>"
                "<b>收盘</b>: %{customdata[3]:.2f}<br>"
                "<b>涨跌幅</b>: %{customdata[4]}<extra></extra>"
            ),
            increasing_line_color=color_up,
            increasing_fillcolor=color_up,
            decreasing_line_color=color_down,
            decreasing_fillcolor=color_down,
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        color_up if c >= o else color_down for c, o in zip(df["close"], df["open"])
    ]

    hover_text_volume = []
    for pos, (_, row) in enumerate(df.iterrows()):
        date_str = date_strs[pos]
        vol_hand = float(row["volume"]) / 100.0
        text = (
            f"<b>日期</b>: {date_str}<br>"
            f"<b>成交量</b>: {vol_hand:.2f} 手<br>"
            f"<b>涨跌幅</b>: {pct_vs_prev_close[pos]}"
        )
        hover_text_volume.append(text)

    fig.add_trace(
        go.Bar(
            x=date_strs,
            y=df["volume"],
            name="成交量",
            text=hover_text_volume,
            hoverinfo="text",
            marker_color=volume_colors,
            marker_line_width=0,
        ),
        row=2,
        col=1,
    )

    color_grid = "rgba(15, 23, 42, 0.08)"

    fig.update_layout(
        title=f"{stock_code} {stock_name} (连续K线)",
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#f8fafc",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#0d9488",
            font=dict(color="#0f172a", size=13, family="monospace"),
        ),
    )

    for r in (1, 2):
        fig.update_xaxes(
            type="category",
            tickangle=-45,
            nticks=20,
            showgrid=False,
            linecolor=color_grid,
            tickfont=dict(color="#475569"),
            row=r,
            col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=color_grid,
            zeroline=False,
            linecolor=color_grid,
            tickfont=dict(color="#475569"),
            row=r,
            col=1,
        )

    return fig


def _fetch_tencent_minute_data(stock_code: str):
    """
    腾讯财经分时兜底；返回 (DataFrame, trade_date) 或 (None, None)。
    休市时通常返回最近一个交易日的分时。
    """
    code = str(stock_code).strip().zfill(6)
    prefix = "sh" if code.startswith("6") else "sz"
    symbol = f"{prefix}{code}"
    try:
        s = requests.Session()
        s.trust_env = False
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/minute/query"
            f"?_var=min_data&code={symbol}"
        )
        resp = s.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://gu.qq.com/",
            },
            timeout=15,
        )
        text = resp.text
        json_str = text[text.find("{") :]
        data = json.loads(json_str)
        stock_data = data.get("data", {}).get(symbol, {}).get("data", {})
        bars = stock_data.get("data", [])
        trade_date = stock_data.get("date", "")
        if not bars:
            return None, None
        rows = []
        for bar in bars:
            parts = str(bar).split()
            if len(parts) < 2:
                continue
            time_str = parts[0]
            hour = time_str[:2]
            minute = time_str[2:4]
            ts = (
                f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]} "
                f"{hour}:{minute}:00"
            )
            rows.append(
                {
                    "time": ts,
                    "open": float(parts[1]),
                    "close": float(parts[1]),
                    "high": float(parts[1]),
                    "low": float(parts[1]),
                    "volume": float(parts[2]) if len(parts) > 2 else 0.0,
                    "amount": float(parts[3]) if len(parts) > 3 else 0.0,
                }
            )
        return pd.DataFrame(rows), trade_date
    except Exception:
        return None, None


def _sanitize_minute_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """分钟线 OHLCV 防御性清洗：缺失列补齐、前后向填充，避免监控/信号层 KeyError。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "close" not in out.columns:
        return pd.DataFrame()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = out["close"] if col != "volume" else 0.0
        out[col] = (
            pd.to_numeric(out[col], errors="coerce").ffill().bfill().fillna(0.0)
        )
    close = out["close"].astype(float)
    vol = out["volume"].astype(float)
    if "amount" in out.columns:
        amt = pd.to_numeric(out["amount"], errors="coerce")
        proxy = close * vol
        out["amount"] = amt.where(amt.notna() & (amt > 0), proxy)
    else:
        out["amount"] = close * vol
    out["amount"] = out["amount"].ffill().bfill().fillna(0.0)
    return out


def get_realtime_min_data(stock_code: object) -> pd.DataFrame | None:
    """获取当日（或最近交易日）1 分钟线；优先腾讯，失败则 AkShare。"""
    import akshare as ak

    ensure_eastmoney_no_proxy_if_configured()
    stock_code = str(stock_code).strip().zfill(6)
    today_str = datetime.now().strftime("%Y-%m-%d")
    start_s = f"{today_str} 09:25:00"
    end_s = f"{today_str} 15:05:00"

    fetch_date_str = today_str
    df, tc_date = _fetch_tencent_minute_data(stock_code)
    if tc_date:
        fetch_date_str = f"{tc_date[:4]}-{tc_date[4:6]}-{tc_date[6:8]}"

    if df is None or df.empty:
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=stock_code,
                start_date=start_s,
                end_date=end_s,
                period="1",
                adjust="qfq",
            )
        except Exception:
            df = None

    if df is None or df.empty:
        return None

    rename_map = {
        "时间": "time",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    need = {"time", "open", "close", "high", "low", "volume"}
    if not need.issubset(df.columns):
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[df["time"].dt.strftime("%Y-%m-%d") == fetch_date_str].reset_index(drop=True)
    for c in ("open", "close", "high", "low", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"])
    df = _sanitize_minute_ohlcv(df)
    if df.empty or df["close"].isna().all():
        return None
    return df


def draw_realtime_line_chart(
    df: pd.DataFrame | None,
    stock_code: object,
    stock_name: object,
) -> go.Figure | None:
    """绘制浅色主题下的极简分时折线图。"""
    if df is None or df.empty:
        return None

    open_price = float(df["open"].iloc[0])
    latest_price = float(df["close"].iloc[-1])
    pct_chg = (latest_price / open_price - 1.0) * 100.0 if open_price > 0 else 0.0

    theme_color = "#FF3333" if pct_chg >= 0 else "#00CC66"
    fill_color = (
        "rgba(255, 51, 51, 0.05)" if pct_chg >= 0 else "rgba(0, 204, 102, 0.05)"
    )

    times = df["time"].dt.strftime("%H:%M").tolist()
    code_s = str(stock_code).strip().zfill(6)
    name_s = str(stock_name).strip()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=df["close"],
            mode="lines",
            name="分时价",
            line=dict(color=theme_color, width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor=fill_color,
        )
    )

    fig.update_layout(
        title=f"⏳ {code_s} {name_s} ({pct_chg:+.2f}%)",
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#f8fafc",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=180,
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        visible=True,
        tickfont=dict(color="#64748b", size=9),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=True,
        tickfont=dict(color="#64748b", size=9),
    )

    return fig


def draw_tonghuashun_intraday(
    df: pd.DataFrame,
    stock_code: str,
    stock_name: str,
    prev_close: float,
) -> go.Figure | None:
    """同花顺风格分时图：价格线 + 均线 + 昨收基准线，右轴百分比。"""
    if df is None or df.empty:
        return None

    close = df["close"].astype(float)
    times = df["time"].dt.strftime("%H:%M").tolist()
    latest_price = float(close.iloc[-1])

    volume = df["volume"].astype(float)
    cum_amt = (close * volume).cumsum()
    cum_vol = volume.cumsum()
    avg_price = (cum_amt / cum_vol.replace(0, float("nan"))).ffill().bfill()

    pct_y = (close - prev_close) / prev_close * 100.0
    pct_latest = (latest_price - prev_close) / prev_close * 100.0

    theme_color = "#dc2626" if pct_latest >= 0 else "#16a34a"
    fill_color = (
        "rgba(220,38,38,0.06)" if pct_latest >= 0 else "rgba(22,163,74,0.06)"
    )

    code_s = str(stock_code).strip().zfill(6)
    name_s = str(stock_name).strip()

    fig = go.Figure()
    fig.add_hline(y=prev_close, line=dict(color="#94a3b8", width=1, dash="dash"))

    fig.add_trace(
        go.Scatter(
            x=times,
            y=close,
            mode="lines",
            name="价格",
            line=dict(color=theme_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            customdata=pct_y,
            hovertemplate=(
                "%{x}<br>价格: %{y:.2f}<br>涨跌: %{customdata:+.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=avg_price,
            mode="lines",
            name="均线",
            line=dict(color="#f59e0b", width=1.2),
            hoverinfo="skip",
        )
    )

    pct_range = max(
        max(abs(float(pct_y.min())), abs(float(pct_y.max()))) * 1.15,
        2.0,
    )

    fig.update_layout(
        title=f"{code_s} {name_s}  {latest_price:.2f}  {pct_latest:+.2f}%",
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#f8fafc",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=60, t=50, b=10),
        height=400,
        hovermode="x unified",
    )

    fig.update_yaxes(
        title_text="价格",
        showgrid=True,
        gridcolor="rgba(15,23,42,0.06)",
        side="left",
    )
    fig.update_layout(
        yaxis2=dict(
            title_text="涨跌幅",
            overlaying="y",
            side="right",
            range=[-pct_range, pct_range],
            tickformat="+.1f",
            ticksuffix="%",
            tickfont=dict(color="#64748b", size=10),
            zeroline=True,
            zerolinecolor="#94a3b8",
        )
    )

    return fig
