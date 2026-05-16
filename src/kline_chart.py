"""
K 线图模块：优先从本地 SQLite 读取日线；缺失时在线拉取并写回缓存。
"""
from __future__ import annotations

import sqlite3
import requests
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import DB_PATH, ensure_eastmoney_no_proxy_if_configured
from .data_fetcher import fetch_daily_hist
from .database import upsert_stock_daily_klines
from .utils import get_kline_incremental_end_trade_date


def lookup_stock_display_name(stock_code: object) -> str:
    """从本地 ``stock_daily_kline`` 取该股最新一条记录里的名称（同步脚本写入）。"""
    code = str(stock_code).strip().zfill(6)
    try:
        conn = sqlite3.connect(str(DB_PATH))
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
        conn.close()
        if row and row[0]:
            n = str(row[0]).strip()
            if n and n != "未知":
                return n
    except Exception:
        pass
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
    """
    本地最后一根若早于「截至今日的最近交易日」，则用在线日线补尾部并 UPSERT，
    避免出现图表停在昨日、而增量脚本尚未覆盖该股的情况。
    """
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
        conn = sqlite3.connect(str(DB_PATH))
        limit_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
            SELECT date, open, high, low, close, volume
            FROM stock_daily_kline
            WHERE stock_code = ? AND date >= ?
            ORDER BY date ASC
        """
        df = pd.read_sql_query(sql, conn, params=(stock_code, limit_date))
        conn.close()

        if df is not None and not df.empty and len(df) >= 30:
            df["date"] = pd.to_datetime(df["date"])
            df = _merge_online_tail_if_local_stale(df, stock_code, calendar_days=days)
            return _append_ma(df)

    except Exception as exc:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
        print(f"在线补充拉取失败: {exc}")

    return None


def draw_candlestick(df, stock_code, stock_name):
    """绘制无留白连续 K 线图（中文悬浮提示）。"""
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

    # Candlestick 不支持可靠的 text+hoverinfo；用 customdata + hovertemplate。
    # 均线后画会挡住 K 线悬浮命中：先画均线并跳过 hover，最后画 K 线置于顶层。
    ohlc_cd = df[["open", "high", "low", "close"]].to_numpy(dtype=float)

    # 相对前一交易日收盘涨跌幅（与行情软件「日涨跌幅」一致）
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
        [float(ohlc_cd[i, 0]), float(ohlc_cd[i, 1]), float(ohlc_cd[i, 2]), float(ohlc_cd[i, 3]), pct_vs_prev_close[i]]
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

    fig.update_xaxes(
        type="category",
        tickangle=-45,
        nticks=20,
        showgrid=False,
        linecolor=color_grid,
        tickfont=dict(color="#475569"),
        showspikes=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        type="category",
        tickangle=-45,
        nticks=20,
        showgrid=False,
        linecolor=color_grid,
        tickfont=dict(color="#475569"),
        showspikes=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=color_grid,
        zeroline=False,
        linecolor=color_grid,
        tickfont=dict(color="#475569"),
        showspikes=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=color_grid,
        zeroline=False,
        linecolor=color_grid,
        tickfont=dict(color="#475569"),
        showspikes=False,
        row=2,
        col=1,
    )

    return fig


def _fetch_tencent_minute_data(stock_code: str):
    """
    Fallback: get 1-min data from Tencent finance.
    Returns (DataFrame, date_str) or (None, None).
    Automatically returns the most recent trading day when market is closed.
    """
    code = str(stock_code).strip().zfill(6)
    prefix = 'sh' if code.startswith('6') else 'sz'
    symbol = f'{prefix}{code}'
    import json
    try:
        s = requests.Session()
        s.trust_env = False
        url = f'https://web.ifzq.gtimg.cn/appstock/app/minute/query?_var=min_data&code={symbol}'
        resp = s.get(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://gu.qq.com/',
        }, timeout=15)
        text = resp.text
        json_str = text[text.find('{'):]
        data = json.loads(json_str)
        stock_data = data.get('data', {}).get(symbol, {}).get('data', {})
        bars = stock_data.get('data', [])
        trade_date = stock_data.get('date', '')
        if not bars:
            return None, None
        rows = []
        for bar in bars:
            parts = str(bar).split()
            if len(parts) < 2:
                continue
            time_str = parts[0]  # HHMM
            hour = time_str[:2]
            minute = time_str[2:4]
            ts = f'{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]} {hour}:{minute}:00'
            rows.append({
                'time': ts,
                'open': float(parts[1]),
                'close': float(parts[1]),  # minute query only has price
                'high': float(parts[1]),
                'low': float(parts[1]),
                'volume': float(parts[2]) if len(parts) > 2 else 0.0,
                'amount': float(parts[3]) if len(parts) > 3 else 0.0,
            })
        return pd.DataFrame(rows), trade_date
    except Exception:
        return None, None


def get_realtime_min_data(stock_code: object) -> pd.DataFrame | None:
    """
    Get 1-min intraday data. Tries AkShare first, then Tencent API.
    For off-session, Tencent returns the most recent trading day automatically.
    """
    import akshare as ak
    import json as _json

    ensure_eastmoney_no_proxy_if_configured()
    stock_code = str(stock_code).strip().zfill(6)
    today_str = datetime.now().strftime("%Y-%m-%d")
    start_s = f"{today_str} 09:25:00"
    end_s = f"{today_str} 15:05:00"

    df = None
    # Method 1: Tencent (primary)
    fetch_date_str = today_str
    df, tc_date = _fetch_tencent_minute_data(stock_code)
    if tc_date:
        fetch_date_str = f'{tc_date[:4]}-{tc_date[4:6]}-{tc_date[6:8]}' 

    # Method 2: AkShare fallback
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

    # Rename columns for compatibility
    rename_map = {
        "时间": "time", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
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
    df = df.dropna(subset=["close"])
    return df if not df.empty else None
def draw_realtime_line_chart(
    df: pd.DataFrame | None,
    stock_code: object,
    stock_name: object,
) -> go.Figure | None:
    """
    绘制浅色主题下的极简分时折线图（无网格、面积填充）。
    """
    if df is None or df.empty:
        return None

    open_price = float(df["open"].iloc[0])
    latest_price = float(df["close"].iloc[-1])
    if open_price > 0:
        pct_chg = (latest_price / open_price - 1.0) * 100.0
    else:
        pct_chg = 0.0

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

def draw_tonghuashun_intraday(
    df: pd.DataFrame,
    stock_code: str,
    stock_name: str,
    prev_close: float,
) -> go.Figure | None:
    """
    同花顺风格分时图：价格线 + 均线 + 昨收基准线，右轴百分比。
    """
    if df is None or df.empty:
        return None

    close = df["close"].astype(float)
    times = df["time"].dt.strftime("%H:%M").tolist()
    open_price = float(close.iloc[0])
    latest_price = float(close.iloc[-1])

    # Compute average price (cumulative VWAP-style)
    volume = df["volume"].astype(float)
    cum_amt = (close * volume).cumsum()
    cum_vol = volume.cumsum()
    avg_price = cum_amt / cum_vol.replace(0, float("nan"))
    avg_price = avg_price.ffill().bfill()

    # Percentage from prev_close
    pct_y = (close - prev_close) / prev_close * 100.0
    pct_latest = (latest_price - prev_close) / prev_close * 100.0

    # Color: red if up, green if down (A-share convention)
    theme_color = "#dc2626" if pct_latest >= 0 else "#16a34a"
    fill_color = "rgba(220,38,38,0.06)" if pct_latest >= 0 else "rgba(22,163,74,0.06)"

    # Determine tick colors for the price line
    tick_colors = [
        "#dc2626" if i > 0 and float(close.iloc[i]) >= float(close.iloc[i-1])
        else "#16a34a"
        for i in range(len(close))
    ]

    code_s = str(stock_code).strip().zfill(6)
    name_s = str(stock_name).strip()

    fig = go.Figure()

    # Previous close reference line (full width)
    fig.add_hline(
        y=prev_close,
        line=dict(color="#94a3b8", width=1, dash="dash"),
        annotation_text=f"昨收 {prev_close:.2f}",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#64748b"),
    )

    # Price line with area fill
    fig.add_trace(
        go.Scatter(
            x=times,
            y=close,
            mode="lines",
            name="价格",
            line=dict(color=theme_color, width=2, shape="linear"),
            fill="tozeroy",
            fillcolor=fill_color,
            yaxis="y",
            customdata=pct_y,
            hovertemplate="%{x}<br>价格: %{y:.2f}<br>涨跌: %{customdata:+.2f}%<extra></extra>",
        )
    )

    # Average price line (orange)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=avg_price,
            mode="lines",
            name="均线",
            line=dict(color="#f59e0b", width=1.2, dash="solid"),
            yaxis="y",
            hoverinfo="skip",
        )
    )

    pct_min = float(pct_y.min())
    pct_max = float(pct_y.max())
    pct_range = max(abs(pct_min), abs(pct_max)) * 1.15
    pct_range = max(pct_range, 2.0)

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

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#64748b", size=10),
    )

    fig.update_yaxes(
        title_text="价格",
        showgrid=True,
        gridcolor="rgba(15,23,42,0.06)",
        zeroline=False,
        tickfont=dict(color="#334155", size=10),
        side="left",
    )

    # Right Y-axis: percentage from prev_close
    fig.update_layout(
        yaxis2=dict(
            title_text="涨跌幅",
            overlaying="y",
            side="right",
            range=[-pct_range, pct_range],
            showgrid=False,
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1,
            tickformat="+.1f",
            ticksuffix="%",
            tickfont=dict(color="#64748b", size=10),
        )
    )

    return fig


    return fig
