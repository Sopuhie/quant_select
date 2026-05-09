"""
K 线图模块：优先从本地 SQLite 读取日线；缺失时在线拉取并写回缓存。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import DB_PATH
from .data_fetcher import fetch_daily_hist
from .database import upsert_stock_daily_klines


def _append_ma(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True)
    out["ma5"] = out["close"].rolling(window=5).mean()
    out["ma10"] = out["close"].rolling(window=10).mean()
    out["ma20"] = out["close"].rolling(window=20).mean()
    return out


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

    hover_text_candlestick = []
    for pos, (_, row) in enumerate(df.iterrows()):
        date_str = date_strs[pos]
        text = (
            f"<b>日期</b>: {date_str}<br>"
            f"<b>开盘</b>: {row['open']:.2f}<br>"
            f"<b>最高</b>: {row['high']:.2f}<br>"
            f"<b>最低</b>: {row['low']:.2f}<br>"
            f"<b>收盘</b>: {row['close']:.2f}"
        )
        hover_text_candlestick.append(text)

    fig.add_trace(
        go.Candlestick(
            x=date_strs,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            text=hover_text_candlestick,
            hoverinfo="text",
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
        text = f"<b>日期</b>: {date_str}<br><b>成交量</b>: {vol_hand:.2f} 手"
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

    def get_ma_hover_text(ma_series, name="MA"):
        hover_texts = []
        for pos, val in enumerate(ma_series):
            date_str = date_strs[pos]
            val_str = f"{val:.2f}" if pd.notna(val) else "—"
            hover_texts.append(f"<b>日期</b>: {date_str}<br><b>{name}</b>: {val_str}")
        return hover_texts

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma5"],
            name="MA5",
            text=get_ma_hover_text(df["ma5"], "MA5"),
            hoverinfo="text",
            line=dict(color="#00FFCC", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma10"],
            name="MA10",
            text=get_ma_hover_text(df["ma10"], "MA10"),
            hoverinfo="text",
            line=dict(color="#FF00FF", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma20"],
            name="MA20",
            text=get_ma_hover_text(df["ma20"], "MA20"),
            hoverinfo="text",
            line=dict(color="#FFFF00", width=1.5),
        ),
        row=1,
        col=1,
    )

    color_grid = "rgba(255, 255, 255, 0.04)"

    fig.update_layout(
        title=f"{stock_code} {stock_name} (连续K线)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
        hoverlabel=dict(
            bgcolor="#141622",
            bordercolor="#00FFCC",
            font=dict(color="#ffffff", size=13, family="monospace"),
        ),
    )

    fig.update_xaxes(
        type="category",
        showgrid=False,
        linecolor=color_grid,
        tickfont=dict(color="#8f9cae"),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        type="category",
        showgrid=False,
        linecolor=color_grid,
        tickfont=dict(color="#8f9cae"),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=color_grid,
        zeroline=False,
        linecolor=color_grid,
        tickfont=dict(color="#8f9cae"),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=color_grid,
        zeroline=False,
        linecolor=color_grid,
        tickfont=dict(color="#8f9cae"),
        row=2,
        col=1,
    )

    return fig
