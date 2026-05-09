"""
K线图模块
用于获取股票历史数据并绘制无缝连续的K线图
"""
from __future__ import annotations

from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

try:
    import baostock as bs

    _HAS_BAOSTOCK = True
except ImportError:
    bs = None  # type: ignore[assignment,misc]
    _HAS_BAOSTOCK = False
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _append_ma(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True)
    out["ma5"] = out["close"].rolling(window=5).mean()
    out["ma10"] = out["close"].rolling(window=10).mean()
    out["ma20"] = out["close"].rolling(window=20).mean()
    return out


def get_stock_kline_data(stock_code, days=60):
    """
    获取股票K线数据
    """
    stock_code = str(stock_code).zfill(6)

    if stock_code.startswith("6"):
        sec_code = f"sh.{stock_code}"
    else:
        sec_code = f"sz.{stock_code}"

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    if _HAS_BAOSTOCK and bs is not None:
        try:
            bs.login()
            rs = bs.query_history_k_data_plus(
                sec_code,
                "date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )

            data_list = []
            while (rs.error_code == "0") and rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(
                    data_list, columns=["date", "open", "high", "low", "close", "volume"]
                )
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["date"] = pd.to_datetime(df["date"])
                df = df.dropna(subset=["open", "high", "low", "close"])
                return _append_ma(df)
        except Exception as e:
            print(f"Baostock 获取失败: {e}")
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    try:
        symbol = stock_code
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if df is not None and len(df) > 0:
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                }
            )
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df = df.dropna(subset=["open", "high", "low", "close"])
            return _append_ma(df)
    except Exception as e:
        print(f"AkShare 获取失败: {e}")

    return None


def draw_candlestick(df, stock_code, stock_name):
    """
    绘制无留白连续K线图

    Args:
        df: 包含 date, open, high, low, close, volume 的 DataFrame
        stock_code: 股票代码
        stock_name: 股票名称
    """
    if df is None or len(df) == 0:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    if "ma5" not in df.columns:
        df = _append_ma(df)

    # 格式化日期显示为 YYYY-MM-DD 字符串，作为 category 类型的坐标，实现无缝连贯
    date_strs = df["date"].dt.strftime("%Y-%m-%d").tolist()

    # 创建双子图，上方K线占 75%，下方成交量占 25%
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # 1. 绘制 K 线部分 (红涨绿跌，契合您的要求)
    # 阳线颜色（Red） 阴线颜色（Green）
    color_up = "#FF3333"
    color_down = "#00CC66"

    fig.add_trace(
        go.Candlestick(
            x=date_strs,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color=color_up,
            increasing_fillcolor=color_up,
            decreasing_line_color=color_down,
            decreasing_fillcolor=color_down,
        ),
        row=1,
        col=1,
    )

    # 2. 绘制成交量柱状图 (颜色与 K 线完全一致)
    volume_colors = [color_up if c >= o else color_down for c, o in zip(df["close"], df["open"])]

    fig.add_trace(
        go.Bar(
            x=date_strs,
            y=df["volume"],
            name="成交量",
            marker_color=volume_colors,
            marker_line_width=0,
        ),
        row=2,
        col=1,
    )

    # 3. 绘制均线 (使用荧光配色凸显暗黑风格)
    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma5"],
            name="MA5",
            line=dict(color="#00FFCC", width=1.5),  # 荧光青
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma10"],
            name="MA10",
            line=dict(color="#FF00FF", width=1.5),  # 荧光紫
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["ma20"],
            name="MA20",
            line=dict(color="#FFFF00", width=1.5),  # 荧光黄
        ),
        row=1,
        col=1,
    )

    # ================= 🚀 极简暗黑 Layout 升级（去除繁琐网格） =================
    COLOR_GRID = "rgba(255, 255, 255, 0.04)"  # 极淡的边框线

    fig.update_layout(
        title=f"{stock_code} {stock_name} (无间断连续K线)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
    )

    # 核心点：强制 X 轴为 category 类型。这会使 Plotly 忽略周末的时间差，让蜡烛图完美连贯！
    fig.update_xaxes(
        type="category",
        showgrid=False,  # 关掉网格线，让画面简洁干净
        linecolor=COLOR_GRID,
        tickfont=dict(color="#8f9cae"),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        type="category",
        showgrid=False,
        linecolor=COLOR_GRID,
        tickfont=dict(color="#8f9cae"),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,  # y轴仅保留淡淡的水平参考线
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont=dict(color="#8f9cae"),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont=dict(color="#8f9cae"),
        row=2,
        col=1,
    )

    return fig


def get_current_price(stock_code):
    """获取当前实时价格"""
    stock_code = str(stock_code).zfill(6)
    if stock_code.startswith("6"):
        sec_code = f"sh.{stock_code}"
    else:
        sec_code = f"sz.{stock_code}"

    if not _HAS_BAOSTOCK or bs is None:
        return None

    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            sec_code,
            "date,close",
            start_date=datetime.now().strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
        if rs and rs.next():
            data = rs.get_row_data()
            if data and len(data) > 1:
                return float(data[1])
    except Exception as e:
        print(f"获取当前价格失败: {e}")
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    return None
