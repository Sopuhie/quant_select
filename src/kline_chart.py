"""
K线图模块
用于获取股票历史数据并绘制K线图
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

    Args:
        stock_code: 股票代码，如 '600219' 或 '000983'
        days: 获取多少天的数据

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, ma5, ma10, ma20
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
    绘制K线图

    Args:
        df: 包含 date, open, high, low, close, volume 的 DataFrame（可含 ma5/ma10/ma20）
        stock_code: 股票代码
        stock_name: 股票名称

    Returns:
        plotly Figure 对象
    """
    if df is None or len(df) == 0:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    if "ma5" not in df.columns:
        df = _append_ma(df)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )

    colors = ["red" if c >= o else "green" for c, o in zip(df["close"], df["open"])]

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="成交量",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma5"],
            name="MA5",
            line=dict(color="orange", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma10"],
            name="MA10",
            line=dict(color="blue", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma20"],
            name="MA20",
            line=dict(color="purple", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title=f"{stock_code} {stock_name} - K线图",
        xaxis_title="日期",
        yaxis_title="价格",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

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
