"""
Streamlit 复盘与全功能控制台（暗黑霓虹科技风最终整合版）。
支持一键在界面运行：
  1. 全量/增量本地行情同步（stock_daily_kline，对齐「全 A」上限可配）
  2. 模型重新训练
  3. 每日选股预测
  4. 历史收益回填
  5. 2024 滚动回测
  6. 🎨 视觉图形选股（本地 K 线形态相似度扫描）

审计：`run_command_interactive` 结束后写入 ``system_logs``；「📋 系统运行日志」Tab 可追溯控制台输出。
"""
from __future__ import annotations

import io
import json
import locale
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

from src.config import DATA_DIR, DB_PATH, FEATURE_COLUMNS, MODEL_PATH, TOP_N_SELECTION
from src.config_manager import config_manager
from src.database import init_db, insert_system_log, query_df
from src.pattern_matcher import find_similar_patterns
from src.kline_chart import (
    draw_candlestick,
    draw_realtime_line_chart,
    get_realtime_min_data,
    get_stock_kline_data,
    lookup_stock_display_name,
)

# 若设置 QUANT_TRAIN_END_DATE=YYYY-MM-DD，面板训练将仅使用该日及以前的本地 K 线；不设则使用库内全部日期。
_QUANT_TRAIN_END_DATE_ENV = os.environ.get("QUANT_TRAIN_END_DATE", "").strip()

# ================= 1. 全局页面配置 =================
st.set_page_config(
    page_title="A-Quant Lite 控制台",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_db()

# ================= 2. 定义全局酷炫 Plotly 绘图模板 =================
COLOR_CYBER_TEAL = "#00FFCC"  # 霓虹青色
COLOR_CYBER_ORANGE = "#FF9900"  # 基准橙色线
COLOR_TEXT = "#8f9cae"
COLOR_GRID = "rgba(255, 255, 255, 0.04)"  # 非常淡的网格


def get_cyber_layout(title: str = "图表名称") -> go.Layout:
    return go.Layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT),
        xaxis=dict(
            showgrid=False,
            linecolor=COLOR_GRID,
            tickfont=dict(color="#8f9cae"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            linecolor=COLOR_GRID,
            tickfont=dict(color="#8f9cae"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLOR_TEXT),
        ),
        margin=dict(l=50, r=20, t=60, b=40),
    )


def draw_cyber_area_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    name: str = "策略",
) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=name,
            line=dict(color=COLOR_CYBER_TEAL, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 204, 0.04)",
        )
    )
    return fig


@st.fragment(run_every=15)
def render_intraday_live_charts(top3: list[tuple[str, str]]) -> None:
    """仅重跑本片段以刷新分时图，避免整页 st_autorefresh。"""
    live_cols = st.columns(3)
    for i, col in enumerate(live_cols):
        if i >= len(top3):
            break
        code, name = top3[i]
        with col:
            with st.spinner(f"正在捕获 {name} 分时..."):
                df_live = get_realtime_min_data(code)
                if df_live is not None and not df_live.empty:
                    fig_live = draw_realtime_line_chart(df_live, code, name)
                    if fig_live:
                        st.plotly_chart(
                            fig_live,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                    else:
                        st.info("等候开盘交易数据...")
                else:
                    st.info("☕ 非交易时段或无今日分时数据")


# ================= 3. 注入自定义 CSS 样式 =================
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    .stApp {
        background-color: #0d0e15;
    }

    /* 指标卡片玻璃拟态 */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', Courier, monospace;
        color: #00FFCC !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.4);
        font-size: 1.8rem !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #8f9cae !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 34, 51, 0.3);
        border: 1px solid rgba(0, 255, 204, 0.15);
        border-radius: 10px;
        padding: 12px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* 选项卡美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(30, 34, 51, 0.3);
        padding: 8px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        color: #8f9cae !important;
        background-color: transparent !important;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        color: #00FFCC !important;
        background-color: rgba(0, 255, 204, 0.1) !important;
        border-bottom: 2px solid #00FFCC !important;
        text-shadow: 0 0 5px rgba(0, 255, 204, 0.3);
    }

    /* 按钮微调：突出主按钮的霓虹感 */
    .stButton>button {
        background-color: rgba(30, 34, 51, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #00FFCC;
        color: #00FFCC;
        box-shadow: 0 0 8px rgba(0, 255, 204, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================= 4. 辅助执行函数 (带实时日志捕获) =================
def _decode_subprocess_stdout(chunk: bytes) -> str:
    """Windows 控制台常为 GBK/CP936，子进程若未开 UTF-8 模式会乱码；此处做多编码兜底。"""
    if not chunk:
        return ""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return chunk.decode(enc)
        except UnicodeDecodeError:
            continue
    pref = (locale.getpreferredencoding(False) or "").strip()
    if pref and pref.lower() not in ("utf-8", "utf8"):
        try:
            return chunk.decode(pref)
        except (UnicodeDecodeError, LookupError):
            pass
    for enc in ("gbk", "cp936"):
        try:
            return chunk.decode(enc)
        except UnicodeDecodeError:
            continue
    return chunk.decode("utf-8", errors="replace")


def run_command_interactive(
    args: list[str],
    task_name: str = "后台任务",
) -> tuple[int, str]:
    """执行后台脚本，实时捕获标准输出并在 streamlit 端展示；结束后写入 ``system_logs``。"""
    placeholder = st.empty()
    log_stream: list[str] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # 强制子进程 Python 使用 UTF-8 标准输出（Windows 默认常为 GBK，会导致父进程按 UTF-8 读时乱码）
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.platform == "win32":
        env.setdefault("PYTHONUTF8", "1")

    process = subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )

    assert process.stdout is not None
    while True:
        raw = process.stdout.readline()
        if not raw and process.poll() is not None:
            break
        if raw:
            line = _decode_subprocess_stdout(raw)
            log_stream.append(line)
            placeholder.code("".join(log_stream[-25:]), language="bash")

    returncode = process.wait()
    full_log = "".join(log_stream)
    status = "SUCCESS" if returncode == 0 else "FAILED"
    parameters_str = json.dumps(args, ensure_ascii=False)
    try:
        insert_system_log(
            task_name=task_name,
            status=status,
            parameters=parameters_str,
            log_output=full_log,
        )
    except Exception as exc:
        print(f"写入 system_logs 失败: {exc}")

    return returncode, full_log


# ================= 5. 头部标题栏 =================
st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 25px;">
        <div style="background-color: #00FFCC; width: 6px; height: 35px; border-radius: 3px; margin-right: 15px; box-shadow: 0 0 10px #00FFCC;"></div>
        <h1 style="color: #ffffff; margin: 0; font-family: 'Segoe UI', sans-serif; font-weight: 800; letter-spacing: 1px;">A-QUANT LITE <span style="color: #00FFCC; font-size: 1.2rem; font-weight: 400;">智能选股控制舱</span></h1>
    </div>
    """,
    unsafe_allow_html=True,
)


def _latest_top3() -> pd.DataFrame:
    return query_df(
        """
        SELECT trade_date, rank, stock_code, stock_name, score, close_price,
               next_day_return, hold_5d_return
        FROM daily_selections
        WHERE trade_date = (SELECT MAX(trade_date) FROM daily_selections)
        ORDER BY rank ASC
        """
    )


def _sanitize_latest_selection(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    msgs: list[str] = []
    if df.empty:
        return df, msgs

    out = df[df["rank"].isin(range(1, TOP_N_SELECTION + 1))].copy()
    if len(out) < len(df):
        msgs.append("部分记录的 rank 不在 1–3，已从今日推荐展示中排除。")

    before = len(out)
    out = out.sort_values("score", ascending=False).drop_duplicates(
        subset=["rank"], keep="first"
    )
    out = out.sort_values("rank").reset_index(drop=True)
    if len(out) < before:
        msgs.append("检测到同一 rank 对应多只股票，已保留该 rank 下 score 较高的一条。")

    if len(out) > TOP_N_SELECTION:
        msgs.append(
            f"检测到超过 {TOP_N_SELECTION} 条有效记录，仅展示前 {TOP_N_SELECTION} 条。"
        )
        out = out.head(TOP_N_SELECTION).reset_index(drop=True)

    return out, msgs


def _pattern_range_from_plotly_state(
    state: object,
    allowed_dates: set[str],
) -> tuple[str, str] | None:
    """从 ``st.plotly_chart(..., on_select='rerun')`` 的状态中解析框选日期区间（仅保留当前 K 线中出现的交易日）。"""
    if state is None:
        return None
    try:
        raw = dict(state) if hasattr(state, "keys") else state
        sel = raw.get("selection") or {}
        pts = sel.get("points") or []
    except (TypeError, AttributeError, ValueError):
        return None
    if not pts:
        return None
    xs: list[str] = []
    for p in pts:
        if not isinstance(p, dict):
            continue
        x = p.get("x")
        if x is None:
            continue
        d = str(x).strip()[:10]
        if d in allowed_dates:
            xs.append(d)
    if not xs:
        return None
    xs_u = sorted(set(xs))
    return xs_u[0], xs_u[-1]


# ================= 6. 选项卡定义 =================
(
    tab_today,
    tab_hist,
    tab_match,
    tab_backtest,
    tab_perf,
    tab_data,
    tab_logs,
    tab_settings,
) = st.tabs(
    [
        "🎯 今日推荐",
        "📜 历史与股票K线查询",
        "🎨 视觉图形选股",
        "📈 历史回测",
        "⚡ 模型表现",
        "⚙️ 系统控制台",
        "📋 系统运行日志",
        "🔧 环境设置",
    ]
)

# ----------------- TAB 1: 今日推荐 -----------------
with tab_today:
    t3 = _latest_top3()
    today_df, selection_warnings = _sanitize_latest_selection(t3)
    if t3.empty:
        st.info("暂无选股记录。请先在「系统控制台」中运行数据更新与预测。")
    else:
        latest_date = t3.iloc[0]["trade_date"]
        st.subheader(f"📅 最近交易日 · {latest_date}")

        for w in selection_warnings:
            st.warning(w)

        if "selected_stock" not in st.session_state:
            st.session_state.selected_stock = None

        if not today_df.empty:
            cols = st.columns(len(today_df))
            for i, col in enumerate(cols):
                row = today_df.iloc[i]
                with col:
                    st.markdown(
                        f"""
                <div style="
                            background-color: #141622;
                            border: 2px solid #00FFCC;
                            padding: 22px;
                            border-radius: 12px;
                            margin: 10px 0;
                    text-align: center;
                            box-shadow: 0 4px 20px rgba(0, 255, 204, 0.15);
                        ">
                            <span style="
                                background-color: #00FFCC;
                                color: #0d0e15;
                                font-weight: bold;
                                padding: 3px 12px;
                                border-radius: 20px;
                                font-size: 0.85rem;
                            ">RANK #{int(row["rank"])}</span>
                            <h2 style="color: #ffffff; margin: 15px 0 5px 0; font-family: monospace; font-size: 2rem; letter-spacing: 1px;">{row["stock_code"]}</h2>
                            <h4 style="color: #8f9cae; margin: 0 0 15px 0; font-weight: 500;">{row["stock_name"]}</h4>
                            <div style="border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 12px; display: flex; justify-content: space-around;">
                                <div>
                                    <div style="font-size: 0.75rem; color: #6272a4;">得分</div>
                                    <div style="color: #00FFCC; font-weight: bold; font-family: monospace;">{float(row["score"]):.4f}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.75rem; color: #6272a4;">收盘价</div>
                                    <div style="color: #ffffff; font-weight: bold; font-family: monospace;">{row["close_price"]}</div>
                                </div>
                            </div>
                </div>
                """,
                        unsafe_allow_html=True,
                    )

                    btn_key = f"kline_btn_{row['stock_code']}"
                    if st.button(
                        f"📊 查看K线 - {row['stock_name']}",
                        key=btn_key,
                        use_container_width=True,
                    ):
                        st.session_state.selected_stock = {
                            "code": row["stock_code"],
                            "name": row["stock_name"],
                        }

        if not today_df.empty:
            st.markdown(
                "<div style='margin-top: 30px;'></div>", unsafe_allow_html=True
            )
            st.subheader("⏱️ 推荐股今日实时分时走势对比")
            st.caption(
                "股市开盘期间（9:30–11:30，13:00–15:00），下方三支分时仅在本区域每约 15 秒刷新一次，"
                "不会触发整页重载。"
            )
            top3_pairs = [
                (str(row["stock_code"]), str(row["stock_name"]))
                for _, row in today_df.head(3).iterrows()
            ]
            render_intraday_live_charts(top3_pairs)

        if not t3.empty and st.session_state.selected_stock:
            st.markdown("---")
            col_title, col_close = st.columns([4, 1])
            with col_title:
                st.subheader(
                    f"📈 {st.session_state.selected_stock['code']} "
                    f"{st.session_state.selected_stock['name']} - K线图"
                )
            with col_close:
                if st.button("❌ 关闭", use_container_width=True):
                    st.session_state.selected_stock = None
                    st.rerun()

            with st.spinner("从本地读取K线中..."):
                df_kline = get_stock_kline_data(
                    st.session_state.selected_stock["code"],
                    days=365,
                )

            if df_kline is not None and len(df_kline) > 0:
                fig = draw_candlestick(
                    df_kline,
                    st.session_state.selected_stock["code"],
                    st.session_state.selected_stock["name"],
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    latest = df_kline.iloc[-1]
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("最新价", f"{float(latest['close']):.2f}")
                    with col_b:
                        st.metric(
                            "MA5",
                            f"{float(latest['ma5']):.2f}"
                            if pd.notna(latest.get("ma5"))
                            else "—",
                        )
                    with col_c:
                        st.metric(
                            "MA10",
                            f"{float(latest['ma10']):.2f}"
                            if pd.notna(latest.get("ma10"))
                            else "—",
                        )
                    with col_d:
                        st.metric(
                            "MA20",
                            f"{float(latest['ma20']):.2f}"
                            if pd.notna(latest.get("ma20"))
                            else "—",
                        )
                else:
                    st.error("绘制K线图失败")

# ----------------- TAB 2: 历史复盘与个股极速检索 -----------------
with tab_hist:
    st.subheader("🔍 全市场个股本地 K 线秒开浏览器")

    col_input, col_info = st.columns([1.5, 3.5])
    with col_input:
        search_code = st.text_input(
            "📝 输入任意 6 位股票代码查询K线",
            value="600519",
            max_chars=6,
            key="kline_search_code",
        )
    with col_info:
        st.caption(
            "优先通过本地 SQLite（stock_daily_kline）秒开约 365 日行情；"
            "本地缺失时将自动在线补全并写回缓存。"
        )

    q = str(search_code or "").strip()
    if q:
        with st.spinner("极速加载本地行情 K 线中..."):
            df_kline = get_stock_kline_data(q, days=365)
        if df_kline is not None and len(df_kline) > 0:
            code_z = str(q).zfill(6)
            qname = lookup_stock_display_name(code_z) or "自选查询"
            fig = draw_candlestick(df_kline, code_z, qname)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("绘制K线图失败")
        else:
            st.warning(
                "⚠️ 暂无该股票完整本地数据。请前往「系统控制台」运行「本地行情同步」，"
                "或等待在线补全拉取完成。"
            )

    st.markdown("---")
    st.subheader("📜 历史选股记录复盘")

    history_df = query_df(
        """
        SELECT trade_date, rank, stock_code, stock_name, score, close_price,
               next_day_return, hold_5d_return
        FROM daily_selections
        ORDER BY trade_date DESC, rank ASC
        """
    )
    if not history_df.empty:
        display_df = history_df.copy()
        for col in ["next_day_return", "hold_5d_return"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{float(x):.2%}" if pd.notna(x) else "—"
                )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无历史推荐股票。")

# ----------------- TAB 3: 🎨 视觉图形选股 -----------------
with tab_match:
    st.subheader("🎨 走势形态相似度匹配（以股找股）")
    st.caption(
        "输入样板股票代码后加载 **日线 K 线图**；在图上用 **框选（Box Select）** 选定模板区间。"
        "未框选时默认使用图中 **最近约 60 个交易日**。检索基于本地 ``stock_daily_kline`` 收盘价形态。"
    )

    col_t1, col_t2, col_clr = st.columns([1.4, 1.6, 1])
    with col_t1:
        target_stock = st.text_input(
            "样板股票代码",
            value="688012",
            max_chars=6,
            key="pattern_target_code",
            help="加载该股本地日线并作为形态模板来源",
        )
    with col_t2:
        compare_days_ui = st.number_input(
            "候选行情回溯（日历天）",
            min_value=60,
            max_value=800,
            value=180,
            step=30,
            help="全市场扫描时自模板结束日往前多读多少日历天的数据（越大越慢）",
            key="pattern_compare_days",
        )
    with col_clr:
        st.write("")
        if st.button("清除框选区间", key="pattern_clear_box", use_container_width=True):
            st.session_state.pop("pattern_tpl_range", None)
            st.rerun()

    match_algo = st.radio(
        "相似度算法",
        options=["pearson", "dtw"],
        format_func=lambda x: (
            "皮尔逊相关 + 形状距离 (Pearson)"
            if x == "pearson"
            else "动态时间规整 (DTW)"
        ),
        horizontal=True,
        key="pattern_match_algo",
        help=(
            "Pearson：同一时间轴上的相关性与 RMSE 组合得分。"
            "DTW：纯 NumPy 双行 DP + Sakoe-Chiba 带状加速（固定带宽下约 O(N·W)），"
            "允许形态在时间轴上局部伸缩对齐；得分 100/(1+距离)。"
        ),
    )

    code_raw = str(target_stock or "").strip()
    code_z = code_raw.zfill(6) if code_raw else ""

    prev_pat_code = st.session_state.get("_pattern_watch_code")
    if prev_pat_code != code_z:
        st.session_state["_pattern_watch_code"] = code_z
        st.session_state.pop("pattern_tpl_range", None)

    st.markdown("---")

    if not code_raw:
        st.info("请输入 6 位样板股票代码。")
    else:
        with st.spinner("加载本地日线 K 线…"):
            df_full = get_stock_kline_data(code_z, days=730)

        if df_full is None or df_full.empty or len(df_full) < 5:
            st.warning(
                f"未获取到 **{code_z}** 足够本地日线。请在「系统控制台」运行行情同步后再试。"
            )
        else:
            df_full = df_full.sort_values("date").reset_index(drop=True)
            df_full["date"] = pd.to_datetime(df_full["date"])
            date_str_set = set(df_full["date"].dt.strftime("%Y-%m-%d"))

            plot_state = st.session_state.get("pattern_kline_sel")
            rng_pick = _pattern_range_from_plotly_state(plot_state, date_str_set)
            if rng_pick:
                st.session_state["pattern_tpl_range"] = rng_pick

            pair = st.session_state.get("pattern_tpl_range")
            if pair:
                start_str, end_str = str(pair[0])[:10], str(pair[1])[:10]
                if pd.Timestamp(start_str) > pd.Timestamp(end_str):
                    start_str, end_str = end_str, start_str
            else:
                tail_n = min(60, len(df_full))
                seg_df = df_full.iloc[-tail_n:]
                start_str = seg_df.iloc[0]["date"].strftime("%Y-%m-%d")
                end_str = seg_df.iloc[-1]["date"].strftime("%Y-%m-%d")

            qname = lookup_stock_display_name(code_z) or code_z
            fig_k = draw_candlestick(df_full, code_z, qname)
            if fig_k:
                try:
                    fig_k.add_vrect(
                        x0=start_str,
                        x1=end_str,
                        fillcolor="rgba(0, 255, 204, 0.14)",
                        line=dict(color="rgba(0, 255, 204, 0.5)", width=1),
                        layer="below",
                    )
                except Exception:
                    pass

                st.caption(
                    "👉 点击图表右上角工具栏 **「Box Select」**（方框），在 K 线上拖选一段区间；"
                    "松手后页面会自动刷新并采用框内交易日作为模板起止。"
                )
                st.plotly_chart(
                    fig_k,
                    use_container_width=True,
                    key="pattern_kline_sel",
                    on_select="rerun",
                    selection_mode=("box",),
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                    },
                )

            tpl_df = query_df(
                """
                SELECT date, close FROM stock_daily_kline
                WHERE stock_code = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                (code_z, start_str, end_str),
            )

            params_now = (
                code_z,
                start_str,
                end_str,
                int(compare_days_ui),
                str(match_algo),
            )

            if tpl_df.empty:
                st.warning(
                    f"本地库中 **{code_z}** 在 {start_str}～{end_str} 无成交日线，请同步行情或调整框选区间。"
                )
            else:
                tpl_df = tpl_df.copy()
                tpl_df["close"] = pd.to_numeric(tpl_df["close"], errors="coerce")
                tpl_df = tpl_df.dropna(subset=["close"])
                n_tpl = len(tpl_df)

                src_note = (
                    "来自图上框选"
                    if st.session_state.get("pattern_tpl_range")
                    else "默认最近约 60 个交易日（尚未框选）"
                )
                st.info(
                    f"**当前模板区间**：{start_str} ~ {end_str} · **{n_tpl}** 个交易日 · {src_note}"
                )

                if n_tpl < 5:
                    st.warning("模板区间少于 5 根 K 线，请在图上框选更长区间后再匹配。")
                else:
                    fig_tpl = go.Figure()
                    fig_tpl.add_trace(
                        go.Scatter(
                            x=tpl_df["date"].astype(str),
                            y=tpl_df["close"],
                            mode="lines+markers",
                            name="模板收盘",
                            line=dict(color=COLOR_CYBER_TEAL, width=3),
                            marker=dict(size=4, color=COLOR_CYBER_TEAL),
                        )
                    )
                    fig_tpl.update_layout(
                        get_cyber_layout(f"{code_z} 模板收盘价（{n_tpl} 日）"),
                        height=240,
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_tpl, use_container_width=True)

                    run_match = st.button(
                        "🔍 全市场形态匹配",
                        type="primary",
                        use_container_width=True,
                        key="pattern_run_match",
                    )

                    if run_match:
                        _spin = (
                            "正在按 DTW（双行 DP）扫描全市场，可能较慢…"
                            if match_algo == "dtw"
                            else "正在扫描本地全市场行情，请稍候…"
                        )
                        with st.spinner(_spin):
                            st.session_state["pm_results"] = find_similar_patterns(
                                target_code=code_z,
                                start_date=start_str,
                                end_date=end_str,
                                compare_days=int(compare_days_ui),
                                limit_results=3,
                                algorithm=str(match_algo),
                            )
                            st.session_state["pm_cached_params"] = params_now

                    cached = st.session_state.get("pm_cached_params")
                    match_results = st.session_state.get("pm_results")

                    if cached != params_now:
                        match_results = None

                    if (
                        cached == params_now
                        and match_results is not None
                        and len(match_results) == 0
                    ):
                        st.warning(
                            "未找到相似标的：请确认本地库有足够 K 线，或增大「候选行情回溯」后再试。"
                        )

                    if match_results:
                        st.markdown("### 🏆 相似形态 Top 3")
                        match_cols = st.columns(3)
                        for i, col in enumerate(match_cols):
                            if i >= len(match_results):
                                break
                            res = match_results[i]
                            with col:
                                st.markdown(
                                    f"""
                        <div style="background-color:#141622;border:1.5px solid {COLOR_CYBER_ORANGE};padding:14px;border-radius:8px;text-align:center;margin-bottom:12px;">
                            <span style="background-color:{COLOR_CYBER_ORANGE};color:#0d0e15;font-weight:bold;padding:2px 10px;border-radius:12px;font-size:0.78rem;">MATCH #{i + 1}</span>
                            <h3 style="color:#fff;margin:10px 0 2px 0;font-family:monospace;">{res["stock_code"]}</h3>
                            <h5 style="color:#8f9cae;margin:0 0 8px 0;">{res["stock_name"]}</h5>
                            <div style="font-size:0.78rem;color:#6272a4;">形态契合度（启发式）</div>
                            <div style="color:{COLOR_CYBER_TEAL};font-size:1.45rem;font-weight:bold;font-family:monospace;">{res["similarity"]:.1f}%</div>
                        </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        _algo_note = (
                            "当前排序：**皮尔逊 + 形状距离** 启发式得分。"
                            if match_algo == "pearson"
                            else (
                                "当前排序：**DTW**（NumPy 双行动态规划，默认 Sakoe-Chiba 带状；"
                                "必要时回退完整 DTW）相似度 `100/(1+距离)`。"
                            )
                        )
                        st.caption(
                            "三张折线图分别对比：**样板股**与 **各 Top 匹配股** 在各自窗口内做 Min-Max 归一化后的收盘价曲线；"
                            "横轴为样板模板区间的交易日顺序。"
                            + _algo_note
                        )
                        tpl_label = f"样板 {code_z}"
                        if qname:
                            tpl_label += f" ({qname[:16]}{'…' if len(qname) > 16 else ''})"

                        ref = match_results[0]
                        n_pts = len(ref["target_trajectory"])
                        x_idx = list(range(n_pts))
                        tpl_dates = (
                            pd.to_datetime(tpl_df["date"], errors="coerce")
                            .dt.strftime("%Y-%m-%d")
                            .tolist()
                        )
                        if len(tpl_dates) != n_pts:
                            tpl_dates = [str(i + 1) for i in x_idx]

                        # 高饱和霓虹色 + 略粗线宽，深色背景下更易辨认
                        match_line_styles: list[dict[str, object]] = [
                            {
                                "color": "#FF2D95",
                                "width": 3.5,
                                "shape": "spline",
                            },
                            {
                                "color": "#FFEA00",
                                "width": 3.5,
                                "shape": "spline",
                            },
                            {
                                "color": "#00F5FF",
                                "width": 3.5,
                                "shape": "spline",
                            },
                        ]

                        chart_cols = st.columns(3)
                        for i, res in enumerate(match_results):
                            if i >= len(chart_cols):
                                break
                            sty = match_line_styles[
                                i % len(match_line_styles)
                            ]
                            nm = str(res.get("stock_name") or "")[:12]
                            fig_i = go.Figure()
                            fig_i.add_trace(
                                go.Scatter(
                                    x=x_idx,
                                    y=ref["target_trajectory"],
                                    mode="lines",
                                    name=tpl_label,
                                    line=dict(
                                        color=COLOR_CYBER_TEAL,
                                        width=3,
                                        dash="dash",
                                    ),
                                )
                            )
                            fig_i.add_trace(
                                go.Scatter(
                                    x=x_idx,
                                    y=res["candidate_trajectory"],
                                    mode="lines",
                                    name=f"#{i + 1} {res['stock_code']} {nm}",
                                    line=dict(
                                        color=sty["color"],
                                        width=float(sty["width"]),
                                        shape=str(sty["shape"]),
                                    ),
                                )
                            )
                            fig_i.update_layout(
                                get_cyber_layout(
                                    f"样板 vs 匹配 #{i + 1} · {res['stock_code']}"
                                ),
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="top",
                                    y=-0.08,
                                    xanchor="center",
                                    x=0.5,
                                    font=dict(size=10, color=COLOR_TEXT),
                                    bgcolor="rgba(13,14,21,0.7)",
                                ),
                                height=400,
                                margin=dict(l=50, r=20, t=55, b=100),
                            )
                            fig_i.update_xaxes(
                                ticktext=tpl_dates,
                                tickvals=x_idx,
                                tickangle=-50,
                            )
                            with chart_cols[i]:
                                st.plotly_chart(
                                    fig_i, use_container_width=True
                                )

# ----------------- TAB 4: 📈 历史回测 -----------------
with tab_backtest:
    st.subheader("📈 策略历史回测分析")
    st.caption("基于 LightGBM 历史滚动预测与周期换仓策略（2024 样本外滚动）。")

    backtest_csv_path = DATA_DIR / "backtest_results.csv"

    if not backtest_csv_path.exists():
        st.info("📊 尚未生成历史回测数据。请前往「系统控制台」中运行历史回测。")
    else:
        try:
            df_res = pd.read_csv(backtest_csv_path)
            df_res["strategy_cum"] = (
                df_res["nav"] / df_res["nav"].iloc[0] - 1.0
            ) * 100

            has_bench = (
                "benchmark_close" in df_res.columns
                and df_res["benchmark_close"].notna().any()
            )
            if has_bench:
                valid_bench = df_res["benchmark_close"].dropna()
                if not valid_bench.empty:
                    df_res["bench_cum"] = (
                        df_res["benchmark_close"] / valid_bench.iloc[0] - 1.0
                    ) * 100

            cum_ret = df_res["nav"].iloc[-1] / df_res["nav"].iloc[0] - 1.0
            peak = df_res["nav"].cummax()
            mdd = (df_res["nav"] / peak - 1.0).min()

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("策略累计收益", f"{cum_ret * 100:.2f}%")
            with col_m2:
                st.metric("最大回撤 (MDD)", f"{mdd * 100:.2f}%")
            with col_m3:
                st.metric("回测交易日数", f"{len(df_res)}")

            fig = go.Figure()
            draw_cyber_area_trace(
                fig,
                df_res,
                x_col="trade_date",
                y_col="strategy_cum",
                name="A-Quant 策略",
            )

            if has_bench and "bench_cum" in df_res.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_res["trade_date"],
                        y=df_res["bench_cum"],
                        mode="lines",
                        name="沪深 300 基准",
                        line=dict(color=COLOR_CYBER_ORANGE, width=2, dash="dash"),
                    )
                )

            fig.update_layout(
                get_cyber_layout("策略 vs 沪深300 累计收益率 (%)"), height=550
            )
            st.plotly_chart(fig, use_container_width=True)

            buf_res = io.StringIO()
            df_res.to_csv(buf_res, index=False)
            st.download_button(
                "💾 导出回测 CSV",
                data=buf_res.getvalue().encode("utf-8-sig"),
                file_name="backtest_results_export.csv",
                mime="text/csv",
            )

        except Exception as err:
            st.error(f"解析回测数据出错: {err}")

# ----------------- TAB 4: 模型表现 -----------------
with tab_perf:
    st.subheader("⚡ 模型特征贡献度分析")

    conn = sqlite3.connect(str(DB_PATH))
    model_df = pd.read_sql_query(
        "SELECT version, is_active FROM model_versions ORDER BY id DESC LIMIT 5",
        conn,
    )
    conn.close()

    st.subheader("🎯 因子贡献度排行 (Feature Importance)")

    if not MODEL_PATH.exists():
        st.warning("⚠️ 找不到本地模型文件。请在「系统控制台」中启动模型训练。")
    else:
        try:
            model = joblib.load(MODEL_PATH)

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "feature_importance"):
                importances = model.feature_importance()
            else:
                importances = None

            if importances is not None:
                imp_arr = (
                    importances
                    if hasattr(importances, "__len__")
                    else list(importances)
                )
                n = min(len(imp_arr), len(FEATURE_COLUMNS))
                if n == 0:
                    st.info("特征重要性长度为 0。")
                else:
                    if len(imp_arr) != len(FEATURE_COLUMNS):
                        st.warning(
                            "模型特征数与配置 FEATURE_COLUMNS 不一致，已按较短一侧截取展示。"
                        )
                    feat_imp_df = pd.DataFrame(
                        {
                            "Feature": FEATURE_COLUMNS[:n],
                            "Importance": list(imp_arr)[:n],
                        }
                    )
                    feat_imp_df = feat_imp_df.sort_values(
                        "Importance", ascending=True
                    )

                    fig_imp = go.Figure()
                    fig_imp.add_trace(
                        go.Bar(
                            y=feat_imp_df["Feature"],
                            x=feat_imp_df["Importance"],
                            orientation="h",
                            marker=dict(
                                color=feat_imp_df["Importance"],
                                colorscale=[
                                    [0, "rgba(0, 255, 204, 0.2)"],
                                    [1, "rgba(0, 255, 204, 1.0)"],
                                ],
                                line=dict(color=COLOR_CYBER_TEAL, width=1.5),
                            ),
                            name="因子分裂次数",
                        )
                    )

                    fig_imp.update_layout(
                        get_cyber_layout(
                            "模型决策树分裂频次排行 (分值越高代表因子越核心)"
                        ),
                        xaxis=dict(showgrid=True, gridcolor=COLOR_GRID),
                        height=500,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("无法解析特征重要性指标。")
        except Exception as e:
            st.error(f"解析出错: {e}")

    st.markdown("---")
    st.subheader("🧩 模型版本历史")
    st.dataframe(model_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 全市场模型得分排名")
    st.caption(
        "数据来自每日预测写入的「全市场打分」表；可选交易日浏览任意股票的模型分与名次，便于对比非 Top 推荐标的。"
    )

    pred_dates_df = query_df(
        """
        SELECT DISTINCT trade_date FROM daily_predictions
        ORDER BY trade_date DESC
        LIMIT 366
        """
    )
    if pred_dates_df.empty:
        st.info("暂无全市场打分记录。请先在「系统控制台」运行「每日预测」。")
    else:
        date_list = pred_dates_df["trade_date"].astype(str).tolist()
        rank_sel_date = st.selectbox(
            "排名所属交易日",
            options=date_list,
            index=0,
            key="perf_ranking_trade_date",
        )
        rank_full_df = query_df(
            """
            SELECT rank_in_market, stock_code, stock_name, score
            FROM daily_predictions
            WHERE trade_date = ?
            ORDER BY rank_in_market ASC
            """,
            (rank_sel_date,),
        )
        if rank_full_df.empty:
            st.warning("该交易日库中无预测明细。")
        else:
            n_all = len(rank_full_df)
            scope_labels = ["前 200 名", "前 500 名", "前 1000 名", "全部"]
            scope_map = {scope_labels[0]: 200, scope_labels[1]: 500, scope_labels[2]: 1000}
            default_label = scope_labels[1] if n_all >= 500 else (
                scope_labels[2] if n_all > 200 else scope_labels[3]
            )
            scope_pick = st.selectbox(
                "展示范围",
                options=scope_labels,
                index=scope_labels.index(default_label)
                if default_label in scope_labels
                else len(scope_labels) - 1,
                key="perf_ranking_scope",
            )
            cap = n_all if scope_pick == "全部" else min(scope_map[scope_pick], n_all)
            show_df = rank_full_df.head(cap).copy()
            show_df = show_df.rename(
                columns={
                    "rank_in_market": "全市场名次",
                    "stock_code": "代码",
                    "stock_name": "名称",
                    "score": "模型得分",
                }
            )
            show_df["模型得分"] = show_df["模型得分"].map(lambda x: f"{float(x):.4f}")
            st.dataframe(
                show_df,
                use_container_width=True,
                hide_index=True,
                height=min(520, 28 * (cap + 1)),
            )
            st.caption(
                f"当日共 **{n_all}** 只股票参与排名；当前表格展示 **{cap}** 行（按名次升序）。"
            )

# ----------------- TAB 5: ⚙️ 系统控制台 -----------------
with tab_data:
    st.subheader("⚙️ 量化核心任务总控制台")
    st.caption("无需打开终端，在这里一键调度并监视所有后台算法与数据任务。")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown(
            """
            <div style="background-color: rgba(30, 34, 51, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 18px; border-radius: 8px;">
                <h4 style="color: #00FFCC; margin-top:0;">📊 任务 A：全量/增量行情同步</h4>
                <p style="font-size: 0.85rem; color: #8f9cae;">写入本地表 stock_daily_kline：无历史则拉约 365 自然日；已有则从最近日期次日增量 UPSERT。默认最多约 6000 只；勾选下方「全 A」则不限数量（收盘后补当日日线更完整，但更慢）。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sync_full_a_market = st.checkbox(
            "同步全 A（不限 6000 只，收盘后拉齐当日日线）",
            value=False,
            key="sync_full_market_kline",
            help="等价于命令行 python scripts/update_local_data.py --all-stocks",
        )
        data_btn = st.button(
            "🚀 运行本地行情同步", key="run_data_update", use_container_width=True
        )
        if data_btn:
            with st.spinner("正在下载日线并写入本地 SQLite..."):
                data_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "update_local_data.py"),
                ]
                if sync_full_a_market:
                    data_cmd.append("--all-stocks")
                ret_code, _log = run_command_interactive(
                    data_cmd,
                    task_name="本地行情同步",
                )
                if ret_code == 0:
                    st.success("✅ 本地行情同步完成！")
                else:
                    st.error("❌ 同步异常，请查看上方实时日志")

    with row1_col2:
        st.markdown(
            """
            <div style="background-color: rgba(30, 34, 51, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 18px; border-radius: 8px;">
                <h4 style="color: #00FFCC; margin-top:0;">🎯 任务 B：重新训练选股模型</h4>
                <p style="font-size: 0.85rem; color: #8f9cae;">读取本地 SQLite（stock_daily_kline）计算因子并重训 LightGBM LambdaRank（按交易日分组）。可选命令行 <code>--tune</code> 进行 Optuna 调参；默认使用库内全部日期；截止日可设环境变量 QUANT_TRAIN_END_DATE。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        train_btn = st.button(
            "🚀 开始模型训练", key="run_train", use_container_width=True
        )
        if train_btn:
            with st.spinner("正在重新计算因子并重训模型..."):
                train_cmd = [sys.executable, str(ROOT / "train_model.py")]
                if _QUANT_TRAIN_END_DATE_ENV:
                    train_cmd.extend(
                        ["--train-end-date", _QUANT_TRAIN_END_DATE_ENV[:10]]
                    )
                ret_code, _log = run_command_interactive(
                    train_cmd, task_name="模型训练"
                )
                if ret_code == 0:
                    st.success("✅ 模型重训完成，最新模型权重已保存！")
                else:
                    st.error("❌ 训练执行异常，请查看日志")

    with row2_col1:
        st.markdown(
            """
            <div style="background-color: rgba(30, 34, 51, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 18px; border-radius: 8px;">
                <h4 style="color: #00FFCC; margin-top:0;">📡 任务 C：执行每日智能选股</h4>
                <p style="font-size: 0.85rem; color: #8f9cae;">K 线与因子计算均基于本地 stock_daily_kline；截面清洗与 LightGBM 打分后产出 Top 推荐。股票池默认取库内有足够历史的代码；需要成分池时可命令行加 --online-pool。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin-top: 10px;'></div>", unsafe_allow_html=True
        )
        col_cb1, col_cb2 = st.columns(2)
        with col_cb1:
            include_300 = st.checkbox(
                "🟢 包含创业板 (300 / 301)",
                value=True,
                help="未勾选时，选股池将排除代码以 300、301 开头的股票",
            )
        with col_cb2:
            include_688 = st.checkbox(
                "🔵 包含科创板 (688)",
                value=True,
                help="未勾选时，选股池将排除代码以 688 开头的股票",
            )
        predict_btn = st.button(
            "🚀 运行每日预测", key="run_predict", use_container_width=True
        )
        if predict_btn:
            with st.spinner("提取全市场实时因子，进行 LightGBM 测算中..."):
                cmd = [sys.executable, str(ROOT / "run_daily.py")]
                if include_300:
                    cmd.append("--include-300")
                if include_688:
                    cmd.append("--include-688")
                ret_code, _log = run_command_interactive(
                    cmd, task_name="每日智能选股"
                )
                if ret_code == 0:
                    st.success(
                        "✅ 今日推荐选股计算完成！请前往「今日推荐」卡片查看。"
                    )
                else:
                    st.error("❌ 预测执行异常，请查阅控制台")

    with row2_col2:
        st.markdown(
            """
            <div style="background-color: rgba(30, 34, 51, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 18px; border-radius: 8px;">
                <h4 style="color: #00FFCC; margin-top:0;">📈 任务 D：运行策略历史滚动回测</h4>
                <p style="font-size: 0.85rem; color: #8f9cae;">基于本地 stock_daily_kline；不传日期时默认回测区间为库内最早日至最晚日。基准 000300 优先读库。不足可加 --online-fallback。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        backtest_btn = st.button(
            "🚀 启动历史滚动回测",
            key="run_backtest_tab",
            use_container_width=True,
        )
        if backtest_btn:
            with st.spinner("滚动回测执行中，请耐心等待..."):
                ret_code, _log = run_command_interactive(
                    [sys.executable, str(ROOT / "scripts" / "backtest.py")],
                    task_name="历史滚动回测",
                )
                if ret_code == 0:
                    st.success(
                        f"✅ 回测完成！已生成 {DATA_DIR / 'backtest_results.csv'}"
                    )
                else:
                    st.error("❌ 回测异常，请查阅上方调试日志")

    st.markdown("---")
    st.subheader("🔄 任务 E：历史选股收益回填")
    st.caption("对数据库中的历史推荐股票进行持股收益率自动追踪和数据补全。")

    col_ret1, col_ret2 = st.columns([3, 1])
    with col_ret1:
        st.info(
            "回填工具会自动追踪历史选股后的 +1日收益率 和 +5日持有期累计收益率，从而精确评估算法胜率。"
        )
    with col_ret2:
        return_btn = st.button(
            "🚀 开始收益率回填",
            key="run_return_update",
            use_container_width=True,
        )
    if return_btn:
        with st.spinner("正在调取接口，更新并回填历史选股收益率数据..."):
            ret_code, _log = run_command_interactive(
                [sys.executable, str(ROOT / "update_returns.py")],
                task_name="历史收益回填",
            )
            if ret_code == 0:
                st.success("✅ 历史收益率数据回填完成！")
            else:
                st.error("❌ 回填发生异常")

# ----------------- TAB 6: 📋 系统运行日志 -----------------
with tab_logs:
    st.subheader("📋 系统运行日志")
    st.caption(
        "「系统控制台」中通过子进程执行的任务结束后，会将退出码与完整控制台输出写入 SQLite（``system_logs``）。"
    )

    logs_df = query_df(
        """
        SELECT id, task_name, status, run_time, parameters
        FROM system_logs
        ORDER BY id DESC
        LIMIT 200
        """
    )

    if logs_df.empty:
        st.info("暂无运行日志。请在「系统控制台」执行一次任务后刷新本页。")
    else:
        view_df = logs_df.copy()
        view_df["状态"] = view_df["status"].map(
            lambda x: "🟢 SUCCESS" if str(x).upper() == "SUCCESS" else "🔴 FAILED"
        )
        view_df = view_df.drop(columns=["status"]).rename(
            columns={
                "id": "ID",
                "task_name": "任务",
                "run_time": "执行时间",
                "parameters": "命令参数(JSON)",
            }
        )
        st.markdown("##### 最近记录")
        st.dataframe(view_df, use_container_width=True, hide_index=True, height=320)

        log_ids = logs_df["id"].astype(int).tolist()

        def _fmt_log_choice(lid: int) -> str:
            row = logs_df.loc[logs_df["id"] == lid].iloc[0]
            st_label = "SUCCESS" if str(row["status"]).upper() == "SUCCESS" else "FAILED"
            return f"#{lid} | {row['run_time']} | {row['task_name']} ({st_label})"

        st.markdown("---")
        st.markdown("##### 完整控制台输出")
        pick_id = st.selectbox(
            "选择一条记录查看 stdout/stderr 合并日志",
            options=log_ids,
            format_func=_fmt_log_choice,
            key="system_log_detail_select",
        )
        detail_df = query_df(
            "SELECT log_output FROM system_logs WHERE id = ?",
            (int(pick_id),),
        )
        payload = ""
        if not detail_df.empty and detail_df["log_output"].iloc[0] is not None:
            payload = str(detail_df["log_output"].iloc[0])
        if payload.strip():
            st.code(payload, language="bash")
        else:
            st.info("该条记录无日志正文（可能子进程无输出）。")

# ----------------- TAB 7: 🔧 环境设置 -----------------
with tab_settings:
    st.subheader("⚙️ 宿主运行环境与系统配置")

    with st.expander("🔔 钉钉报警推送设置"):
        ding = config_manager.get_dingtalk_config()
        enabled = st.checkbox(
            "启用通知", value=bool(ding.get("enabled", False))
        )
        webhook_url = st.text_input(
            "Webhook URL（完整 webhook 地址）",
            value=str(ding.get("webhook_url", "") or ""),
        )
        secret = st.text_input(
            "安全密钥 (Secret)",
            value=str(ding.get("secret", "") or ""),
            type="password",
        )
        send_time = st.text_input(
            "推送时间过滤",
            value=str(ding.get("send_time", "16:00") or "16:00"),
        )
        if st.button("💾 保存通知设置"):
            ok = config_manager.set_dingtalk_config(
                enabled=enabled,
                webhook_url=webhook_url,
                secret=secret,
                send_time=send_time or "16:00",
            )
            if ok:
                st.success("配置已保存到 config.json！")
            else:
                st.error("保存失败，请检查磁盘权限或路径。")

    st.markdown("---")
    st.subheader("💻 宿主运行环境监控")

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("操作系统", platform.system())
    col_s2.metric("Python 版本", platform.python_version())

    db_size = 0.0
    if DB_PATH.exists():
        db_size = DB_PATH.stat().st_size / (1024 * 1024)
    col_s3.metric("SQLite 数据库大小", f"{db_size:.2f} MB")
