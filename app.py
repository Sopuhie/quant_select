"""
Streamlit 复盘与全功能控制台（暗黑霓虹科技风最终整合版）。
支持一键在界面运行：
  1. 全量/增量本地行情同步（stock_daily_kline，对齐「全 A」上限可配）
  2. 模型重新训练
  3. 每日选股预测
  4. 历史收益回填
  5. 2024 滚动回测
"""
from __future__ import annotations

import io
import locale
import os
import platform
import sqlite3
import subprocess
import sys
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
from src.database import init_db, query_df
from src.kline_chart import draw_candlestick, get_stock_kline_data

DEFAULT_TRAIN_END_DATE = os.environ.get("QUANT_TRAIN_END_DATE", "2024-12-31")

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


def run_command_interactive(args: list[str]) -> tuple[int, str]:
    """执行后台脚本，实时捕获标准输出并在 streamlit 端展示。"""
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
    return returncode, "".join(log_stream)


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


# ================= 6. 选项卡定义 =================
tab_today, tab_hist, tab_backtest, tab_perf, tab_data, tab_settings = st.tabs(
    [
        "🎯 今日推荐",
        "📜 历史与股票K线查询",
        "📈 历史回测",
        "⚡ 模型表现",
        "⚙️ 系统控制台",
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
            value="000300",
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
            fig = draw_candlestick(df_kline, str(q).zfill(6), "自选查询")
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

# ----------------- TAB 3: 📈 历史回测 -----------------
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
                <p style="font-size: 0.85rem; color: #8f9cae;">写入本地表 stock_daily_kline：无历史则拉约 365 自然日；已有则从最近日期次日增量 UPSERT（默认最多约 6000 只，见脚本参数）。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        data_btn = st.button(
            "🚀 运行本地行情同步", key="run_data_update", use_container_width=True
        )
        if data_btn:
            with st.spinner("正在下载日线并写入本地 SQLite..."):
                ret_code, _log = run_command_interactive(
                    [sys.executable, str(ROOT / "scripts" / "update_local_data.py")]
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
                <p style="font-size: 0.85rem; color: #8f9cae;">读取本地流程采集的历史因子面板，使用 LightGBM 重训模型。截止日期默认 2024-12-31，可用环境变量 QUANT_TRAIN_END_DATE 覆盖。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        train_btn = st.button(
            "🚀 开始模型训练", key="run_train", use_container_width=True
        )
        if train_btn:
            with st.spinner("正在重新计算因子并重训模型..."):
                ret_code, _log = run_command_interactive(
                    [
                        sys.executable,
                        str(ROOT / "train_model.py"),
                        "--train-end-date",
                        DEFAULT_TRAIN_END_DATE,
                    ]
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
                <p style="font-size: 0.85rem; color: #8f9cae;">一键计算最新因子的中位数去极值（MAD）与标准化，并跑出今日 Top 推荐。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        predict_btn = st.button(
            "🚀 运行每日预测", key="run_predict", use_container_width=True
        )
        if predict_btn:
            with st.spinner("提取全市场实时因子，进行 LightGBM 测算中..."):
                ret_code, _log = run_command_interactive(
                    [sys.executable, str(ROOT / "run_daily.py")]
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
                <p style="font-size: 0.85rem; color: #8f9cae;">执行 2024 全年样本外滚动回测，并自动输出累计收益与基准超额资产曲线。</p>
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
                    [sys.executable, str(ROOT / "scripts" / "backtest.py")]
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
                [sys.executable, str(ROOT / "update_returns.py")]
            )
            if ret_code == 0:
                st.success("✅ 历史收益率数据回填完成！")
            else:
                st.error("❌ 回填发生异常")

# ----------------- TAB 6: 🔧 环境设置 -----------------
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
