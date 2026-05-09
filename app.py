"""
Streamlit 复盘界面（暗黑霓虹科技风整合版）。

启动（在 quant_select 目录下）:
  streamlit run app.py
"""
from __future__ import annotations

import io
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
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import DATA_DIR, DB_PATH, FEATURE_COLUMNS, MODEL_PATH, TOP_N_SELECTION
from src.config_manager import config_manager
from src.database import init_db, query_df
from src.kline_chart import draw_candlestick, get_stock_kline_data
from src.predictor import is_st_stock_row

BACKTEST_CSV_PATH = DATA_DIR / "backtest_results.csv"

# ================= 1. 全局页面配置 =================
st.set_page_config(
    page_title="A-Quant Lite 控制台",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_db()

# ================= 2. 注入自定义 Cyberpunk CSS 样式 =================
st.markdown(
    """
    <style>
    /* 隐藏 Streamlit 顶部红线与页脚 */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* 全局暗黑背景微调 */
    .stApp {
        background-color: #0d0e15;
    }

    /* 针对 st.metric 指标卡片进行玻璃拟态美化 */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', Courier, monospace;
        color: #00FFCC !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
        font-size: 1.8rem !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #8f9cae !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 34, 51, 0.4);
        border: 1px solid rgba(0, 255, 204, 0.15);
        border-radius: 10px;
        padding: 12px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 255, 204, 0.6);
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.2);
    }

    /* 自定义 Tab 选项卡样式 */
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= 3. 头部标题栏 =================
st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 25px;">
        <div style="background-color: #00FFCC; width: 6px; height: 35px; border-radius: 3px; margin-right: 15px; box-shadow: 0 0 10px #00FFCC;"></div>
        <h1 style="color: #ffffff; margin: 0; font-family: 'Segoe UI', sans-serif; font-weight: 800; letter-spacing: 1px;">A-QUANT LITE <span style="color: #00FFCC; font-size: 1.2rem; font-weight: 400;">智能选股系统</span></h1>
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
        msgs.append(f"检测到超过 {TOP_N_SELECTION} 条有效记录，仅展示前 {TOP_N_SELECTION} 条。")
        out = out.head(TOP_N_SELECTION).reset_index(drop=True)

    return out, msgs


# ================= 4. 核心功能选项卡定义 =================
tab_today, tab_hist, tab_backtest, tab_perf, tab_data, tab_pred, tab_settings = st.tabs(
    ["今日推荐", "历史复盘", "📈 历史回测", "模型表现", "数据管理", "全市场预测", "⚙️ 系统设置"]
)

# ----------------- TAB 1: 今日推荐 -----------------
with tab_today:
    t3 = _latest_top3()
    today_df, selection_warnings = _sanitize_latest_selection(t3)
    if t3.empty:
        st.info("暂无选股记录。请先完成训练并运行 run_daily.py。")
    else:
        latest_date = t3.iloc[0]["trade_date"]
        st.subheader(f"📅 最近交易日 · {latest_date}")

        for w in selection_warnings:
            st.warning(w)

        if len(today_df) != TOP_N_SELECTION:
            st.info(
                f"{latest_date} 暂无完整的 Top{TOP_N_SELECTION} 选股展示数据"
                f"（当前有效记录 {len(today_df)} 条）。"
                f"若数据库存在重复写入，可运行：`python scripts/clean_duplicates.py` 后刷新页面。"
            )

        if "selected_stock" not in st.session_state:
            st.session_state.selected_stock = None

        if not today_df.empty:
            cols = st.columns(len(today_df))
            for i, col in enumerate(cols):
                row = today_df.iloc[i]
                with col:
                    # 重新美化为暗黑青边霓虹卡片
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
                            transition: transform 0.3s ease;
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

                    if pd.notna(row.get("next_day_return")):
                        h5 = row.get("hold_5d_return")
                        if pd.notna(h5):
                            st.caption(
                                f"📈 已回填：次日 {float(row['next_day_return']):.2%} · "
                                f"5日 {float(h5):.2%}"
                            )
                        else:
                            st.caption(f"📈 已回填：次日 {float(row['next_day_return']):.2%}")

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

            with st.spinner("加载K线数据中..."):
                df_kline = get_stock_kline_data(
                    st.session_state.selected_stock["code"],
                    days=60,
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
            else:
                st.error("无法获取K线数据，请检查网络或数据源")

# ----------------- TAB 2: 历史复盘 -----------------
with tab_hist:
    history_df = query_df(
        """
        SELECT trade_date, rank, stock_code, stock_name, score, close_price,
               next_day_return, hold_5d_return, created_at
        FROM daily_selections
        ORDER BY trade_date DESC, rank ASC
        """
    )
    if history_df.empty:
        st.info("暂无选股记录。")
    else:
        st.subheader("📜 历史选股记录")

        st.subheader("🔍 查看历史股票K线")
        history_df = history_df.copy()
        history_df["display"] = (
            history_df["trade_date"].astype(str)
            + " - "
            + history_df["stock_code"].astype(str)
            + " "
            + history_df["stock_name"].astype(str)
        )
        options = history_df["display"].tolist()
        selected = st.selectbox(
            "选择股票查看K线图",
            options=options,
            key="history_select",
        )

        if selected:
            sel_row = history_df.loc[history_df["display"] == selected].iloc[0]
            selected_code = sel_row["stock_code"]
            selected_name = sel_row["stock_name"]

            with st.spinner(f"加载 {selected_code} {selected_name} K线数据..."):
                df_kline = get_stock_kline_data(selected_code, days=90)
            if df_kline is not None and len(df_kline) > 0:
                fig = draw_candlestick(df_kline, selected_code, selected_name)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("绘制K线图失败")
            else:
                st.error("无法获取K线数据，请检查网络或数据源")

        st.markdown("---")
        st.subheader("📋 历史记录表")

        display_df = history_df.drop(columns=["display"], errors="ignore").copy()
        for col in ["next_day_return", "hold_5d_return"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{float(x):.2%}" if pd.notna(x) else "—"
                )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("📊 历史统计")
        valid_returns = history_df[history_df["next_day_return"].notna()]
        if len(valid_returns) > 0:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                win_rate = (valid_returns["next_day_return"] > 0).mean()
                st.metric("次日胜率", f"{win_rate:.1%}")
            with col_b:
                avg_return = valid_returns["next_day_return"].mean()
                st.metric("平均次日收益", f"{avg_return:.2%}")
            with col_c:
                st.metric("样本数量", len(valid_returns))
        else:
            st.info("暂无收益率数据，请先运行 scripts/update_returns.py")

# ----------------- TAB 3: 📈 历史回测 -----------------
with tab_backtest:
    st.subheader("📈 策略历史回测分析")
    st.caption(
        "基于 LightGBM 历史滚动预测与周期换仓策略（2024 样本外滚动，默认持有 5 个交易日）。"
    )

    if not BACKTEST_CSV_PATH.exists():
        st.info(
            "📊 尚未生成历史回测数据。你可以点击下方按钮运行 2024 全年滚动回测（程序会优先使用本地 SQLite 历史行情，若数据不足则会在线补充）。"
        )
        col_run, _ = st.columns([2, 3])
        with col_run:
            if st.button(
                "🚀 开始历史滚动回测 (LGB + 周期换仓)",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner(
                    "回测执行中，正在滚动计算因子并打分选股，请耐心等待..."
                ):
                    try:
                        res = subprocess.run(
                            [
                                sys.executable,
                                str(ROOT / "scripts" / "backtest.py"),
                                "--start-date",
                                "2024-01-01",
                                "--end-date",
                                "2024-12-31",
                            ],
                            cwd=str(ROOT),
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            check=False,
                        )
                        if res.returncode == 0:
                            st.success(
                                "✅ 回测执行成功！已成功生成 data/backtest_results.csv"
                            )
                            st.rerun()
                        else:
                            st.error("❌ 回测执行失败，错误日志如下：")
                            log = (res.stderr or "") + "\n" + (res.stdout or "")
                            st.code(log.strip() or "(无输出)")
                    except Exception as ex:
                        st.error(f"❌ 执行异常: {ex}")
    else:
        try:
            df_res = pd.read_csv(BACKTEST_CSV_PATH)
            df_res["strategy_cum"] = (df_res["nav"] / df_res["nav"].iloc[0] - 1.0) * 100

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

            total_days = len(df_res)
            cum_ret = df_res["nav"].iloc[-1] / df_res["nav"].iloc[0] - 1.0

            ann_factor = 242 / max(total_days - 1, 1)
            ann_ret = (df_res["nav"].iloc[-1] / df_res["nav"].iloc[0]) ** ann_factor - 1.0

            peak = df_res["nav"].cummax()
            dd = df_res["nav"] / peak - 1.0
            mdd = dd.min()

            daily_returns = df_res["nav"].pct_change().fillna(0.0)
            rf_d = (1.0 + 0.03) ** (1.0 / 242) - 1.0
            excess = daily_returns - rf_d
            std_d = daily_returns.std(ddof=1)
            sharpe = (
                (excess.mean() / std_d * (242**0.5)) if std_d > 1e-12 else 0.0
            )

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("策略累计收益", f"{cum_ret * 100:.2f}%")
            col_m2.metric("年化收益率", f"{ann_ret * 100:.2f}%")
            col_m3.metric("最大回撤 (MDD)", f"{mdd * 100:.2f}%")
            col_m4.metric("夏普比率 (Sharpe)", f"{sharpe:.3f}")

            if has_bench and "bench_cum" in df_res.columns:
                bench_cum_ret = df_res["bench_cum"].iloc[-1]
                alpha_ret = df_res["strategy_cum"].iloc[-1] - bench_cum_ret
                col_m5.metric("超额收益 (vs基准)", f"{alpha_ret:.2f}%")
            else:
                col_m5.metric("超额收益", "N/A")

            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_res["trade_date"],
                    y=df_res["strategy_cum"],
                    mode="lines",
                    name="A-Quant Lite 策略",
                    line=dict(color="#00FFCC", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(0, 255, 204, 0.04)",
                )
            )

            if has_bench and "bench_cum" in df_res.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_res["trade_date"],
                        y=df_res["bench_cum"],
                        mode="lines",
                        name="沪深 300 基准",
                        line=dict(color="#FF9900", width=2, dash="dash"),
                    )
                )

            fig.update_layout(
                title="2024 滚动样本外策略 vs 沪深300 累计收益率 (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", suffix="%"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            c_btn1, c_btn2, _ = st.columns([1.5, 1.5, 7])
            with c_btn1:
                if st.button("🔄 重新执行回测", use_container_width=True):
                    BACKTEST_CSV_PATH.unlink(missing_ok=True)
                    st.rerun()
            with c_btn2:
                buf_res = io.StringIO()
                df_res.to_csv(buf_res, index=False)
                st.download_button(
                    "💾 导出回测 CSV",
                    data=buf_res.getvalue().encode("utf-8-sig"),
                    file_name="backtest_results_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.expander("📋 查看每日调仓资产账单明细"):
                display_cols = [
                    "trade_date",
                    "nav",
                    "cash",
                    "hold_mv",
                    "n_positions",
                    "daily_return",
                ]
                existing_cols = [c for c in display_cols if c in df_res.columns]
                detail_df = df_res[existing_cols].copy()
                if "daily_return" in detail_df.columns:
                    detail_df["daily_return"] = detail_df["daily_return"].apply(
                        lambda x: f"{float(x):.2%}" if pd.notna(x) else "—"
                    )
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

        except Exception as err:
            st.error(f"解析回测数据出错: {err}")
            if st.button("🧹 清理损坏的回测缓存"):
                BACKTEST_CSV_PATH.unlink(missing_ok=True)
                st.rerun()

# ----------------- TAB 4: 模型表现 -----------------
with tab_perf:
    st.subheader("📊 模型表现与特征贡献度分析")
    st.caption(
        "分析当前激活模型的特征重要性（Feature Importance）以及回填收益率后的多维度胜率统计。"
    )

    conn = sqlite3.connect(str(DB_PATH))
    perf_df = pd.read_sql_query(
        """
        SELECT trade_date, rank, stock_code, stock_name,
               next_day_return, hold_5d_return
        FROM daily_selections
        WHERE next_day_return IS NOT NULL
        ORDER BY trade_date DESC
        """,
        conn,
    )
    model_df = pd.read_sql_query(
        """
        SELECT version, train_end_date, is_active, features, metrics, created_at
        FROM model_versions
        ORDER BY created_at DESC
        LIMIT 5
        """,
        conn,
    )
    conn.close()

    st.subheader("🎯 因子贡献度排行 (Feature Importance)")

    if not MODEL_PATH.exists():
        st.warning("⚠️ 找不到本地模型文件，请先在终端运行 train_model.py 训练模型。")
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
                feat_imp_df = pd.DataFrame(
                    {"Feature": FEATURE_COLUMNS, "Importance": importances}
                )
                feat_imp_df = feat_imp_df.sort_values("Importance", ascending=True)

                import plotly.graph_objects as go

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
                            line=dict(color="#00FFCC", width=1.5),
                        ),
                        name="因子分裂次数",
                    )
                )

                fig_imp.update_layout(
                    title="模型决策树分裂频次排行 (分值越高代表因子越核心)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(showgrid=False),
                    height=450,
                    margin=dict(l=120, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("无法从当前保存的模型中解析特征重要性指标。")
        except Exception as e:
            st.error(f"解析特征重要性时出错: {e}")

    st.markdown("---")

    if len(model_df) > 0:
        st.subheader("🧩 最近模型版本历史")
        display_model_df = model_df.copy()
        if "features" in display_model_df.columns:
            display_model_df = display_model_df.drop(columns=["features"], errors="ignore")
        st.dataframe(display_model_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    if len(perf_df) > 0:
        st.subheader("📈 历史回填数据核心指标")

        col1, col2, col3, col4, col5 = st.columns(5)

        win_rate_1d = (perf_df["next_day_return"] > 0).mean()
        col1.metric(
            "次日胜率",
            f"{win_rate_1d:.1%}",
            help="推荐股票次日上涨的概率",
        )

        avg_return_1d = perf_df["next_day_return"].mean()
        col2.metric(
            "平均次日收益",
            f"{avg_return_1d:.2%}",
            help="推荐股票次日的平均收益率",
        )

        top1_df = perf_df[perf_df["rank"] == 1]
        if len(top1_df) > 0:
            top1_win_rate = (top1_df["next_day_return"] > 0).mean()
            col3.metric(
                "Top1 胜率",
                f"{top1_win_rate:.1%}",
                help="排名第一的股票次日上涨概率",
            )
        else:
            col3.metric("Top1 胜率", "暂无数据")

        col4.metric("样本数量", len(perf_df), help="参与统计的选股记录数")

        best_return = perf_df["next_day_return"].max()
        worst_return = perf_df["next_day_return"].min()
        col5.metric(
            "收益区间",
            f"{best_return:.1%} / {worst_return:.1%}",
            help="最大收益 / 最小收益",
        )

        st.markdown("---")
        st.subheader("📊 按排名分组表现")

        rank_stats = perf_df.groupby("rank").agg(
            平均收益=("next_day_return", "mean"),
            样本数=("next_day_return", "count"),
            胜率=("next_day_return", lambda x: (x > 0).mean()),
        )
        rank_stats["胜率"] = rank_stats["胜率"].apply(lambda x: f"{x:.1%}")
        rank_stats["平均收益"] = rank_stats["平均收益"].apply(lambda x: f"{x:.2%}")

        st.dataframe(rank_stats, use_container_width=True)

        st.markdown("---")
        st.subheader("📉 次日收益分布")

        try:
            import plotly.express as px

            fig_hist = px.histogram(
                perf_df,
                x="next_day_return",
                nbins=30,
                title="次日收益率分布",
                labels={"next_day_return": "次日收益率", "count": "频次"},
                color_discrete_sequence=["#00FFCC"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(perf_df["next_day_return"], bins=20, edgecolor="black", color="#00FFCC")
            ax.axvline(x=0, color="r", linestyle="--", linewidth=2)
            ax.set_xlabel("次日收益率")
            ax.set_ylabel("频次")
            ax.set_title("次日收益率分布")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.subheader("📈 历史表现趋势")

        daily_avg = (
            perf_df.groupby("trade_date")["next_day_return"]
            .agg(mean="mean", count="count")
            .reset_index()
        )
        daily_avg.columns = ["日期", "平均收益", "股票数量"]
        daily_avg = daily_avg.sort_values("日期")

        if len(daily_avg) > 1:
            try:
                import plotly.express as px

                fig_line = px.line(
                    daily_avg,
                    x="日期",
                    y="平均收益",
                    title="每日平均收益趋势",
                    markers=True,
                )
                fig_line.add_hline(y=0, line_dash="dash", line_color="red")
                fig_line.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception:
                st.line_chart(daily_avg.set_index("日期")["平均收益"])
        else:
            st.info("需要至少2个交易日的数据才能显示趋势图")
    else:
        st.info("📭 暂无真实选股后回填的收益率数据。")

# ----------------- TAB 5: 数据管理 -----------------
with tab_data:
    n_sel = int(query_df("SELECT COUNT(*) AS c FROM daily_selections")["c"].iloc[0])
    n_pred = int(query_df("SELECT COUNT(*) AS c FROM daily_predictions")["c"].iloc[0])
    n_mv = int(query_df("SELECT COUNT(*) AS c FROM model_versions")["c"].iloc[0])
    mx = query_df("SELECT MAX(trade_date) AS m FROM daily_selections")
    mx_d = mx.iloc[0]["m"] if not mx.empty else None

    st.markdown(f"**当前本地 SQLite 数据库路径**：`{DB_PATH}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已选股票总记录", n_sel)
    c2.metric("全市场预测总记录", n_pred)
    c3.metric("训练模型总版本", n_mv)
    c4.metric("最新选股日期", mx_d or "—")

    mv_all = query_df(
        "SELECT version, train_end_date, is_active, created_at FROM model_versions ORDER BY id DESC LIMIT 5"
    )
    st.subheader("🧩 最近注册的模型版本状态")
    st.dataframe(mv_all, hide_index=True, use_container_width=True)

    if st.button("🔄 手动回填历史推荐股票的真实收益率（次日 + 5日）", type="primary"):
        try:
            from src.return_updater import update_all_returns

            with st.spinner("正在拉取行情并更新数据库…"):
                out = update_all_returns()
            st.success(f"收益率回填更新完成：{out}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    full = query_df(
        """
        SELECT trade_date, rank, stock_code, stock_name, score, close_price,
               next_day_return, hold_5d_return
        FROM daily_selections
        ORDER BY trade_date DESC, rank ASC
        """
    )
    buf = io.StringIO()
    full.to_csv(buf, index=False)
    st.markdown("---")
    st.download_button(
        "💾 导出全部历史选股记录 CSV 文件",
        data=buf.getvalue().encode("utf-8-sig"),
        file_name="daily_selections_all.csv",
        mime="text/csv",
    )

# ----------------- TAB 6: 全市场预测 -----------------
with tab_pred:
    st.caption(
        "展示每日全市场股票打分结果（最近 30 个交易日）。历史数据已过滤并排除 ST 股票。"
    )
    hide_st = st.checkbox("隐藏 ST / 风险警示股票", value=True, key="hide_st_pred")
    dates = query_df(
        "SELECT DISTINCT trade_date FROM daily_predictions ORDER BY trade_date DESC LIMIT 30"
    )
    if dates.empty:
        st.info("暂无预测记录。请先运行每日预测脚本。")
    else:
        pick = st.selectbox(
            "选择交易日进行打分溯源", dates["trade_date"].tolist(), key="pred_date"
        )
        pred = query_df(
            """
            SELECT rank_in_market, stock_code, stock_name, score
            FROM daily_predictions
            WHERE trade_date = ?
            ORDER BY rank_in_market ASC
            LIMIT 500
            """,
            (pick,),
        )
        if hide_st and not pred.empty:
            mask = ~pred.apply(
                lambda r: is_st_stock_row(r.get("stock_code"), r.get("stock_name")),
                axis=1,
            )
            pred = pred[mask].reset_index(drop=True)
            pred["rank_in_market"] = range(1, len(pred) + 1)
        st.dataframe(pred, use_container_width=True, hide_index=True)

# ----------------- TAB 7: 系统设置 -----------------
with tab_settings:
    st.subheader("⚙️ 系统状态与配置管理")

    # 既然钉钉推送暂不使用，我们把钉钉配置归纳进折叠面板
    with st.expander("🔌 钉钉群机器人集成设置（暂不启用）"):
        ding_config = config_manager.get_dingtalk_config()
        time_options = ["09:30", "15:00", "16:00", "17:00", "18:00", "20:00"]
        try:
            send_time_index = time_options.index(ding_config.get("send_time", "16:00"))
        except ValueError:
            send_time_index = 2

        with st.form("dingtalk_config_form"):
            enabled = st.checkbox(
                "启用钉钉自动推送", value=bool(ding_config.get("enabled", False))
            )
            webhook_url = st.text_input(
                "Webhook 地址", value=ding_config.get("webhook_url", "")
            )
            secret = st.text_input(
                "加签密钥（可选）",
                value=ding_config.get("secret", ""),
                type="password",
            )
            send_time = st.selectbox(
                "期望推送时间（备忘）", options=time_options, index=send_time_index
            )

            c_save, c_test = st.columns(2)
            with c_save:
                submitted = st.form_submit_button(
                    "💾 保存配置", use_container_width=True
                )
            with c_test:
                test_btn = st.form_submit_button(
                    "🔔 测试推送", use_container_width=True
                )

        if submitted:
            if config_manager.set_dingtalk_config(
                enabled, webhook_url, secret, send_time
            ):
                st.success("✅ 配置已保存")
                st.rerun()

        if test_btn:
            try:
                from src.dingtalk_notifier import DingTalkNotifier

                if not webhook_url.strip():
                    st.warning("请先填写 Webhook")
                else:
                    DingTalkNotifier(webhook_url.strip(), secret.strip() or None).send_text(
                        "A-Quant Lite：钉钉测试推送"
                    )
                    st.success("已发送测试消息（若机器人正常应秒内收到）")
            except Exception as ex:
                st.error(f"测试失败: {ex}")

    # 新增系统运行状态面板
    st.markdown("---")
    st.subheader("🖥️ 宿主运行环境状态检测")

    import os
    import platform

    col_sys1, col_sys2, col_sys3 = st.columns(3)

    with col_sys1:
        st.markdown(
            f"""
            <div style="background-color: #141622; border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px;">
                <div style="color: #6272a4; font-size: 0.85rem;">操作系统架构</div>
                <div style="color: #ffffff; font-weight: bold; font-family: monospace; font-size: 1.1rem; margin-top: 5px;">{platform.system()} ({platform.machine()})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_sys2:
        st.markdown(
            f"""
            <div style="background-color: #141622; border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px;">
                <div style="color: #6272a4; font-size: 0.85rem;">Python 解释器版本</div>
                <div style="color: #ffffff; font-weight: bold; font-family: monospace; font-size: 1.1rem; margin-top: 5px;">{platform.python_version()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_sys3:
        db_size_mb = 0.0
        if Path(DB_PATH).exists():
            db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
        st.markdown(
            f"""
            <div style="background-color: #141622; border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px;">
                <div style="color: #6272a4; font-size: 0.85rem;">SQLite 数据库占用</div>
                <div style="color: #00FFCC; font-weight: bold; font-family: monospace; font-size: 1.1rem; margin-top: 5px;">{db_size_mb:.2f} MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
