"""
Streamlit 复盘界面。

启动（在 quant_select 目录下）:
  streamlit run app.py
"""
from __future__ import annotations

import io
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import DB_PATH, TOP_N_SELECTION
from src.config_manager import CONFIG_PATH, config_manager
from src.database import init_db, query_df
from src.kline_chart import draw_candlestick, get_stock_kline_data
from src.predictor import is_st_stock_row

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore[misc, assignment]

st.set_page_config(page_title="量化选股复盘", layout="wide")
init_db()

BACKTEST_CSV = ROOT / "data" / "backtest_results.csv"
BACKTEST_TRADING_DAYS_YR = 242


def _backtest_perf_metrics(df: pd.DataFrame, rf_annual: float = 0.03) -> dict[str, Any]:
    """由 nav 序列计算回测看板指标（与 scripts/backtest.py 口径一致）。"""
    nav = pd.to_numeric(df["nav"], errors="coerce").astype(float)
    nav = nav.dropna()
    if nav.empty or len(nav) < 2:
        return {}
    equity = nav / float(nav.iloc[0])
    cum_ret = float(equity.iloc[-1] - 1.0)
    n = len(nav)
    ann_factor = BACKTEST_TRADING_DAYS_YR / max(n - 1, 1)
    ann_ret = float((equity.iloc[-1] ** ann_factor) - 1.0)

    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    trough_i = int(dd.idxmin())
    peak_i = int(equity.iloc[: trough_i + 1].idxmax())
    dd_start = str(df["trade_date"].iloc[peak_i])
    dd_end = str(df["trade_date"].iloc[trough_i])

    daily_ret = nav.pct_change()
    daily_ret = daily_ret.fillna(0.0)
    rf_d = (1.0 + rf_annual) ** (1.0 / BACKTEST_TRADING_DAYS_YR) - 1.0
    excess = daily_ret - rf_d
    tail = daily_ret.iloc[1:]
    std_d = float(tail.std(ddof=1)) if len(tail) > 1 else float("nan")
    sharpe = (
        float(excess.iloc[1:].mean() / std_d * np.sqrt(BACKTEST_TRADING_DAYS_YR))
        if std_d > 1e-12
        else float("nan")
    )
    win_daily = float((tail > 0).mean()) if len(tail) > 0 else float("nan")

    bench_cum = float("nan")
    excess_cum = float("nan")
    if "benchmark_close" in df.columns and df["benchmark_close"].notna().any():
        bc = pd.to_numeric(df["benchmark_close"], errors="coerce")
        m = pd.DataFrame({"trade_date": df["trade_date"], "nav": nav.values, "bc": bc.values})
        m = m.dropna(subset=["bc"])
        if len(m) >= 2:
            be = m["bc"].astype(float)
            bench_equity = be / float(be.iloc[0])
            strat_equity = m["nav"].astype(float) / float(m["nav"].iloc[0])
            bench_cum = float(bench_equity.iloc[-1] - 1.0)
            excess_cum = float(strat_equity.iloc[-1] - bench_equity.iloc[-1])

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "max_drawdown": mdd,
        "drawdown_start": dd_start,
        "drawdown_end": dd_end,
        "sharpe_ratio": sharpe,
        "win_rate_daily": win_daily,
        "benchmark_cumulative_return": bench_cum,
        "excess_cumulative_return": excess_cum,
        "n_days": n,
    }


st.title("量化选股 · 复盘与数据")


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
    """今日推荐展示用：只认 rank 1–TOP_N，同一 rank 多条时保留 score 最高的一条。"""
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


tab_today, tab_hist, tab_backtest, tab_perf, tab_data, tab_pred, tab_settings = st.tabs(
    [
        "今日推荐",
        "历史复盘",
        "📈 历史回测",
        "模型表现",
        "数据管理",
        "全市场预测",
        "⚙️ 系统设置",
    ]
)

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

        if today_df.empty:
            pass
        else:
            cols = st.columns(len(today_df))
            for i, col in enumerate(cols):
                row = today_df.iloc[i]
                with col:
                    st.markdown(
                        f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 15px;
                    margin: 10px;
                    text-align: center;
                ">
                    <h2 style="color: white; margin: 0;">#{int(row["rank"])}</h2>
                    <h3 style="color: white; margin: 10px 0;">{row["stock_code"]}</h3>
                    <h4 style="color: white; margin: 5px 0;">{row["stock_name"]}</h4>
                    <p style="color: #ddd; margin: 5px 0;">分数: {float(row["score"]):.4f}</p>
                    <p style="color: #ddd; margin: 5px 0;">收盘: {row["close_price"]}</p>
                </div>
                """,
                        unsafe_allow_html=True,
                    )
                    if pd.notna(row.get("next_day_return")):
                        h5 = row.get("hold_5d_return")
                        if pd.notna(h5):
                            st.caption(
                                f"已回填：次日 {float(row['next_day_return']):.2%} · "
                                f"5日 {float(h5):.2%}"
                            )
                        else:
                            st.caption(f"已回填：次日 {float(row['next_day_return']):.2%}")

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

with tab_backtest:
    st.subheader("📈 策略历史回测分析")
    st.caption(
        "数据来自 `data/backtest_results.csv`（由 `scripts/backtest.py` 生成）。"
        "图表为暗黑霓虹风格；基准列为空时仅展示策略曲线。"
    )

    if not BACKTEST_CSV.exists():
        st.info("📊 尚未生成历史回测数据。可点击下方按钮运行 **2024 全年**滚动回测（耗时取决于股票池与网络）。")
        st.markdown(
            "或在项目根目录执行：  \n`python scripts/backtest.py --start-date 2024-01-01 --end-date 2024-12-31`"
        )
        if st.button("🚀 一键运行历史回测 (2024年)", type="primary", key="run_backtest_2024"):
            with st.spinner("回测运行中：正在拉取行情并滚动打分，请耐心等待…"):
                try:
                    proc = subprocess.run(
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
                        timeout=7200,
                    )
                    if proc.returncode != 0:
                        st.error(f"回测进程退出码 {proc.returncode}")
                        if proc.stderr:
                            st.code(proc.stderr[-8000:], language="text")
                        if proc.stdout:
                            with st.expander("stdout"):
                                st.code(proc.stdout[-8000:], language="text")
                    elif not BACKTEST_CSV.exists():
                        st.warning("进程已结束但未找到 backtest_results.csv，请查看上方日志。")
                    else:
                        st.success("回测执行完成，已生成 data/backtest_results.csv")
                        st.rerun()
                except subprocess.TimeoutExpired:
                    st.error("回测超时（>2h），请在终端手动运行或缩小股票池。")
                except Exception as e:
                    st.error(f"回测执行失败: {e}")
    else:
        try:
            df_res = pd.read_csv(BACKTEST_CSV)
        except Exception as e:
            st.error(f"无法读取回测 CSV：{e}")
            df_res = pd.DataFrame()

        if df_res.empty or "nav" not in df_res.columns or "trade_date" not in df_res.columns:
            st.warning("回测文件格式异常或为空，请删除后重新运行回测。")
            if st.button("🗑️ 删除无效结果并重试", key="del_bad_backtest"):
                BACKTEST_CSV.unlink(missing_ok=True)
                st.rerun()
        else:
            df_res = df_res.copy()
            df_res["trade_date"] = df_res["trade_date"].astype(str)
            nav0 = pd.to_numeric(df_res["nav"], errors="coerce").iloc[0]
            if pd.isna(nav0) or nav0 == 0:
                st.error("净值序列无效（nav 首行为空或为 0）。")
            else:
                df_res["strategy_cum_pct"] = (
                    pd.to_numeric(df_res["nav"], errors="coerce") / float(nav0) - 1.0
                ) * 100.0

                has_bench = (
                    "benchmark_close" in df_res.columns
                    and pd.to_numeric(df_res["benchmark_close"], errors="coerce").notna().any()
                )
                if has_bench:
                    bc = pd.to_numeric(df_res["benchmark_close"], errors="coerce")
                    first_b = bc.dropna().iloc[0]
                    if first_b and first_b > 0:
                        df_res["bench_cum_pct"] = (bc / float(first_b) - 1.0) * 100.0
                    else:
                        has_bench = False

                metrics = _backtest_perf_metrics(df_res)

                if go is not None:
                    fig_bt = go.Figure()
                    fig_bt.add_trace(
                        go.Scatter(
                            x=df_res["trade_date"],
                            y=df_res["strategy_cum_pct"],
                            mode="lines",
                            name="A-Quant Lite（策略）",
                            line=dict(color="#00FFCC", width=3),
                            fill="tozeroy",
                            fillcolor="rgba(0, 255, 204, 0.12)",
                        )
                    )
                    if has_bench and "bench_cum_pct" in df_res.columns:
                        fig_bt.add_trace(
                            go.Scatter(
                                x=df_res["trade_date"],
                                y=df_res["bench_cum_pct"],
                                mode="lines",
                                name="沪深300（基准）",
                                line=dict(color="#FF9900", width=2, dash="dash"),
                                fill="tozeroy",
                                fillcolor="rgba(255, 153, 0, 0.06)",
                            )
                        )
                    fig_bt.update_layout(
                        title=dict(
                            text="策略 vs 沪深300 · 累计收益率 (%)",
                            font=dict(size=18, color="#E0E0E0"),
                        ),
                        template="plotly_dark",
                        paper_bgcolor="rgba(10,12,18,0.95)",
                        plot_bgcolor="rgba(15,18,28,0.9)",
                        font=dict(color="#CCCCCC"),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(255,255,255,0.08)",
                            zeroline=False,
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(255,255,255,0.08)",
                            ticksuffix="%",
                            zeroline=True,
                            zerolinecolor="rgba(0,255,204,0.35)",
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            bgcolor="rgba(0,0,0,0.4)",
                        ),
                        margin=dict(l=48, r=24, t=60, b=48),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)
                else:
                    st.warning("未安装 plotly，请执行 pip install plotly")

                st.markdown("##### 核心指标")
                mcols_a = st.columns(3)
                with mcols_a[0]:
                    cr = metrics.get("cumulative_return", float("nan"))
                    ex = metrics.get("excess_cumulative_return", float("nan"))
                    st.metric(
                        "累计收益率",
                        f"{cr:.2%}" if np.isfinite(cr) else "—",
                        delta=f"超额 {ex:.2%}" if np.isfinite(ex) else None,
                        delta_color="normal",
                    )
                with mcols_a[1]:
                    ar = metrics.get("annualized_return", float("nan"))
                    st.metric(
                        "年化收益率",
                        f"{ar:.2%}" if np.isfinite(ar) else "—",
                        help=f"按每年 {BACKTEST_TRADING_DAYS_YR} 个交易日折算",
                    )
                with mcols_a[2]:
                    bc = metrics.get("benchmark_cumulative_return", float("nan"))
                    st.metric(
                        "基准累计收益",
                        f"{bc:.2%}" if np.isfinite(bc) else "—（无基准列）",
                    )

                mcols_b = st.columns(3)
                with mcols_b[0]:
                    mdd = metrics.get("max_drawdown", float("nan"))
                    st.metric(
                        "最大回撤",
                        f"{mdd:.2%}" if np.isfinite(mdd) else "—",
                        help=(
                            f"{metrics.get('drawdown_start', '')} → {metrics.get('drawdown_end', '')}"
                            if metrics.get("drawdown_start")
                            else None
                        ),
                    )
                with mcols_b[1]:
                    sh = metrics.get("sharpe_ratio", float("nan"))
                    st.metric("夏普比率", f"{sh:.3f}" if np.isfinite(sh) else "—", help="Rf≈3% 年化")
                with mcols_b[2]:
                    wd = metrics.get("win_rate_daily", float("nan"))
                    st.metric(
                        "胜率（按日）",
                        f"{wd:.1%}" if np.isfinite(wd) else "—",
                    )

                mcols_c = st.columns(2)
                with mcols_c[0]:
                    st.metric(
                        "胜率（按平仓）",
                        "见脚本日志",
                        help="CSV 未写入单笔平仓；运行 scripts/backtest.py 时终端会打印「按平仓笔数」胜率。",
                    )
                with mcols_c[1]:
                    st.metric("回测样本天数", f"{metrics.get('n_days', '—')}")

                st.caption(
                    f"最大回撤区间：{metrics.get('drawdown_start', '—')} → {metrics.get('drawdown_end', '—')}"
                )

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔄 重新跑回测（删除 CSV）", key="rerun_backtest_del"):
                        BACKTEST_CSV.unlink(missing_ok=True)
                        st.rerun()
                with c2:
                    if st.button("🔁 再次执行 2024 回测（覆盖）", key="rerun_backtest_ovr"):
                        with st.spinner("正在运行 2024 回测…"):
                            try:
                                proc = subprocess.run(
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
                                    timeout=7200,
                                )
                                if proc.returncode != 0:
                                    st.error(f"退出码 {proc.returncode}")
                                    if proc.stderr:
                                        st.code(proc.stderr[-6000:], language="text")
                                else:
                                    st.success("已完成覆盖写入")
                                    st.rerun()
                            except Exception as e:
                                st.error(str(e))

                with st.expander("📋 资产明细（净值 / 现金 / 持仓市值）", expanded=False):
                    show_cols = [
                        c
                        for c in (
                            "trade_date",
                            "nav",
                            "cash",
                            "hold_mv",
                            "n_positions",
                            "daily_return",
                            "benchmark_close",
                        )
                        if c in df_res.columns
                    ]
                    disp = df_res[show_cols].copy()
                    if "daily_return" in disp.columns:
                        disp["daily_return"] = pd.to_numeric(
                            disp["daily_return"], errors="coerce"
                        ).map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
                    if "nav" in disp.columns:
                        disp["nav"] = pd.to_numeric(disp["nav"], errors="coerce").map(
                            lambda x: f"{x:,.2f}" if pd.notna(x) else "—"
                        )
                    if "cash" in disp.columns:
                        disp["cash"] = pd.to_numeric(disp["cash"], errors="coerce").map(
                            lambda x: f"{x:,.2f}" if pd.notna(x) else "—"
                        )
                    if "hold_mv" in disp.columns:
                        disp["hold_mv"] = pd.to_numeric(disp["hold_mv"], errors="coerce").map(
                            lambda x: f"{x:,.2f}" if pd.notna(x) else "—"
                        )
                    st.dataframe(disp, use_container_width=True, hide_index=True)

with tab_perf:
    st.subheader("📊 模型表现概览")

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
        SELECT version, train_end_date, is_active, created_at
        FROM model_versions
        ORDER BY created_at DESC
        LIMIT 5
        """,
        conn,
    )

    conn.close()

    if len(model_df) > 0:
        st.subheader("🧩 最近模型版本")
        st.dataframe(model_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    if len(perf_df) > 0:
        st.subheader("📈 核心指标")

        col1, col2, col3, col4, col5 = st.columns(5)

        win_rate_1d = (perf_df["next_day_return"] > 0).mean()
        col1.metric(
            "次日胜率",
            f"{win_rate_1d:.1%}",
            delta=None,
            help="推荐股票次日上涨的概率",
        )

        avg_return_1d = perf_df["next_day_return"].mean()
        col2.metric(
            "平均次日收益",
            f"{avg_return_1d:.2%}",
            delta=None,
            help="推荐股票次日的平均收益率",
        )

        top1_df = perf_df[perf_df["rank"] == 1]
        if len(top1_df) > 0:
            top1_win_rate = (top1_df["next_day_return"] > 0).mean()
            col3.metric(
                "Top1 胜率",
                f"{top1_win_rate:.1%}",
                delta=None,
                help="排名第一的股票次日上涨概率",
            )
        else:
            col3.metric("Top1 胜率", "暂无数据")

        col4.metric(
            "样本数量",
            len(perf_df),
            help="参与统计的选股记录数",
        )

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
                color_discrete_sequence=["#667eea"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        except ImportError:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(perf_df["next_day_return"], bins=20, edgecolor="black", color="#667eea")
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
                st.plotly_chart(fig_line, use_container_width=True)
            except ImportError:
                st.line_chart(daily_avg.set_index("日期")["平均收益"])
        else:
            st.info("需要至少2个交易日的数据才能显示趋势图")

    else:
        st.info("📭 暂无收益率数据")
        st.markdown(
            """
### 如何获取收益率数据？

1. **等待下一个交易日**：选股后需要等到次日收盘才能计算收益。
2. **手动运行更新脚本**：在项目目录执行 `python scripts/update_returns.py`。
3. **查看选股记录**：切换到「历史复盘」查看已选股票。
"""
        )

with tab_data:
    n_sel = int(query_df("SELECT COUNT(*) AS c FROM daily_selections")["c"].iloc[0])
    n_pred = int(query_df("SELECT COUNT(*) AS c FROM daily_predictions")["c"].iloc[0])
    n_mv = int(query_df("SELECT COUNT(*) AS c FROM model_versions")["c"].iloc[0])
    mx = query_df("SELECT MAX(trade_date) AS m FROM daily_selections")
    mx_d = mx.iloc[0]["m"] if not mx.empty else None
    st.markdown(f"**数据库** `{DB_PATH}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("选股记录数", n_sel)
    c2.metric("预测记录数", n_pred)
    c3.metric("模型版本数", n_mv)
    c4.metric("最新选股日", mx_d or "—")

    mv = query_df("SELECT version, train_end_date, is_active, created_at FROM model_versions ORDER BY id DESC LIMIT 5")
    st.subheader("最近模型版本")
    st.dataframe(mv, hide_index=True, use_container_width=True)

    if st.button("手动回填收益（次日 + 5日）", type="primary"):
        try:
            from src.return_updater import update_all_returns

            with st.spinner("正在拉取行情并更新数据库…"):
                out = update_all_returns()
            st.success(f"更新完成：{out}")
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
    st.download_button(
        "导出选股 CSV",
        data=buf.getvalue().encode("utf-8-sig"),
        file_name="daily_selections.csv",
        mime="text/csv",
    )

with tab_pred:
    st.caption("每日全市场打分（最近 30 个交易日）；历史数据可能含 ST，可用开关过滤展示。")
    hide_st = st.checkbox("隐藏 ST / 风险警示股", value=True, key="hide_st_pred")
    dates = query_df(
        "SELECT DISTINCT trade_date FROM daily_predictions ORDER BY trade_date DESC LIMIT 30"
    )
    if dates.empty:
        st.info("暂无预测记录。")
    else:
        pick = st.selectbox("选择交易日", dates["trade_date"].tolist(), key="pred_date")
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

with tab_settings:
    st.subheader("⚙️ 钉钉推送配置")

    ding_config = config_manager.get_dingtalk_config()

    st.markdown(
        """
### 📱 钉钉机器人配置指南

1. 在钉钉群中添加「自定义机器人」
2. 复制 Webhook 地址
3. 若启用「加签」，复制 SEC 开头的密钥并填入下方

> 💡 **提示**：建议先在测试群验证，再切换到正式群。
"""
    )

    time_options = ["09:30", "15:00", "16:00", "17:00", "18:00", "20:00"]
    try:
        send_time_index = time_options.index(ding_config.get("send_time", "16:00"))
    except ValueError:
        send_time_index = 2

    with st.form("dingtalk_config_form"):
        st.markdown("#### 🔧 基本配置")

        enabled = st.checkbox(
            "✅ 启用钉钉推送",
            value=bool(ding_config.get("enabled", False)),
            help="开启后，`run_daily.py` / start.cmd 选股成功会自动推送「今日推荐」。"
            "若控制台仍提示未启用：请勾选此项并保存；仅手写 config.json 时勿省略 enabled 或勿写为 false。",
        )

        webhook_url = st.text_input(
            "Webhook 地址",
            value=ding_config.get("webhook_url", ""),
            placeholder="https://oapi.dingtalk.com/robot/send?access_token=xxx",
            help="钉钉自定义机器人的 Webhook",
        )

        secret = st.text_input(
            "加签密钥（可选）",
            value=ding_config.get("secret", ""),
            type="password",
            placeholder="SEC…（未启用加签可留空）",
            help="机器人安全设置为「加签」时填写",
        )

        send_time = st.selectbox(
            "期望推送时间（备忘）",
            options=time_options,
            index=send_time_index,
            help="实际推送取决于任务计划何时执行 `run_daily.py`，此处仅作记录。",
        )

        st.markdown("---")
        st.markdown("#### 🧪 测试与保存")

        c_save, c_test = st.columns(2)
        with c_save:
            submitted = st.form_submit_button("💾 保存配置", use_container_width=True)
        with c_test:
            test_btn = st.form_submit_button("🔔 测试推送", use_container_width=True)

    if submitted:
        if enabled and not (webhook_url or "").strip():
            st.error("❌ 启用推送时，Webhook 地址不能为空")
        else:
            if config_manager.set_dingtalk_config(
                enabled, webhook_url, secret, send_time
            ):
                st.success("✅ 配置已保存")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ 保存失败，请检查磁盘权限或路径")

    if test_btn:
        if not (webhook_url or "").strip():
            st.error("❌ 请先填写 Webhook 地址")
        else:
            try:
                from src.dingtalk_notifier import DingTalkNotifier

                notifier = DingTalkNotifier(
                    webhook_url.strip(),
                    secret.strip() or None,
                )
                test_selections = [
                    {"rank": 1, "stock_code": "000001", "stock_name": "测试股票1"},
                    {"rank": 2, "stock_code": "000002", "stock_name": "测试股票2"},
                    {"rank": 3, "stock_code": "000003", "stock_name": "测试股票3"},
                ]
                with st.spinner("正在发送测试消息…"):
                    ok = notifier.send_stock_selection("测试推送", test_selections)
                if ok:
                    st.success("✅ 测试推送成功，请到钉钉群查看")
                else:
                    st.error("❌ 推送失败，请检查 Webhook、加签与网络")
            except Exception as e:
                st.error(f"测试推送异常: {e}")

    st.markdown("---")
    st.subheader("📋 当前配置状态")

    ding_cfg_show = config_manager.get_dingtalk_config()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ok_push = bool(ding_cfg_show.get("enabled")) and bool(
            (ding_cfg_show.get("webhook_url") or "").strip()
        )
        st.metric(
            "推送状态",
            f"{'🟢 已就绪' if ok_push else '🔴 未启用/未完成'}",
        )
    with col_b:
        preview = ding_cfg_show.get("webhook_url") or "未配置"
        if len(preview) > 40:
            preview = preview[:40] + "…"
        st.metric("Webhook", preview)
    with col_c:
        st.metric("备忘时间", ding_cfg_show.get("send_time", "16:00"))

    st.caption(f"📁 配置文件：`{CONFIG_PATH.resolve()}`（建议勿提交 Git）")
