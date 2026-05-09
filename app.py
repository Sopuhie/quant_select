"""
Streamlit 复盘界面。

启动（在 quant_select 目录下）:
  streamlit run app.py
"""
from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

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

st.set_page_config(page_title="量化选股复盘", layout="wide")
init_db()

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


tab_today, tab_hist, tab_perf, tab_data, tab_pred, tab_settings = st.tabs(
    ["今日推荐", "历史复盘", "模型表现", "数据管理", "全市场预测", "⚙️ 系统设置"]
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
