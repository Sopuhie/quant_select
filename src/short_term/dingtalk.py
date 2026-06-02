# -*- coding: utf-8 -*-
"""短线选股钉钉推送（复用全局 Webhook，与中长线 Top3 模板独立）。"""
from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Any

from src.config_manager import CONFIG_PATH, config_manager
from src.dingtalk_notifier import DingTalkNotifier
from src.utils import display_trading_date_for_push

from .config import SHORT_HOLD_PLAN, SHORT_TOP_N
from .db import ensure_short_term_tables
from .trade_guide import build_trade_action_guide


def _short_push_notification_enabled() -> bool:
    notif = config_manager.config.get("notification") or {}
    if "send_short_on_success" in notif:
        return bool(notif.get("send_short_on_success"))
    return bool(notif.get("send_on_success", True))


def fetch_short_rows_for_dingtalk(
    conn,
    trade_date: str,
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """从 ``short_daily_selections`` 读取推送行。"""
    import pandas as pd

    ensure_short_term_tables(conn)
    lim = int(top_n if top_n is not None else SHORT_TOP_N)
    td = str(trade_date).strip()[:10]
    df = pd.read_sql_query(
        """
        SELECT rank, stock_code, stock_name,
               COALESCE(final_score, rule_score) AS rule_score,
               close_price,
               COALESCE(pct_change, day_change_pct) AS day_change_pct,
               COALESCE(volume_ratio_5d, vol_ratio_5d) AS vol_ratio_5d,
               kdj_j, advice_text, hold_plan
        FROM short_daily_selections
        WHERE trade_date = ?
        ORDER BY COALESCE(final_score, rule_score) DESC, rank ASC
        LIMIT ?
        """,
        conn,
        params=[td, lim],
    )
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "rank": int(r["rank"]),
                "stock_code": str(r["stock_code"]).strip().zfill(6),
                "stock_name": str(r.get("stock_name") or "").strip(),
                "rule_score": float(r["rule_score"])
                if pd.notna(r.get("rule_score"))
                else None,
                "close_price": float(r["close_price"])
                if pd.notna(r.get("close_price"))
                else None,
                "day_change_pct": float(r["day_change_pct"])
                if pd.notna(r.get("day_change_pct"))
                else None,
                "vol_ratio_5d": float(r["vol_ratio_5d"])
                if pd.notna(r.get("vol_ratio_5d"))
                else None,
                "kdj_j": float(r["kdj_j"]) if pd.notna(r.get("kdj_j")) else None,
                "advice_text": str(r.get("advice_text") or "").strip(),
                "hold_plan": str(r.get("hold_plan") or "").strip(),
            }
        )
    return rows


def build_short_selection_markdown(
    trade_date: str,
    selections: list[dict[str, Any]],
    *,
    market_score: int | None = None,
) -> tuple[str, str]:
    """返回 ``(title, markdown_text)``。"""
    title = "短线规则选股（1日）"
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    td_disp = display_trading_date_for_push(trade_date)
    td_esc = html.escape(td_disp)
    mkt_line = ""
    if market_score is not None:
        mkt_line = f"**大盘环境分：{int(market_score)}**  \n"

    plan = SHORT_HOLD_PLAN
    lines: list[str] = [
        f"⚡ **{title}**  ",
        f"**信号日：{td_esc}**  ",
        mkt_line,
        f"**执行计划：** {html.escape(plan)}  ",
        "",
        "---",
        "",
    ]

    for s in selections:
        rank = int(s.get("rank") or 0)
        code = html.escape(str(s.get("stock_code", "")).zfill(6))
        name = html.escape(str(s.get("stock_name", "")).strip())
        close = s.get("close_price")
        chg = s.get("day_change_pct")
        vr5 = s.get("vol_ratio_5d")
        score = s.get("rule_score")
        parts: list[str] = []
        if close is not None:
            parts.append(f"收 {float(close):.2f}")
        if chg is not None:
            parts.append(f"日涨 {float(chg) * 100:+.2f}%")
        if vr5 is not None:
            parts.append(f"量比 {float(vr5):.2f}")
        if score is not None:
            parts.append(f"得分 {float(score):.1f}")
        meta = " · ".join(parts) if parts else "—"
        advice = str(s.get("advice_text") or "").strip()
        advice_esc = html.escape(advice) if advice else "—"
        close_px = s.get("close_price")
        guide_text = ""
        if close_px is not None:
            try:
                guide = build_trade_action_guide(float(close_px))
                guide_text = html.escape(str(guide.get("summary_text") or ""))
            except (TypeError, ValueError):
                guide_text = ""
        lines.append(f"**{rank}. {code} {name}**  ")
        lines.append(f"{meta}  ")
        lines.append(f"> {advice_esc}  ")
        if guide_text:
            lines.append("")
            lines.append("**操作价位**  ")
            for ln in guide_text.split("\n"):
                if ln.strip():
                    lines.append(f"> {ln}  ")
        lines.append("")

    footer = (
        "🤖 短线规则模块自动生成  \n"
        f"⏰ 推送时间：{now_s}  \n"
        "⚠️ 超短线波动大，T+1 仅买 T+2 卖；按推送价位严格执行止盈/止损；仅供参考，不构成投资建议"
    )
    lines.extend(["---", "", footer])
    return title, "\n".join(lines)


def maybe_push_short_selections(
    trade_date: str,
    *,
    market_score: int | None = None,
    db_path=None,
) -> bool:
    """
    若钉钉已启用且允许成功推送，则发送短线选股列表。
    环境变量 ``QUANT_SHORT_SKIP_DINGTALK=1`` 可跳过。
    """
    from src.config import DB_PATH
    from src.database import get_connection, init_db

    if os.environ.get("QUANT_SHORT_SKIP_DINGTALK", "").strip() in (
        "1",
        "true",
        "True",
    ):
        print("已跳过短线钉钉推送（QUANT_SHORT_SKIP_DINGTALK）", flush=True)
        return False

    path = db_path or DB_PATH
    init_db(path)
    config_manager.reload()

    if not config_manager.is_dingtalk_enabled():
        ding = config_manager.get_dingtalk_config()
        wh = (ding.get("webhook_url") or "").strip()
        if not wh and not os.environ.get("QUANT_DINGTALK_WEBHOOK", "").strip():
            print(
                "短线钉钉未推送：未配置 Webhook。"
                f"请在 Streamlit「环境设置」填写，或设置 QUANT_DINGTALK_WEBHOOK。"
                f"（配置文件：{CONFIG_PATH}）",
                flush=True,
            )
        elif ding.get("enabled") is False:
            print(
                "短线钉钉未推送：dingtalk.enabled 为 false。",
                flush=True,
            )
        else:
            print("短线钉钉未推送：钉钉未启用", flush=True)
        return False

    if not _short_push_notification_enabled():
        print("配置已关闭短线成功推送（notification.send_short_on_success / send_on_success）", flush=True)
        return False

    with get_connection(path) as conn:
        rows = fetch_short_rows_for_dingtalk(conn, trade_date)

    if not rows:
        print(f"无短线选股记录（{trade_date}），跳过钉钉推送", flush=True)
        return False

    title, text = build_short_selection_markdown(
        trade_date, rows, market_score=market_score
    )
    notifier = DingTalkNotifier(
        config_manager.get_dingtalk_webhook_url(),
        config_manager.get_dingtalk_secret() or None,
    )
    ok = notifier.send_markdown(title, text)
    if ok:
        print("短线钉钉推送成功", flush=True)
    else:
        print("短线钉钉推送失败，请检查 Webhook、加签与网络", flush=True)
    return ok
