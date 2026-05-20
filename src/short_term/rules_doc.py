# -*- coding: utf-8 -*-
"""短线规则选股说明文案（与 ``strategy.py`` 逻辑一致，供 Streamlit 复盘展示）。"""
from __future__ import annotations

from .config import (
    SHORT_EXCLUDE_BJ,
    SHORT_EXCLUDE_NEAR_LIMIT,
    SHORT_EXCLUDE_ST,
    SHORT_HOLDING_DAYS,
    SHORT_KDJ_J_MAX,
    SHORT_KDJ_J_MIN,
    SHORT_KDJ_J_SLOPE_MIN,
    SHORT_MA_FAST,
    SHORT_MA_SLOW,
    SHORT_MAX_5D_RETURN,
    SHORT_MIN_AMOUNT,
    SHORT_MIN_HISTORY_BARS,
    SHORT_MIN_MARKET_SCORE,
    SHORT_MIN_TURNOVER,
    SHORT_TOP_N,
    SHORT_VOL_RATIO_1D_MIN,
    SHORT_VOL_RATIO_5D_MIN,
)


def _on_off(flag: bool) -> str:
    return "开启" if flag else "关闭"


def short_term_rules_sections() -> list[dict[str, str]]:
    """结构化规则条目，便于表格或复盘对照。"""
    hold_plan = (
        f"T 日收盘确认信号 → T+1 开盘买入 → T+{1 + SHORT_HOLDING_DAYS} 开盘卖出"
        f"（持有 {SHORT_HOLDING_DAYS} 个交易日）"
    )
    amt_wan = SHORT_MIN_AMOUNT / 1e4
    return [
        {"类别": "执行计划", "规则": hold_plan},
        {"类别": "输出数量", "规则": f"按规则得分降序取 Top {SHORT_TOP_N}（环境变量 QUANT_SHORT_TOP_N）"},
        {
            "类别": "大盘环境",
            "规则": (
                f"沪深300 环境分 ≥ {SHORT_MIN_MARKET_SCORE} 才扫描；"
                "收盘价高于 MA20 为 60 分，否则 20 分；无指数数据时默认 50 分"
            ),
        },
        {
            "类别": "股票池",
            "规则": (
                f"剔除 ST（{_on_off(SHORT_EXCLUDE_ST)}）；"
                f"剔除北交所等（{_on_off(SHORT_EXCLUDE_BJ)}）；"
                f"信号日停牌剔除；涨停日剔除；"
                f"近涨停剔除（{_on_off(SHORT_EXCLUDE_NEAR_LIMIT)}，见板块阈值）"
            ),
        },
        {
            "类别": "流动性",
            "规则": (
                f"信号日成交额 ≥ {amt_wan:.0f} 万 **或** 换手率 ≥ {SHORT_MIN_TURNOVER:.1f}%"
                "（否则视为僵尸微盘剔除）"
            ),
        },
        {
            "类别": "历史K线",
            "规则": f"至少 {SHORT_MIN_HISTORY_BARS} 根日线，且最后一根为信号日",
        },
        {
            "类别": "趋势",
            "规则": (
                f"收盘价 > MA{SHORT_MA_FAST} 且 MA{SHORT_MA_FAST} ≥ MA{SHORT_MA_SLOW}；"
                "当日收阳（收盘 > 昨收）"
            ),
        },
        {
            "类别": "当日涨跌幅",
            "规则": (
                "主板：约 -0.5% ~ +7.5%，近涨停 ≥9.5% 剔除；"
                "创业板/科创板：约 -0.5% ~ +15%，近涨停 ≥19.2% 剔除；"
                "北交所：约 -0.5% ~ +25%，近涨停 ≥29.2% 剔除"
            ),
        },
        {
            "类别": "中期动量",
            "规则": f"近 5 日涨幅 ≤ {SHORT_MAX_5D_RETURN * 100:.0f}%（防过度拉升）",
        },
        {
            "类别": "量能",
            "规则": (
                f"5 日量比 ≥ {SHORT_VOL_RATIO_5D_MIN:.2f}；"
                f"1 日量比 ≥ {SHORT_VOL_RATIO_1D_MIN:.2f}"
            ),
        },
        {
            "类别": "MACD",
            "规则": "DIF > DEA（金叉状态）；MACD 柱 ≥ 前柱 ×0.8（红柱不显著缩短）",
        },
        {
            "类别": "KDJ",
            "规则": (
                f"K > D；J ∈ [{SHORT_KDJ_J_MIN:.0f}, {SHORT_KDJ_J_MAX:.0f}]；"
                f"J 较前一日上升 ≥ {SHORT_KDJ_J_SLOPE_MIN:.1f}"
            ),
        },
        {
            "类别": "规则得分",
            "规则": (
                "日涨幅(20%) + 5日量比(25%) + MACD柱改善(25%) + J斜率(15%)，"
                "得分越高排序越靠前"
            ),
        },
        {
            "类别": "实盘建议",
            "规则": (
                "J≥88：极小仓；当日涨幅≥6%：警惕高开不及预期止损；"
                "否则：T+1 买 / T+2 卖"
            ),
        },
    ]


def format_short_term_rules_markdown() -> str:
    """钉钉/Streamlit 用 Markdown（行尾两空格在钉钉中可换行，此处用普通换行）。"""
    lines = [
        "**执行与输出**",
        f"- 持有：T 日信号 → T+1 开盘买 → T+{1 + SHORT_HOLDING_DAYS} 开盘卖",
        f"- Top **{SHORT_TOP_N}**；落库表 ``short_daily_selections`` + ``short_today.json``",
        "",
        "**大盘卡口**",
        f"- 沪深300 环境分 **≥ {SHORT_MIN_MARKET_SCORE}** 才出票（与中长线熔断算法一致：价>MA20→60分，否则20分）",
        "",
        "**硬性剔除**",
        f"- ST：{_on_off(SHORT_EXCLUDE_ST)} · 北交所等：{_on_off(SHORT_EXCLUDE_BJ)} · 停牌 · 一字/涨停日",
        f"- 流动性：成交额 **≥ {SHORT_MIN_AMOUNT/1e4:.0f} 万** 或换手 **≥ {SHORT_MIN_TURNOVER:.1f}%**",
        f"- 近涨停剔除：{_on_off(SHORT_EXCLUDE_NEAR_LIMIT)}（主板≥9.5%，创/科≥19.2%）",
        "",
        "**共振条件（须全部满足）**",
        f"1. 趋势：收 > MA{SHORT_MA_FAST} 且 MA{SHORT_MA_FAST} ≥ MA{SHORT_MA_SLOW}，当日收阳",
        f"2. 当日涨幅在板块允许区间内",
        f"3. 5 日涨幅 ≤ {SHORT_MAX_5D_RETURN*100:.0f}%",
        f"4. 量比：5日 ≥ {SHORT_VOL_RATIO_5D_MIN:.2f}，1日 ≥ {SHORT_VOL_RATIO_1D_MIN:.2f}",
        "5. MACD：DIF>DEA，柱线不弱于前日80%",
        f"6. KDJ：K>D，J∈[{SHORT_KDJ_J_MIN:.0f},{SHORT_KDJ_J_MAX:.0f}]，J上升≥{SHORT_KDJ_J_SLOPE_MIN:.1f}",
        "",
        "**排序得分**",
        "日涨幅 + 5日量比 + MACD柱改善 + J斜率（加权求和，见代码 ``_rule_score``）",
    ]
    return "\n".join(lines)


CHECK_LABELS: dict[str, str] = {
    "trend_ma": "趋势均线",
    "bullish_return": "当日收阳",
    "day_return_band": "当日涨跌幅区间",
    "momentum_5d_cap": "5日涨幅上限",
    "volume": "量比共振",
    "macd_cross": "MACD金叉",
    "macd_slope_relaxed": "MACD柱不弱",
    "kdj": "KDJ条件",
}
