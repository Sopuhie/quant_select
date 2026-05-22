# -*- coding: utf-8 -*-
"""短线规则选股说明文案（与 strategy / execution / config 一致，供 Streamlit 复盘展示）。"""
from __future__ import annotations

from .config import (
    SHORT_HOLD_PLAN,
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
    SHORT_SELL_OFFSET,
    SHORT_STOP_LOSS_RATIO,
    SHORT_TOP_N,
    SHORT_VOL_RATIO_1D_MIN,
    SHORT_VOL_RATIO_5D_MIN,
    SHORT_VOL_RATIO_CLIP_MAX,
    SHORT_EXCLUDE_BJ,
    SHORT_EXCLUDE_NEAR_LIMIT,
    SHORT_EXCLUDE_ST,
)


def _on_off(flag: bool) -> str:
    return "开启" if flag else "关闭"


def _stop_pct() -> str:
    return f"{SHORT_STOP_LOSS_RATIO * 100:.0f}"


def short_term_rules_sections() -> list[dict[str, str]]:
    """结构化规则条目（规则一览表 / 复盘对照）。"""
    amt_wan = SHORT_MIN_AMOUNT / 1e4
    stop_pct = _stop_pct()
    return [
        {
            "类别": "数据与模块",
            "规则": (
                "仅使用本地日线 K 线（open/high/low/close/volume）；"
                "选股逻辑见 strategy.py，模拟买卖见 execution.py；"
                "落库 short_daily_selections + short_order_tracker"
            ),
        },
        {
            "类别": "执行计划",
            "规则": SHORT_HOLD_PLAN,
        },
        {
            "类别": "T+1 止损（纯日线）",
            "规则": (
                f"T+1 最低价 < 买入价×(1-{SHORT_STOP_LOSS_RATIO}) 视为触及止损区；"
                f"若 T+1 开盘≤止损价（一字跌停/大幅低开），平仓价=T+1 收盘价，"
                f"原因 t1_open_below_stop_limit；"
                f"若开盘正常且盘中跌破，平仓价=买入价×{1 - SHORT_STOP_LOSS_RATIO:.2f}，"
                f"原因 t1_intraday_stop_loss"
            ),
        },
        {
            "类别": "未止损平仓",
            "规则": (
                f"未触发止损时，于 T+{SHORT_SELL_OFFSET} 以当日收盘价平仓"
                f"（QUANT_SHORT_SELL_OFFSET=1 为 T+1 收盘，2 为 T+2 收盘）"
            ),
        },
        {
            "类别": "输出数量",
            "规则": (
                f"按 final_score（规则得分）降序取 Top {SHORT_TOP_N}；"
                "环境变量 QUANT_SHORT_TOP_N（默认 5）"
            ),
        },
        {
            "类别": "大盘环境",
            "规则": (
                f"沪深300 环境分 ≥ {SHORT_MIN_MARKET_SCORE} 才扫描；"
                "指数收盘>MA20 为 60 分，否则 20 分；无指数数据时默认 50 分（保守）"
            ),
        },
        {
            "类别": "板块过滤",
            "规则": (
                "默认剔除创业板（300/301）、科创板（688）；"
                "界面或 CLI 可勾选 --include-300 / --include-688 纳入"
            ),
        },
        {
            "类别": "股票池剔除",
            "规则": (
                f"ST：{_on_off(SHORT_EXCLUDE_ST)}；北交所等：{_on_off(SHORT_EXCLUDE_BJ)}；"
                "信号日停牌；涨停日（一字/封板）剔除"
            ),
        },
        {
            "类别": "流动性",
            "规则": (
                f"信号日成交额 ≥ {amt_wan:.0f} 万 **或** 换手率 ≥ {SHORT_MIN_TURNOVER:.1f}%"
                "（二者均不达标则剔除）"
            ),
        },
        {
            "类别": "历史 K 线",
            "规则": f"至少 {SHORT_MIN_HISTORY_BARS} 根日线，最后一根必须为信号日",
        },
        {
            "类别": "趋势（硬性，必过）",
            "规则": (
                f"收盘价 > MA{SHORT_MA_FAST} 且 MA{SHORT_MA_FAST} ≥ MA{SHORT_MA_SLOW}；"
                "且当日收阳：收盘 > 开盘"
            ),
        },
        {
            "类别": "板块近涨停（硬性）",
            "规则": (
                f"近涨停剔除：{_on_off(SHORT_EXCLUDE_NEAR_LIMIT)}；"
                "主板当日涨幅 ≥9.5% 剔除；创/科 ≥19.2%；北交所 ≥29.2%（按代码前缀）"
            ),
        },
        {
            "类别": "中期动量（硬性）",
            "规则": f"近 5 日涨幅 ≤ {SHORT_MAX_5D_RETURN * 100:.0f}%（防过度拉升）",
        },
        {
            "类别": "指标共振（温和）",
            "规则": (
                "**4 项中至少满足 3 项**（否则剔除，避免纯日线条件过苛空仓）："
                f"①5日量比≥{SHORT_VOL_RATIO_5D_MIN:.2f}；"
                f"②1日量比≥{SHORT_VOL_RATIO_1D_MIN:.2f}；"
                "③DIF>DEA 且 MACD柱≥前一日柱×0.8；"
                f"④K>D 且 J∈[{SHORT_KDJ_J_MIN:.0f},{SHORT_KDJ_J_MAX:.0f}] 且 J 较前日升≥{SHORT_KDJ_J_SLOPE_MIN:.1f}"
            ),
        },
        {
            "类别": "量比防畸变",
            "规则": (
                f"计算 5日/1日量比后上限 clip 为 {SHORT_VOL_RATIO_CLIP_MAX:.0f} 倍"
                f"（QUANT_SHORT_VOL_RATIO_CLIP），防停牌复牌/僵尸股污染打分"
            ),
        },
        {
            "类别": "规则得分 final_score",
            "规则": (
                "加权求和后排序："
                "当日涨幅非线性 20%（2.5%~5.5% 满分，>5.5% 每多 1% 扣 15 分，<2.5% 线性缩水）；"
                "5日量比 25%；MACD柱改善 25%；J斜率 15%；"
                "J≥88 额外扣 30 分（超买惩罚）"
            ),
        },
        {
            "类别": "实盘建议文案",
            "规则": (
                "J≥88：极小仓博弈；当日涨幅≥6%：警惕次日高开不及预期；"
                "否则：T 日收盘买入，T+1 评估止损或按配置日收盘平仓"
            ),
        },
        {
            "类别": "落库字段（选股表）",
            "规则": (
                "trade_date、stock_code、close_price、final_score、"
                "pct_change、volume_ratio_5d、macd_bar_improve、j_slope、"
                "is_executed；T1买T2卖涨跌幅=(T2收盘−T+1开盘或收盘)/买入价；"
                "detail_json 含各校验项 checks"
            ),
        },
    ]


def format_short_term_rules_markdown() -> str:
    """Streamlit「短线选股规则说明（复盘对照）」展开区 Markdown。"""
    amt_wan = SHORT_MIN_AMOUNT / 1e4
    stop_pct = _stop_pct()
    sell_day = f"T+{SHORT_SELL_OFFSET}"

    return "\n".join(
        [
            "### 一、系统定位",
            "- **数据**：仅本地日线 `stock_daily_kline`（无分时、无盘口）。",
            "- **流程**：`strategy.py` 扫描共振 → `execution.py` 模拟买卖 → 写入 "
            "`short_daily_selections` / `short_order_tracker` / `short_today.json`。",
            "",
            "### 二、执行与平仓（纯日线模拟）",
            f"- **买入**：信号日 **T 日收盘价** 作为买入价（不再 T+1 开盘买）。",
            f"- **硬止损**：T+1 用 **最低价 low** 判断是否跌破买入价×(1-{SHORT_STOP_LOSS_RATIO})（约 **-{stop_pct}%**）。",
            f"  - **开盘≤止损价**（一字跌停/大幅低开）：无法在 -3% 成交 → 平仓价取 **T+1 收盘价**"
            f"（`t1_open_below_stop_limit`）。",
            f"  - **开盘正常、盘中跌破**：平仓价 = 买入价×{1 - SHORT_STOP_LOSS_RATIO:.2f}"
            f"（`t1_intraday_stop_loss`）。",
            f"- **未触发止损**：于 **{sell_day} 收盘价** 平仓（`QUANT_SHORT_SELL_OFFSET`，默认 1=T+1）。",
            f"- 文案：`{SHORT_HOLD_PLAN}`",
            "",
            "### 三、输出与大盘",
            f"- 环境分 **≥ {SHORT_MIN_MARKET_SCORE}** 才扫描（沪深300：价>MA20→60 分，否则 20 分；无指数默认 50）。",
            f"- 按 **final_score** 降序取 Top **{SHORT_TOP_N}**（`QUANT_SHORT_TOP_N`）。",
            "",
            "### 四、股票池与板块",
            f"- ST 剔除：{_on_off(SHORT_EXCLUDE_ST)}；北交所等：{_on_off(SHORT_EXCLUDE_BJ)}；停牌、涨停日剔除。",
            "- **创业板 300/301、科创板 688**：默认不扫描，界面可勾选纳入。",
            f"- **近涨停剔除**（{_on_off(SHORT_EXCLUDE_NEAR_LIMIT)}）：主板 ≥9.5%、创/科 ≥19.2%、北交所 ≥29.2%。",
            f"- **流动性**：成交额 ≥ **{amt_wan:.0f} 万** 或换手 ≥ **{SHORT_MIN_TURNOVER:.1f}%**（满足其一）。",
            f"- **历史 K 线**：≥ **{SHORT_MIN_HISTORY_BARS}** 根，末根为信号日。",
            "",
            "### 五、硬性门槛（必须全部满足）",
            f"1. **趋势**：收盘 > MA{SHORT_MA_FAST} 且 MA{SHORT_MA_FAST} ≥ MA{SHORT_MA_SLOW}。",
            "2. **收阳**：收盘 > 开盘。",
            f"3. **5 日涨幅** ≤ {SHORT_MAX_5D_RETURN * 100:.0f}%。",
            "4. 未触及板块近涨停阈值（见上）。",
            "",
            "### 六、温和共振（4 项至少 3 项）",
            "以下子项在复盘表「入选明细」中逐项显示 ✓/✗：",
            f"| 子项 | 条件 |",
            f"|------|------|",
            f"| 5日量比 | ≥ {SHORT_VOL_RATIO_5D_MIN:.2f}（计算后上限 {SHORT_VOL_RATIO_CLIP_MAX:.0f} 倍） |",
            f"| 1日量比 | ≥ {SHORT_VOL_RATIO_1D_MIN:.2f}（同上截断） |",
            "| MACD | DIF > DEA 且柱 ≥ 前柱 ×0.8 |",
            f"| KDJ | K>D，J∈[{SHORT_KDJ_J_MIN:.0f},{SHORT_KDJ_J_MAX:.0f}]，J升≥{SHORT_KDJ_J_SLOPE_MIN:.1f} |",
            "",
            "### 七、规则得分 final_score（排序用）",
            "| 维度 | 权重 | 说明 |",
            "|------|------|------|",
            "| 当日涨幅 | 20% | 2.5%~5.5% 满分；>5.5% 每多 1% 扣 15 分；<2.5% 按涨幅×15 缩水 |",
            "| 5日量比 | 25% | 相对阈值 1.25 倍线性映射，上限 100 分 |",
            "| MACD柱改善 | 25% | 当日柱 − 前日柱 |",
            "| J 斜率 | 15% | 当日 J − 前日 J |",
            "| 超买惩罚 | — | **J ≥ 88** 时总分 **−30** |",
            "",
            "### 八、复盘对照说明",
            "- 下方**规则一览表**与代码阈值同步（`src/short_term/config.py`）。",
            "- **入选明细**展开区：`detail_json.checks` 对应趋势、共振子项等。",
            "- **订单状态**见 `short_order_tracker`：`HOLDING` / `CLOSED`，"
            "`stop_loss_triggered`、`exit_reason`。",
            f"- 逻辑持有 **{SHORT_HOLDING_DAYS}** 个交易日（文案口径）；实际平仓日以 "
            f"`SHORT_SELL_OFFSET={SHORT_SELL_OFFSET}` 为准。",
        ]
    )


# 复盘表「入选明细」列名 ↔ detail_json.checks 键
CHECK_LABELS: dict[str, str] = {
    "trend_ma": "趋势均线",
    "bullish_candle": "当日收阳(收>开)",
    "momentum_5d_cap": "5日涨幅上限",
    "near_limit_ok": "未近涨停",
    "resonance_pass": "共振≥3项",
    "resonance_vr5": "5日量比",
    "resonance_vr1": "1日量比",
    "resonance_macd": "MACD共振",
    "resonance_kdj": "KDJ共振",
}
