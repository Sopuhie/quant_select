# -*- coding: utf-8 -*-
"""短线规则选股说明文案（与 strategy / execution / config 一致，供 Streamlit 复盘展示）。"""
from __future__ import annotations

from .config import (
    SHORT_CLOSE_STOP_RATIO,
    SHORT_ENTRY_DIP_OPEN_THRESHOLD,
    SHORT_ENTRY_MAX_CHASE,
    SHORT_ENTRY_MIN_GAP,
    SHORT_ENTITY_RATIO_MIN,
    SHORT_HOLD_PLAN,
    SHORT_HOLDING_DAYS,
    SHORT_KDJ_J_MAX,
    SHORT_KDJ_J_MIN,
    SHORT_KDJ_J_SLOPE_MIN,
    SHORT_MA_FAST,
    SHORT_MA_SLOW,
    SHORT_MARKET_INDEX_CODE,
    SHORT_MARKET_MOMENTUM_DAYS,
    SHORT_MARKET_MOMENTUM_MIN,
    SHORT_MAX_5D_RETURN,
    SHORT_MEDIOCRE_STOP_RATIO,
    SHORT_MIN_AMOUNT,
    SHORT_MIN_HISTORY_BARS,
    SHORT_MIN_MARKET_SCORE,
    SHORT_MIN_TURNOVER,
    SHORT_PCT_SCORE_MAX,
    SHORT_SELL_OFFSET,
    SHORT_T1_STRONG_CLOSE_PCT,
    SHORT_T1_TAKE_PROFIT_TIER1_PCT,
    SHORT_T1_TAKE_PROFIT_TIER2_LOCK,
    SHORT_T1_TAKE_PROFIT_TIER2_PCT,
    SHORT_TOP_N,
    SHORT_TURNOVER_GOLDEN_MAX,
    SHORT_TURNOVER_GOLDEN_MIN,
    SHORT_VOL_RATIO_1D_MIN,
    SHORT_VOL_RATIO_5D_MIN,
    SHORT_VOL_RATIO_CLIP_MAX,
    SHORT_EXCLUDE_BJ,
    SHORT_EXCLUDE_NEAR_LIMIT,
    SHORT_EXCLUDE_ST,
)


def _on_off(flag: bool) -> str:
    return "开启" if flag else "关闭"


def _pct(v: float) -> str:
    return f"{float(v) * 100:.1f}%"


def short_term_rules_sections() -> list[dict[str, str]]:
    """结构化规则条目（规则一览表 / 复盘对照）。"""
    amt_wan = SHORT_MIN_AMOUNT / 1e4
    index_label = "中证1000" if SHORT_MARKET_INDEX_CODE == "000852" else SHORT_MARKET_INDEX_CODE
    return [
        {
            "类别": "数据与模块",
            "规则": (
                "仅使用本地日线 K 线（open/high/low/close/volume/turnover_rate）；"
                "选股见 strategy.py::scan，模拟执行见 execution.py::evaluate_short_trade；"
                "落库 short_daily_selections + short_order_tracker + short_today.json"
            ),
        },
        {
            "类别": "信号与买入",
            "规则": (
                "T 日收盘确认信号；T+1 非对称限价入场。"
                f"开盘相对信号收盘 ∈ [{_pct(SHORT_ENTRY_MIN_GAP)}, {_pct(SHORT_ENTRY_MAX_CHASE)}]，"
                f"否则 SKIPPED。"
                f"微高开 ≤{_pct(SHORT_ENTRY_DIP_OPEN_THRESHOLD)} 按开盘价；"
                f"更高开按 (open+T+1 low)/2 模拟分时低吸。"
            ),
        },
        {
            "类别": "执行计划（文案）",
            "规则": SHORT_HOLD_PLAN,
        },
        {
            "类别": "T+2 双阶梯止盈",
            "规则": (
                f"A 股 T+1 交割：T+1 只买不卖，最早 T+2 评估。"
                f"T+2 盘中最高涨幅 ≥{_pct(SHORT_T1_TAKE_PROFIT_TIER1_PCT)}："
                f"平仓价=(high+close)/2，exit=t2_intraday_take_profit_tier1；"
                f"≥{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)} 且未达第一阶梯："
                f"锁定 +{_pct(SHORT_T1_TAKE_PROFIT_TIER2_LOCK)}，"
                f"exit=t2_intraday_take_profit_tier2"
            ),
        },
        {
            "类别": "T+2 非对称收盘止损",
            "规则": (
                f"T+2 盘中最高涨幅 <{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}（平庸股）："
                f"收盘 < 买入价×(1-{_pct(SHORT_MEDIOCRE_STOP_RATIO)}) → t2_asymmetric_stop_exit；"
                f"盘中曾达 ≥{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}："
                f"收盘 < 买入价×(1-{_pct(SHORT_CLOSE_STOP_RATIO)}) → t2_close_below_stop_limit"
            ),
        },
        {
            "类别": "T+2 持仓分支",
            "规则": (
                f"QUANT_SHORT_SELL_OFFSET={SHORT_SELL_OFFSET}（默认 2）。"
                f"T+1 与 T+2 均走同一止盈/止损链（仅禁止 T+1 卖出）；"
                f"T+1 收盘涨幅 ≥{_pct(SHORT_T1_STRONG_CLOSE_PCT)} 且 offset≥2："
                f"T+2 收盘 > T+1 收盘则 t2_trend_ride_exit，否则 t2_close_exit"
            ),
        },
        {
            "类别": "输出数量",
            "规则": (
                f"候选池按 rule_score 降序取 Top {SHORT_TOP_N}；"
                "环境变量 QUANT_SHORT_TOP_N（默认 5，范围 1~20）"
            ),
        },
        {
            "类别": "大盘环境（双过滤）",
            "规则": (
                f"锚定 {index_label}（{SHORT_MARKET_INDEX_CODE}）："
                f"环境分 ≥ {SHORT_MIN_MARKET_SCORE}（收盘>MA20→60 分，否则 20 分）；"
                f"且 {SHORT_MARKET_MOMENTUM_DAYS} 日指数动量 > "
                f"{SHORT_MARKET_MOMENTUM_MIN * 100:.1f}%（默认 >0）。"
                "两项均满足才扫描，否则当日空仓。"
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
                "信号日停牌；涨停日（封板）剔除"
            ),
        },
        {
            "类别": "流动性（两层）",
            "规则": (
                f"第一层：成交额 ≥ {amt_wan:.0f} 万 **或** 换手 ≥ {SHORT_MIN_TURNOVER:.1f}%"
                "（二者均不达标则剔除）。"
                f"第二层（库内有 turnover_rate 时）：黄金区间 "
                f"{SHORT_TURNOVER_GOLDEN_MIN:.1f}% ~ {SHORT_TURNOVER_GOLDEN_MAX:.1f}%，"
                "过滤僵尸股与死亡换手力竭股"
            ),
        },
        {
            "类别": "量价背离（硬性）",
            "规则": (
                "当日涨幅 > 3% 且 1 日量比 < 1.0 → 一票否决（缩量大涨伪阳线）"
            ),
        },
        {
            "类别": "K 线形态（硬性）",
            "规则": (
                f"实体占比 = (收盘−最低)/(最高−最低) ≥ {SHORT_ENTITY_RATIO_MIN:.1f}，"
                "剔除长上影/炸板形态"
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
                "**4 项中至少满足 3 项**："
                f"①5日量比≥{SHORT_VOL_RATIO_5D_MIN:.2f}；"
                f"②1日量比≥{SHORT_VOL_RATIO_1D_MIN:.2f}；"
                "③DIF>DEA 且 MACD柱≥前一日柱×0.8；"
                f"④K>D 且 J∈[{SHORT_KDJ_J_MIN:.0f},{SHORT_KDJ_J_MAX:.0f}] "
                f"且 J 较前日升≥{SHORT_KDJ_J_SLOPE_MIN:.1f}"
            ),
        },
        {
            "类别": "量比防畸变",
            "规则": (
                f"5日/1日量比计算后上限 clip 为 {SHORT_VOL_RATIO_CLIP_MAX:.0f} 倍"
                "（QUANT_SHORT_VOL_RATIO_CLIP），防停牌复牌畸变"
            ),
        },
        {
            "类别": "规则得分 rule_score",
            "规则": (
                "加权求和后排序（截断≥0）："
                f"当日涨幅非线性 20%（2%~{SHORT_PCT_SCORE_MAX:.1f}% 满分，"
                f">{SHORT_PCT_SCORE_MAX:.1f}% 每多 1% 扣 25 分）；"
                "5日量比 30%；MACD柱改善 20%；J斜率 15%；"
                "光头强阳（实体占比≥98%）额外 +25 分，其余形态最高 +15 分；"
                "J≥88 额外扣 30 分"
            ),
        },
        {
            "类别": "实盘建议文案",
            "规则": (
                "J≥88：极小仓博弈，严格执行 T+2 离场；"
                "当日涨幅≥6%：警惕次日高开不及预期；"
                "否则：T+1 开盘买入，T+2 双阶梯止盈/止损，强势 T+2 收盘骑乘"
            ),
        },
        {
            "类别": "落库与复盘字段",
            "规则": (
                "short_daily_selections：trade_date、stock_code、rule_score、checks 等；"
                "short_order_tracker：buy_price、sell_price、pnl_ratio、exit_reason、"
                "stop_loss_triggered；detail_json 含 turnover_pct、volume_price_ok、"
                "turnover_golden_ok、score_breakdown 等"
            ),
        },
    ]


def format_short_term_rules_markdown() -> str:
    """Streamlit「短线选股规则说明（复盘对照）」展开区 Markdown。"""
    amt_wan = SHORT_MIN_AMOUNT / 1e4
    index_label = "中证1000" if SHORT_MARKET_INDEX_CODE == "000852" else SHORT_MARKET_INDEX_CODE
    sell_day = f"T+{SHORT_SELL_OFFSET}"

    return "\n".join(
        [
            "### 一、系统定位",
            "- **数据**：仅本地日线 `stock_daily_kline` + `index_daily`（无分时、无 L2）。",
            "- **流程**：T 日 `strategy.scan` 选股 → T+1 `execution.evaluate_short_trade` "
            "模拟买卖 → 写入 `short_daily_selections` / `short_order_tracker` / `short_today.json`。",
            "- **回测**：`python scripts/short_term_backtest.py`（与实盘扫描逻辑一致，不写库）。",
            "",
            "### 二、时间轴与买入（T+1 非对称入场）",
            "| 时点 | 动作 |",
            "|------|------|",
            "| **T 日收盘** | 全市场扫描，确认信号（不成交） |",
            f"| **T+1 开盘** | 限价入场区间：相对信号收盘 "
            f"**[{_pct(SHORT_ENTRY_MIN_GAP)}, {_pct(SHORT_ENTRY_MAX_CHASE)}]** |",
            f"| 微高开 ≤{_pct(SHORT_ENTRY_DIP_OPEN_THRESHOLD)} | 买入价 = **T+1 开盘价** |",
            f"| 高开 {_pct(SHORT_ENTRY_DIP_OPEN_THRESHOLD)}~{_pct(SHORT_ENTRY_MAX_CHASE)} | "
            "买入价 = **(open + T+1 low) / 2**（模拟下探低吸） |",
            f"| 高开 >{_pct(SHORT_ENTRY_MAX_CHASE)} 或低开 <{_pct(SHORT_ENTRY_MIN_GAP)} | "
            "**SKIPPED**（不成交） |",
            "",
            "### 三、T+2 平仓评估顺序（A 股 T+1 交割，纯日线模拟）",
            "T+1 仅完成买入；**最早 T+2** 按以下**优先级**依次触发（先触发先平）：",
            "",
            "#### 3.1 双阶梯动态止盈（T+2 日）",
            "| 条件 | 平仓价 | exit_reason |",
            "|------|--------|-------------|",
            f"| 盘中最高涨幅 ≥ **{_pct(SHORT_T1_TAKE_PROFIT_TIER1_PCT)}** | (high + close) / 2 | "
            "`t2_intraday_take_profit_tier1` |",
            f"| 盘中最高涨幅 ≥ **{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}** 且未达第一阶梯 | "
            f"买入价 × (1 + {_pct(SHORT_T1_TAKE_PROFIT_TIER2_LOCK)}) | "
            "`t2_intraday_take_profit_tier2` |",
            "",
            "#### 3.2 非对称收盘止损（T+2 日）",
            "| 盘中表现 | 止损线 | exit_reason |",
            "|----------|--------|-------------|",
            f"| 最高涨幅 **<{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}**（平庸股） | "
            f"收盘破 **-{_pct(SHORT_MEDIOCRE_STOP_RATIO)}** | `t2_asymmetric_stop_exit` |",
            f"| 最高涨幅 **≥{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}**（曾显强势） | "
            f"收盘破 **-{_pct(SHORT_CLOSE_STOP_RATIO)}** | `t2_close_below_stop_limit` |",
            "",
            "#### 3.3 默认收盘（T+2 日）",
            f"- offset=2 且 T+1 收盘涨幅 **≥{_pct(SHORT_T1_STRONG_CLOSE_PCT)}**：",
            "  - T+2 收盘 > T+1 收盘 → `t2_trend_ride_exit`",
            "  - 否则 → `t2_close_exit`",
            "- offset=1：未触发止盈/止损时 → `t2_close_exit`",
            "",
            f"- 完整文案：`{SHORT_HOLD_PLAN}`",
            "",
            "### 四、大盘环境闸（扫描前）",
            f"- **锚定指数**：{index_label}（`{SHORT_MARKET_INDEX_CODE}`），非沪深300。",
            f"- **条件 1**：环境分 **≥ {SHORT_MIN_MARKET_SCORE}**（指数收盘 > MA20 → 60 分，否则 20 分）。",
            f"- **条件 2**：{SHORT_MARKET_MOMENTUM_DAYS} 日指数动量 **> "
            f"{SHORT_MARKET_MOMENTUM_MIN * 100:.1f}%**（默认 >0，过滤贴线弱动能日）。",
            "- **两项同时满足**才进入个股扫描；否则当日输出为空。",
            "",
            "### 五、股票池、板块与流动性",
            f"- ST 剔除：{_on_off(SHORT_EXCLUDE_ST)}；北交所等：{_on_off(SHORT_EXCLUDE_BJ)}。",
            "- 信号日**停牌**、**涨停封板**日剔除。",
            "- **创业板 300/301、科创板 688**：默认不扫，界面可勾选纳入。",
            f"- **近涨停剔除**（{_on_off(SHORT_EXCLUDE_NEAR_LIMIT)}）：主板 ≥9.5%、创/科 ≥19.2%、北交所 ≥29.2%。",
            f"- **流动性第一层**：成交额 ≥ **{amt_wan:.0f} 万** 或 换手 ≥ **{SHORT_MIN_TURNOVER:.1f}%**（满足其一）。",
            f"- **换手率黄金区间**（库内有 `turnover_rate` 时硬性）："
            f"**{SHORT_TURNOVER_GOLDEN_MIN:.1f}% ~ {SHORT_TURNOVER_GOLDEN_MAX:.1f}%**。",
            f"- **历史 K 线**：≥ **{SHORT_MIN_HISTORY_BARS}** 根，末根为信号日。",
            "",
            "### 六、个股硬性门槛（必须全部满足）",
            "以下任一不满足即 `continue` 剔除，不进入打分：",
            "",
            f"1. **趋势**：收盘 > MA{SHORT_MA_FAST} 且 MA{SHORT_MA_FAST} ≥ MA{SHORT_MA_SLOW}。",
            "2. **收阳**：收盘 > 开盘。",
            f"3. **5 日涨幅** ≤ {SHORT_MAX_5D_RETURN * 100:.0f}%。",
            f"4. **K 线实体比** ≥ {SHORT_ENTITY_RATIO_MIN:.1f}（(收−低)/(高−低)，防长上影）。",
            "5. 未触及板块近涨停阈值。",
            "6. **量价背离否决**：涨幅 > 3% 且 1 日量比 < 1.0（缩量大涨）。",
            "",
            "### 七、温和共振（4 项至少 3 项）",
            "复盘表「入选明细」中逐项显示 ✓/✗：",
            "",
            "| 子项 | 条件 |",
            "|------|------|",
            f"| 5日量比 | ≥ {SHORT_VOL_RATIO_5D_MIN:.2f}（clip 上限 {SHORT_VOL_RATIO_CLIP_MAX:.0f} 倍） |",
            f"| 1日量比 | ≥ {SHORT_VOL_RATIO_1D_MIN:.2f}（同上） |",
            "| MACD | DIF > DEA 且 柱 ≥ 前柱 × 0.8 |",
            f"| KDJ | K > D，J ∈ [{SHORT_KDJ_J_MIN:.0f}, {SHORT_KDJ_J_MAX:.0f}]，"
            f"J 升 ≥ {SHORT_KDJ_J_SLOPE_MIN:.1f} |",
            "",
            "### 八、规则得分 rule_score（排序用）",
            "| 维度 | 权重 | 说明 |",
            "|------|------|------|",
            f"| 当日涨幅 | 20% | 2%~{SHORT_PCT_SCORE_MAX:.1f}% 满分；"
            f">{SHORT_PCT_SCORE_MAX:.1f}% 每多 1% 扣 25 分；<2% 线性缩水 |",
            "| 5日量比 | 30% | 相对阈值 1.25 倍映射，上限 100 分 |",
            "| MACD 柱改善 | 20% | 当日柱 − 前日柱 |",
            "| J 斜率 | 15% | 当日 J − 前日 J |",
            "| 光头强阳溢价 | +分 | 实体占比 ≥98% → **+25**；其余最高 +15 |",
            "| 超买惩罚 | −30 | **J ≥ 88** 时触发 |",
            "",
            f"按 **rule_score 降序**取 Top **{SHORT_TOP_N}** 写入当日推荐。",
            "",
            "### 九、exit_reason 速查",
            "| 代码 | 含义 |",
            "|------|------|",
            "| `t1_open_chase_rejected` | T+1 高开超限，放弃买入 |",
            "| `t1_open_gap_down_rejected` | T+1 低开超限，放弃买入 |",
            "| `await_t2_kline` | T+1 已买，等待 T+2 K 线 |",
            "| `t2_intraday_take_profit_tier1` | T+2 冲高 ≥6% 动态止盈 |",
            "| `t2_intraday_take_profit_tier2` | T+2 冲高 ≥5% 锁定 +3% |",
            "| `t2_asymmetric_stop_exit` | 平庸股 T+2 收盘 -4% 止损 |",
            "| `t2_close_below_stop_limit` | 强势股 T+2 收盘 -6% 止损 |",
            "| `t2_trend_ride_exit` / `t2_close_exit` | T+2 趋势骑乘 / 普通平仓 |",
            "",
            "### 十、复盘对照说明",
            "- **规则一览表**与 `src/short_term/config.py` 阈值同步，改配置后刷新页面即可。",
            "- **入选明细**：`detail_json.checks` 对应趋势、共振、换手、量价等子项。",
            "- **订单表** `short_order_tracker`：`HOLDING` / `CLOSED` / 跳过不入库；"
            "`stop_loss_triggered`、`pnl_ratio`、`exit_reason` 用于复盘统计。",
            f"- 文案口径持有 **{SHORT_HOLDING_DAYS}** 日（T+1 买 → 最早 T+2 卖）；"
            f"`QUANT_SHORT_SELL_OFFSET={SHORT_SELL_OFFSET}` 控制 T+1 强势时是否启用"
            f" T+2 趋势骑乘分支（{sell_day} 收盘评估）。",
            "",
            "### 十一、主要环境变量",
            "| 变量 | 含义 | 默认 |",
            "|------|------|------|",
            f"| `QUANT_SHORT_TOP_N` | 每日输出条数 | {SHORT_TOP_N} |",
            f"| `QUANT_SHORT_MIN_MARKET_SCORE` | 大盘环境分下限 | {SHORT_MIN_MARKET_SCORE} |",
            f"| `QUANT_SHORT_MARKET_INDEX` | 锚定指数代码 | {SHORT_MARKET_INDEX_CODE} |",
            f"| `QUANT_SHORT_SELL_OFFSET` | 2=强势骑乘分支，1=全走 T+2 收盘 | {SHORT_SELL_OFFSET} |",
            f"| `QUANT_SHORT_CLOSE_STOP` | 强势股收盘止损比例 | {SHORT_CLOSE_STOP_RATIO} |",
            f"| `QUANT_SHORT_MEDIOCRE_STOP` | 平庸股收盘止损比例 | {SHORT_MEDIOCRE_STOP_RATIO} |",
            f"| `QUANT_SHORT_ENTRY_MAX_CHASE` | T+1 最高追高比例 | {SHORT_ENTRY_MAX_CHASE} |",
            f"| `QUANT_SHORT_ENTITY_RATIO` | K 线实体比下限 | {SHORT_ENTITY_RATIO_MIN} |",
            f"| `QUANT_SHORT_TURNOVER_MIN/MAX` | 换手率黄金区间 | "
            f"{SHORT_TURNOVER_GOLDEN_MIN}/{SHORT_TURNOVER_GOLDEN_MAX} |",
        ]
    )


# 复盘表「入选明细」列名 ↔ detail_json.checks 键
CHECK_LABELS: dict[str, str] = {
    "trend_ma": "趋势均线",
    "bullish_candle": "当日收阳(收>开)",
    "momentum_5d_cap": "5日涨幅上限",
    "near_limit_ok": "未近涨停",
    "resonance_pass": "共振≥3项",
    "turnover_golden_ok": "换手率黄金区间",
    "volume_price_ok": "量价未背离",
    "resonance_vr5": "5日量比",
    "resonance_vr1": "1日量比",
    "resonance_macd": "MACD共振",
    "resonance_kdj": "KDJ共振",
}
