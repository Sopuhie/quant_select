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
            "类别": "系统定位",
            "规则": (
                "纯日线规则引擎：无分时、无 L2、无盘口；"
                "信号 T 日收盘确认，T+1 买入，最早 T+2 卖出（A 股 T+1 交割）；"
                "源码 strategy.py / execution.py / config.py"
            ),
        },
        {
            "类别": "扫描流水线（顺序）",
            "规则": (
                "①大盘双过滤 → ②板块/前缀池 → ③流动性两层 → ④历史 K 线根数 → "
                "⑤停牌/涨停剔除 → ⑥近涨停/5日动量 → ⑦趋势+收阳 → ⑧实体比 → "
                "⑨量价背离 → ⑩共振≥3/4 → ⑪rule_score 排序取 Top N"
            ),
        },
        {
            "类别": "大盘环境（双过滤）",
            "规则": (
                f"锚定 {index_label}（{SHORT_MARKET_INDEX_CODE}）："
                f"环境分 = 收盘>MA20 ? 60 : 20（需 ≥{SHORT_MIN_MARKET_SCORE}）；"
                f"且 {SHORT_MARKET_MOMENTUM_DAYS} 日指数动量 "
                f"> {SHORT_MARKET_MOMENTUM_MIN * 100:.1f}%（(末收/前N收)-1）。"
                "任一不满足 → 当日空仓，不扫个股。"
            ),
        },
        {
            "类别": "股票池与板块",
            "规则": (
                f"ST：{_on_off(SHORT_EXCLUDE_ST)}；北交所等前缀：{_on_off(SHORT_EXCLUDE_BJ)}；"
                "信号日停牌、涨停封板剔除；"
                "创业板 300/301、科创板 688 默认不扫（界面可勾选纳入）"
            ),
        },
        {
            "类别": "流动性（两层）",
            "规则": (
                f"第一层：成交额 ≥ {amt_wan:.0f} 万 **或** 换手 ≥ {SHORT_MIN_TURNOVER:.1f}%（满足其一）。"
                f"第二层（库内有 turnover_rate）：黄金区间 "
                f"{SHORT_TURNOVER_GOLDEN_MIN:.1f}% ~ {SHORT_TURNOVER_GOLDEN_MAX:.1f}%"
            ),
        },
        {
            "类别": "历史 K 线",
            "规则": f"单股至少 {SHORT_MIN_HISTORY_BARS} 根日线，panel 末根日期必须等于信号日",
        },
        {
            "类别": "趋势（硬性）",
            "规则": (
                f"收盘 > MA{SHORT_MA_FAST} 且 MA{SHORT_MA_FAST} ≥ MA{SHORT_MA_SLOW}；"
                "且收阳：收盘 > 开盘"
            ),
        },
        {
            "类别": "中期动量（硬性）",
            "规则": f"5 日涨幅 = close/close[-5]-1 ≤ {SHORT_MAX_5D_RETURN * 100:.0f}%",
        },
        {
            "类别": "近涨停（硬性）",
            "规则": (
                f"近涨停剔除：{_on_off(SHORT_EXCLUDE_NEAR_LIMIT)}；"
                "主板日涨幅 ≥9.5%；创/科 ≥19.2%；北交所 ≥29.2%"
            ),
        },
        {
            "类别": "K 线实体比（硬性）",
            "规则": (
                f"(收盘−最低)/(最高−最低) ≥ {SHORT_ENTITY_RATIO_MIN:.1f}，"
                "剔除长上影/炸板"
            ),
        },
        {
            "类别": "量价背离（硬性）",
            "规则": "日涨幅 > 3% 且 1 日量比 < 1.0 → 一票否决（缩量大涨）",
        },
        {
            "类别": "指标共振（温和）",
            "规则": (
                "4 项至少 3 项："
                f"①5日量比≥{SHORT_VOL_RATIO_5D_MIN:.2f}；"
                f"②1日量比≥{SHORT_VOL_RATIO_1D_MIN:.2f}；"
                "③DIF>DEA 且 MACD柱≥前柱×0.8；"
                f"④K>D，J∈[{SHORT_KDJ_J_MIN:.0f},{SHORT_KDJ_J_MAX:.0f}]，"
                f"J升≥{SHORT_KDJ_J_SLOPE_MIN:.1f}"
            ),
        },
        {
            "类别": "量比 clip",
            "规则": f"5/1 日量比上限 {SHORT_VOL_RATIO_CLIP_MAX:.0f} 倍，防停牌复牌畸变",
        },
        {
            "类别": "rule_score 权重",
            "规则": (
                f"涨幅 20%（2%~{SHORT_PCT_SCORE_MAX:.1f}% 满分，超出每 1% 扣 25）；"
                "5日量比 30%；MACD柱改善 20%；J斜率 15%；"
                "光头强阳（实体≥98%）+25，其余形态最高 +15；J≥88 扣 30"
            ),
        },
        {
            "类别": "输出",
            "规则": f"按 rule_score 降序取 Top {SHORT_TOP_N}（QUANT_SHORT_TOP_N，1~20）",
        },
        {
            "类别": "T+1 买入",
            "规则": (
                f"限价区间 [{_pct(SHORT_ENTRY_MIN_GAP)}, {_pct(SHORT_ENTRY_MAX_CHASE)}]；"
                f"微高开 ≤{_pct(SHORT_ENTRY_DIP_OPEN_THRESHOLD)} 按开盘价；"
                "否则 (open+low)/2；超限 SKIPPED"
            ),
        },
        {
            "类别": "T+2 止盈/止损/收盘",
            "规则": SHORT_HOLD_PLAN,
        },
        {
            "类别": "T+2 双阶梯止盈",
            "规则": (
                f"最高涨幅≥{_pct(SHORT_T1_TAKE_PROFIT_TIER1_PCT)}：(高+收)/2；"
                f"≥{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)} 且未达 tier1："
                f"锁定 +{_pct(SHORT_T1_TAKE_PROFIT_TIER2_LOCK)}"
            ),
        },
        {
            "类别": "T+2 非对称止损",
            "规则": (
                f"盘中最高<{SHORT_T1_TAKE_PROFIT_TIER2_PCT * 100:.0f}%：收盘破 -{_pct(SHORT_MEDIOCRE_STOP_RATIO)}；"
                f"曾≥{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}：收盘破 -{_pct(SHORT_CLOSE_STOP_RATIO)}"
            ),
        },
        {
            "类别": "T+2 骑乘分支",
            "规则": (
                f"SELL_OFFSET={SHORT_SELL_OFFSET}；"
                f"T+1收≥{_pct(SHORT_T1_STRONG_CLOSE_PCT)} 且 T+2收>T+1收 → 趋势骑乘；"
                "否则 T+2 收盘平"
            ),
        },
        {
            "类别": "实盘建议文案",
            "规则": (
                "J≥88：极小仓，T+2 严格执行止盈/止损；"
                "日涨幅≥6%：警惕次日高开不及预期；"
                "否则：T+1 开盘买，T+2 双阶梯止盈/止损，强势骑乘"
            ),
        },
        {
            "类别": "落库与复盘",
            "规则": (
                "short_daily_selections + short_order_tracker + short_today.json；"
                "detail_json.checks 对应入选明细 ✓/✗；"
                "score_breakdown 含 pct/vr5/body/j 惩罚分项"
            ),
        },
    ]


def format_short_term_rules_markdown() -> str:
    """Streamlit「短线选股规则说明（复盘对照）」展开区 Markdown。"""
    amt_wan = SHORT_MIN_AMOUNT / 1e4
    index_label = "中证1000" if SHORT_MARKET_INDEX_CODE == "000852" else SHORT_MARKET_INDEX_CODE
    sell_day = f"T+{SHORT_SELL_OFFSET}"
    lo_pct = (1 + SHORT_ENTRY_MIN_GAP) * 100
    hi_pct = (1 + SHORT_ENTRY_MAX_CHASE) * 100

    return "\n".join(
        [
            "### 一、系统定位与数据边界",
            "- **策略类型**：纯日线规则选股 + 纯日线模拟执行（**无分时、无 L2、无盘口**）。",
            "- **数据源**：本地 SQLite `stock_daily_kline`（个股 OHLCV、可选 `turnover_rate`）"
            f" + `index_daily`（锚定 {index_label} `{SHORT_MARKET_INDEX_CODE}`）。",
            "- **核心模块**：",
            "  - 选股扫描：`strategy.py` → `ShortTermRuleStrategy.scan`",
            "  - 模拟买卖：`execution.py` → `evaluate_short_trade`",
            "  - 阈值配置：`config.py`（环境变量可覆盖，见第十一节）",
            "- **输出**：`short_daily_selections`（选股明细）、`short_order_tracker`（模拟订单）、"
            "`short_today.json`（当日推送快照）。",
            "- **回测**：`python scripts/short_term_backtest.py`，逻辑与实盘扫描一致，**不写库**。",
            "",
            "### 二、完整时间轴（信号 → 买入 → 卖出）",
            "```",
            "T 日收盘     → 全市场 scan，确认信号（不成交）",
            "T+1 开盘     → 非对称限价买入（A 股 T+1 交割：当日买入不可卖）",
            "T+2 及以后   → 最早卖出日；按止盈/止损/收盘链评估",
            "```",
            "",
            "#### 2.1 T 日（信号日）你在做什么",
            "- 收盘后（或指定历史信号日）对全市场逐股计算指标与规则得分。",
            "- **只有收盘确认的信号才会进入 Top N**；信号日本身不模拟成交。",
            "- 若大盘环境闸未通过，当日输出 **空表**（不是降级选股）。",
            "",
            "#### 2.2 T+1（买入日）非对称入场",
            "| 场景 | 条件（相对信号收盘价） | 模拟买入价 | 状态 |",
            "|------|------------------------|------------|------|",
            f"| 低开过多 | 开盘 < 信号收 × {lo_pct:.0f}% | — | `t1_open_gap_down_rejected` |",
            f"| 追高超限 | 开盘 > 信号收 × {hi_pct:.1f}% | — | `t1_open_chase_rejected` |",
            f"| 微幅高开 | 开盘 ≤ 信号收 × {(1 + SHORT_ENTRY_DIP_OPEN_THRESHOLD) * 100:.1f}% | **T+1 开盘价** | 成交 |",
            f"| 人气高开 | 介于微高开与追高上限之间 | **(open + T+1 low) / 2** | 成交（模拟下探低吸） |",
            "",
            "**数值示例**：信号收盘 10.00 元 → 允许开盘区间 **[9.80, 10.55]** 元。",
            "若 T+1 开 10.30、低 10.00，则买入价 = (10.30 + 10.00) / 2 = **10.15** 元。",
            "",
            "#### 2.3 T+2（最早卖出日）评估链",
            "T+1 **只买不卖**；无论 T+1 强弱，均在 T+2 走**同一套**评估顺序（先触发先平）：",
            "",
            "1. **双阶梯动态止盈**（看 T+2 盘中最高价相对买入价涨幅）",
            f"   - 最高涨幅 ≥ **{_pct(SHORT_T1_TAKE_PROFIT_TIER1_PCT)}** → 平仓价 = (high + close) / 2",
            f"   - 最高涨幅 ≥ **{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}** 且未达 tier1"
            f" → 平仓价 = 买入价 × (1 + {_pct(SHORT_T1_TAKE_PROFIT_TIER2_LOCK)})",
            "2. **非对称收盘止损**（看 T+2 收盘价）",
            f"   - 盘中最高 **<{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}**（平庸股）"
            f" → 收盘破 **-{_pct(SHORT_MEDIOCRE_STOP_RATIO)}** 止损",
            f"   - 盘中曾 ≥ **{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}**（显强势）"
            f" → 收盘破 **-{_pct(SHORT_CLOSE_STOP_RATIO)}** 止损",
            "3. **默认 T+2 收盘**",
            f"   - `QUANT_SHORT_SELL_OFFSET=2` 且 T+1 收盘涨幅 **≥{_pct(SHORT_T1_STRONG_CLOSE_PCT)}**：",
            "     - T+2 收盘 **>** T+1 收盘 → `t2_trend_ride_exit`（趋势骑乘）",
            "     - 否则 → `t2_close_exit`",
            "   - `QUANT_SHORT_SELL_OFFSET=1`：未触发 1/2 时一律 `t2_close_exit`",
            "",
            "**T+2 演算示例**（买入价 10.00）：",
            f"- T+2 高 10.70、收 10.20 → tier1 触发，卖价 (10.70+10.20)/2 = **10.45**",
            f"- T+2 高 10.52、收 9.50 → tier2 触发（≥5%），卖价 **10.30**（+3% 锁定）",
            f"- T+2 高 10.20、收 9.55 → 平庸股 -4% 止损，卖价 **9.55**",
            "",
            f"完整执行文案：`{SHORT_HOLD_PLAN}`",
            "",
            "### 三、大盘环境闸（扫描前，一票否决）",
            f"锚定 **{index_label}**（`{SHORT_MARKET_INDEX_CODE}`），**两项同时满足**才进入个股扫描：",
            "",
            "| 条件 | 计算方式 | 默认阈值 | 不满足时 |",
            "|------|----------|----------|----------|",
            f"| **环境分** | 指数收盘 > MA20 → **60 分**；否则 **20 分** | ≥ **{SHORT_MIN_MARKET_SCORE}** | 当日空仓 |",
            f"| **N 日动量** | (信号日收盘 / {SHORT_MARKET_MOMENTUM_DAYS} 交易日前收盘) − 1 | > **{SHORT_MARKET_MOMENTUM_MIN * 100:.1f}%** | 当日空仓 |",
            "",
            "- 指数数据不足 20 根时环境分保守偏低，需先同步 `index_daily`。",
            "- 设计意图：过滤指数贴 MA20 下方、且短期动能未改善的「系统性弱环境」。",
            "",
            "### 四、个股扫描流水线（按代码执行顺序）",
            "对信号日全市场（或 `--max-scan-stocks` 截断后的子集）逐股执行，**任一步失败即 continue**：",
            "",
            "| 步骤 | 检查项 | 说明 |",
            "|------|--------|------|",
            "| 1 | 板块前缀 | 默认剔除 300/301/688；北交所等前缀可配置剔除 |",
            "| 2 | ST 名称 | 名称含 ST / *ST 剔除 |",
            f"| 3 | 流动性第一层 | 成交额 ≥ **{amt_wan:.0f} 万** **或** 换手 ≥ **{SHORT_MIN_TURNOVER:.1f}%** |",
            f"| 4 | 换手率黄金区间 | 库内有 `turnover_rate` 时："
            f"**{SHORT_TURNOVER_GOLDEN_MIN:.1f}% ~ {SHORT_TURNOVER_GOLDEN_MAX:.1f}%** |",
            f"| 5 | 历史 K 线 | ≥ **{SHORT_MIN_HISTORY_BARS}** 根，且末根日期 = 信号日 |",
            "| 6 | 停牌 | 信号日无量/停牌 bar 剔除 |",
            "| 7 | 涨停封板 | 信号日涨停（按板块涨跌幅规则）剔除 |",
            f"| 8 | 近涨停 | 日涨幅 ≥ 板块阈值（主板 9.5% / 创科 19.2% / 北交 29.2%） |",
            f"| 9 | 5 日涨幅上限 | ≤ **{SHORT_MAX_5D_RETURN * 100:.0f}%**，防过度拉升 |",
            f"| 10 | 趋势均线 | 收盘 > MA{SHORT_MA_FAST} 且 MA{SHORT_MA_FAST} ≥ MA{SHORT_MA_SLOW} |",
            "| 11 | 收阳 | 收盘 > 开盘 |",
            f"| 12 | K 线实体比 | (收−低)/(高−低) ≥ **{SHORT_ENTITY_RATIO_MIN:.1f}** |",
            "| 13 | 量价背离 | 涨幅 > 3% 且 1 日量比 < 1.0 → 否决 |",
            "| 14 | 指标共振 | 4 子项中至少 **3** 项通过（见第五节） |",
            "| 15 | rule_score | 加权打分后降序，取 Top N |",
            "",
            "### 五、指标共振（4 项至少 3 项，温和门槛）",
            "相比「4 项全过」，降低纯日线下频繁空仓；复盘表「入选明细」逐项显示 ✓/✗。",
            "",
            "| 子项 | checks 键 | 通过条件 |",
            "|------|-----------|----------|",
            f"| 5 日量比 | `resonance_vr5` | 当日量 / 5 日均量 ≥ **{SHORT_VOL_RATIO_5D_MIN:.2f}**"
            f"（clip ≤ {SHORT_VOL_RATIO_CLIP_MAX:.0f}） |",
            f"| 1 日量比 | `resonance_vr1` | 当日量 / 昨日量 ≥ **{SHORT_VOL_RATIO_1D_MIN:.2f}**（同上 clip） |",
            "| MACD | `resonance_macd` | DIF > DEA **且** 当日 MACD 柱 ≥ 前日柱 × **0.8** |",
            f"| KDJ | `resonance_kdj` | K > D；J ∈ **[{SHORT_KDJ_J_MIN:.0f}, {SHORT_KDJ_J_MAX:.0f}]**；"
            f"J 较前日升 ≥ **{SHORT_KDJ_J_SLOPE_MIN:.1f}** |",
            "",
            "**MACD 计算**：EMA12/EMA26 → DIF → DEA(9) → 柱 = (DIF−DEA)×2。",
            "**KDJ 计算**：9 日 RSV → K/D 平滑 → J = 3K − 2D。",
            "",
            "### 六、rule_score 排序公式（0~100+，截断 ≥0）",
            "通过全部硬性门槛后计算，用于 **Top N 排序**（非硬性阈值）。",
            "",
            "| 维度 | 权重 | 子分计算 |",
            "|------|------|----------|",
            f"| 当日涨幅 | **20%** | 2%~{SHORT_PCT_SCORE_MAX:.1f}% → 100 分；"
            f">{SHORT_PCT_SCORE_MAX:.1f}% 每多 1pp 扣 25；<2% 按 pct×15 线性 |",
            f"| 5 日量比 | **30%** | min(100, vr5 / {SHORT_VOL_RATIO_5D_MIN:.2f} × 60) |",
            "| MACD 柱改善 | **20%** | clamp(50 + (柱今−柱昨)×200, 0, 100) |",
            "| J 斜率 | **15%** | clamp(50 + (J今−J昨)×5, 0, 100) |",
            "| 形态溢价 | **+分** | 实体占比 (收−开)/(高−低)：≥98% → **+25**；否则最高 +15 |",
            "| 超买惩罚 | **−30** | **J ≥ 88** 时触发 |",
            "",
            f"**最终**：`rule_score = max(0, 加权和 + 形态溢价 − J惩罚)`，按降序取 Top **{SHORT_TOP_N}**。",
            "",
            "### 七、实盘建议文案（展示列，非硬性规则）",
            "| 条件 | 文案含义 |",
            "|------|----------|",
            "| J ≥ 88 | 超买区，仅极小仓；T+2 严格执行止盈/止损链 |",
            "| 日涨幅 ≥ 6% | 信号日已大阳线，警惕 T+1 高开不及预期 |",
            "| 其余 | T+1 开盘买，T+2 双阶梯止盈/止损，T+1 强势则 T+2 骑乘 |",
            "",
            "### 八、exit_reason 与订单状态",
            "| 代码 | 含义 |",
            "|------|------|",
            "| `t1_open_chase_rejected` | T+1 高开超限，放弃买入 |",
            "| `t1_open_gap_down_rejected` | T+1 低开超限，放弃买入 |",
            "| `await_t1_kline` / `await_t2_kline` | 等待 T+1 / T+2 K 线补齐 |",
            f"| `t2_intraday_take_profit_tier1` | T+2 冲高 ≥{_pct(SHORT_T1_TAKE_PROFIT_TIER1_PCT)} |",
            f"| `t2_intraday_take_profit_tier2` | T+2 冲高 ≥{_pct(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}，锁定 +{_pct(SHORT_T1_TAKE_PROFIT_TIER2_LOCK)} |",
            f"| `t2_asymmetric_stop_exit` | 平庸股 T+2 收盘 −{_pct(SHORT_MEDIOCRE_STOP_RATIO)} |",
            f"| `t2_close_below_stop_limit` | 曾显强势，T+2 收盘 −{_pct(SHORT_CLOSE_STOP_RATIO)} |",
            "| `t2_trend_ride_exit` | T+1 强势且 T+2 续强，T+2 收盘卖 |",
            "| `t2_close_exit` | 默认 T+2 收盘平仓 |",
            "",
            "**订单状态**：`CLOSED`（已模拟平仓）/ `HOLDING`（T+1 已买、缺 T+2 数据）/ `SKIPPED`（未成交）。",
            "",
            "### 九、复盘对照（界面字段 ↔ 代码）",
            "- **规则一览表**：本节 + 上表，与 `config.py` 阈值自动同步，改环境变量后刷新页面。",
            "- **入选明细 checks**（`detail_json.checks`）：",
            "  - `trend_ma` 趋势均线 · `bullish_candle` 收阳 · `momentum_5d_cap` 5日涨幅上限",
            "  - `near_limit_ok` 未近涨停 · `turnover_golden_ok` 换手黄金 · `volume_price_ok` 量价未背离",
            "  - `resonance_pass` 共振≥3 · `resonance_vr5/vr1/macd/kdj` 四项子明细",
            "- **score_breakdown**：`pct_score` / `vr5_score` / `candle_body_bonus` / `j_overbought_penalty`。",
            "- **订单表** `short_order_tracker`：`buy_price`、`sell_price`、`pnl_ratio`、"
            "`stop_loss_triggered`、`exit_reason`、`hold_days`。",
            f"- **持有口径**：文案 **{SHORT_HOLDING_DAYS}** 个交易日（T+1 买 → 最早 T+2 卖）；"
            f"`QUANT_SHORT_SELL_OFFSET={SHORT_SELL_OFFSET}` 控制 T+1 强势骑乘分支（{sell_day} 评估）。",
            "",
            "### 十、主要环境变量",
            "| 变量 | 含义 | 默认 |",
            "|------|------|------|",
            f"| `QUANT_SHORT_TOP_N` | 每日输出条数 | {SHORT_TOP_N} |",
            f"| `QUANT_SHORT_MIN_MARKET_SCORE` | 大盘环境分下限 | {SHORT_MIN_MARKET_SCORE} |",
            f"| `QUANT_SHORT_MARKET_INDEX` | 锚定指数代码 | {SHORT_MARKET_INDEX_CODE} |",
            f"| `QUANT_SHORT_MKT_MOM_DAYS` | 指数动量回看天数 | {SHORT_MARKET_MOMENTUM_DAYS} |",
            f"| `QUANT_SHORT_MKT_MOM_MIN` | 指数动量下限（小数） | {SHORT_MARKET_MOMENTUM_MIN} |",
            f"| `QUANT_SHORT_SELL_OFFSET` | 2=强势骑乘分支，1=全走 T+2 收盘 | {SHORT_SELL_OFFSET} |",
            f"| `QUANT_SHORT_CLOSE_STOP` | 强势股收盘止损 | {SHORT_CLOSE_STOP_RATIO} |",
            f"| `QUANT_SHORT_MEDIOCRE_STOP` | 平庸股收盘止损 | {SHORT_MEDIOCRE_STOP_RATIO} |",
            f"| `QUANT_SHORT_T1_TP_T1/T2/T2_LOCK` | 双阶梯止盈 6%/5%/+3% | "
            f"{SHORT_T1_TAKE_PROFIT_TIER1_PCT}/{SHORT_T1_TAKE_PROFIT_TIER2_PCT}/{SHORT_T1_TAKE_PROFIT_TIER2_LOCK} |",
            f"| `QUANT_SHORT_T1_STRONG` | T+1 强势骑乘阈值 | {SHORT_T1_STRONG_CLOSE_PCT} |",
            f"| `QUANT_SHORT_ENTRY_MAX_CHASE` | T+1 最高追高 | {SHORT_ENTRY_MAX_CHASE} |",
            f"| `QUANT_SHORT_ENTRY_MIN_GAP` | T+1 最大低开 | {SHORT_ENTRY_MIN_GAP} |",
            f"| `QUANT_SHORT_ENTRY_DIP_OPEN` | 微高开按开盘价阈值 | {SHORT_ENTRY_DIP_OPEN_THRESHOLD} |",
            f"| `QUANT_SHORT_ENTITY_RATIO` | K 线实体比下限 | {SHORT_ENTITY_RATIO_MIN} |",
            f"| `QUANT_SHORT_MAX_5D_RET` | 5 日涨幅上限 | {SHORT_MAX_5D_RETURN} |",
            f"| `QUANT_SHORT_VR5_MIN` / `VR1_MIN` | 共振量比阈值 | "
            f"{SHORT_VOL_RATIO_5D_MIN} / {SHORT_VOL_RATIO_1D_MIN} |",
            f"| `QUANT_SHORT_VOL_RATIO_CLIP` | 量比 clip 上限 | {SHORT_VOL_RATIO_CLIP_MAX} |",
            f"| `QUANT_SHORT_TURNOVER_MIN/MAX` | 换手率黄金区间 | "
            f"{SHORT_TURNOVER_GOLDEN_MIN} / {SHORT_TURNOVER_GOLDEN_MAX} |",
            f"| `QUANT_SHORT_MIN_BARS` | 最少历史 K 线 | {SHORT_MIN_HISTORY_BARS} |",
            f"| `QUANT_SHORT_MIN_AMOUNT` | 最低成交额（元） | {int(SHORT_MIN_AMOUNT)} |",
            f"| `QUANT_SHORT_MIN_TURNOVER` | 最低换手率（%） | {SHORT_MIN_TURNOVER} |",
            f"| `QUANT_SHORT_PCT_SCORE_MAX` | 涨幅打分满分上沿（%） | {SHORT_PCT_SCORE_MAX} |",
            f"| `QUANT_SHORT_KDJ_J_MIN/MAX/SLOPE` | KDJ 共振区间与斜率 | "
            f"{SHORT_KDJ_J_MIN}/{SHORT_KDJ_J_MAX}/{SHORT_KDJ_J_SLOPE_MIN} |",
            f"| `QUANT_SHORT_EXCLUDE_ST/BJ/NEAR_LIMIT` | ST/北交/近涨停剔除 | "
            f"{_on_off(SHORT_EXCLUDE_ST)}/{_on_off(SHORT_EXCLUDE_BJ)}/{_on_off(SHORT_EXCLUDE_NEAR_LIMIT)} |",
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
