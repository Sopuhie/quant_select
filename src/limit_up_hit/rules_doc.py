# -*- coding: utf-8 -*-
"""打板规则说明（Streamlit 复盘对照）。"""
from __future__ import annotations

from .config import (
    LUH_BUY_REQUIRE_CLOSE_LIMIT,
    LUH_ENTITY_RATIO_MIN,
    LUH_HOLD_PLAN,
    LUH_MIN_AMOUNT,
    LUH_MIN_MARKET_SCORE,
    LUH_MIN_TURNOVER,
    LUH_MARKET_MOMENTUM_MIN,
    LUH_SEAL_CLOSE_HIGH_MIN,
    LUH_T2_STOP_LOSS,
    LUH_T2_TAKE_PROFIT_TIER1,
    LUH_T2_TAKE_PROFIT_TIER2,
    LUH_TOP_N,
    LUH_TURNOVER_GOLDEN_MAX,
    LUH_TURNOVER_GOLDEN_MIN,
)


def format_limit_up_rules_markdown() -> str:
    buy_rule = (
        "T+1 非一字且收盘涨停 → 以理论涨停价买入"
        if LUH_BUY_REQUIRE_CLOSE_LIMIT
        else "T+1 非一字且盘中触及涨停 → 以理论涨停价买入"
    )
    return f"""
### 打板选股规则（纯日线模拟）

**流程**：{LUH_HOLD_PLAN}

#### 大盘环境
- 锚定指数环境分 ≥ {LUH_MIN_MARKET_SCORE}，且 N 日动量 > {LUH_MARKET_MOMENTUM_MIN * 100:.1f}%（打板默认已放宽）

#### 信号日 T（选股）
| 条件 | 说明 |
|------|------|
| 涨停封板 | 收盘价触及板块涨停价 |
| 封板质量 | 收盘/最高 ≥ {LUH_SEAL_CLOSE_HIGH_MIN:.3f}，实体比 ≥ {LUH_ENTITY_RATIO_MIN} |
| 流动性 | 成交额 ≥ {LUH_MIN_AMOUNT / 1e8:.0f} 亿 **或** 换手 ≥ {LUH_MIN_TURNOVER}% |
| 换手区间 | [{LUH_TURNOVER_GOLDEN_MIN}%, {LUH_TURNOVER_GOLDEN_MAX}%] |
| 剔除 | ST、北交所、停牌 |

#### 排序（Top {LUH_TOP_N}）
连板高度 35% + 封板强度 30% + 换手适中 20% + 成交额 15%

#### T+1 打板买入
- {buy_rule}
- T+1 一字涨停 → 跳过（无法买入）

#### T+2 卖出（实盘模拟）
- 止盈：冲高 ≥ {LUH_T2_TAKE_PROFIT_TIER1 * 100:.0f}% / ≥ {LUH_T2_TAKE_PROFIT_TIER2 * 100:.0f}% 双阶梯
- 止损：收盘 < 买入价 × (1 - {LUH_T2_STOP_LOSS * 100:.0f}%)
- 连板续强 → T+2 收盘离场或骑乘

#### 历史回测（优化版，默认）

| 项目 | 规则（纯日线可实现部分） |
|------|--------------------------|
| 大盘闸 | 指数 20 日涨幅 > 0 **且** 全市场涨停家数 > 50 |
| 板块 | 同概念板块涨停数 ≥ 3（需 ``stock_concept_boards`` / ``board_stocks.json``） |
| 买入 | T+1 收盘涨停 + 低开板次数代理（Low/Close≥99.5%）+ 开盘价×(1+0.5%滑点) |
| 止盈 | 收盘涨幅 ≥ 10% / ≥ 6% |
| T+2 | 收盘涨停且换手≥1% → 续骑至 T+3；否则 T+2 收盘卖 |
| 续骑 | 之后一字涨停续持，开板日开盘价卖 |

> 封板时间≤11:30、开板次数、分钟级止盈等需分时数据，当前以日线代理；``QUANT_LUH_BT_LEGACY=1`` 可恢复旧规则。

- 命令行：``python scripts/limit_up_backtest.py`` · ``--legacy`` 旧版
"""


def limit_up_rules_sections() -> list[dict[str, str]]:
    return [
        {"类别": "信号", "规则": "T 日收盘涨停且封板质量达标"},
        {"类别": "买入", "规则": "T+1 非一字再次封板，涨停价模拟成交"},
        {"类别": "卖出", "规则": "最早 T+2 双阶梯止盈 / 收盘止损"},
        {"类别": "数据", "规则": "仅日线 OHLCV，无分时盘口"},
        {"类别": "回测", "规则": "T+1 开盘价买入 → T+2 起一字续骑 → 开板日开盘价卖出"},
    ]
