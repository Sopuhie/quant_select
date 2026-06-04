"""打板模块专用配置（独立于 short_term / run_daily）。"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _env_int_bounded(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(str(os.environ.get(name, str(default))).strip())
    except ValueError:
        v = default
    return max(lo, min(hi, v))


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, str(default))).strip())
    except ValueError:
        return default


# 信号日 T 收盘确认涨停封板 → T+1 打板买入 → 最早 T+2 卖出（A 股 T+1 交割）
LUH_HOLDING_DAYS = 2
LUH_SELL_OFFSET = _env_int_bounded("QUANT_LUH_SELL_OFFSET", 2, 1, 2)

LUH_TOP_N = _env_int_bounded("QUANT_LUH_TOP_N", 5, 1, 20)
LUH_MIN_HISTORY_BARS = int(os.environ.get("QUANT_LUH_MIN_BARS", "20"))

# 大盘环境分（复用 short_term 同一套指数逻辑；打板默认放宽，弱市仍可扫描）
LUH_MIN_MARKET_SCORE = _env_int_bounded("QUANT_LUH_MIN_MARKET_SCORE", 20, 0, 100)
LUH_MARKET_INDEX_CODE = str(
    os.environ.get("QUANT_LUH_MARKET_INDEX", "000852")
).strip().zfill(6)
LUH_MARKET_MOMENTUM_DAYS = int(os.environ.get("QUANT_LUH_MKT_MOM_DAYS", "5"))
# 打板弱市仍可扫描：允许指数 5 日动量小幅为负（默认 -5%）
LUH_MARKET_MOMENTUM_MIN = _env_float("QUANT_LUH_MKT_MOM_MIN", -0.05)

# 信号日封板质量：收盘/最高 >= 该比例视为成功封板
LUH_SEAL_CLOSE_HIGH_MIN = _env_float("QUANT_LUH_SEAL_MIN", 0.995)
# 封板实体占比下限（防长上影假板）
LUH_ENTITY_RATIO_MIN = _env_float("QUANT_LUH_ENTITY_RATIO", 0.85)

# 流动性
LUH_MIN_AMOUNT = _env_float("QUANT_LUH_MIN_AMOUNT", 80_000_000.0)
LUH_MIN_TURNOVER = _env_float("QUANT_LUH_MIN_TURNOVER", 3.0)
LUH_TURNOVER_GOLDEN_MIN = _env_float("QUANT_LUH_TURNOVER_MIN", 4.0)
LUH_TURNOVER_GOLDEN_MAX = _env_float("QUANT_LUH_TURNOVER_MAX", 35.0)

# T+1 打板买入：必须 T+1 再次收盘涨停（非一字）才视为成交
LUH_BUY_REQUIRE_CLOSE_LIMIT = os.environ.get("QUANT_LUH_BUY_CLOSE_LIMIT", "1") in (
    "1",
    "true",
    "True",
)

# T+2 止盈 / 止损
LUH_T2_TAKE_PROFIT_TIER1 = _env_float("QUANT_LUH_T2_TP1", 0.10)
LUH_T2_TAKE_PROFIT_TIER2 = _env_float("QUANT_LUH_T2_TP2", 0.06)
LUH_T2_TAKE_PROFIT_TIER2_LOCK = _env_float("QUANT_LUH_T2_TP2_LOCK", 0.04)
LUH_T2_STOP_LOSS = _env_float("QUANT_LUH_T2_STOP", 0.07)
LUH_T2_RIDE_MIN_PCT = _env_float("QUANT_LUH_T2_RIDE", 0.05)

LUH_EXCLUDE_ST = os.environ.get("QUANT_LUH_EXCLUDE_ST", "1") not in (
    "0",
    "false",
    "False",
)
LUH_EXCLUDE_BJ = os.environ.get("QUANT_LUH_EXCLUDE_BJ", "1") not in (
    "0",
    "false",
    "False",
)

LUH_HOLD_PLAN = (
    f"T 日收盘涨停封板确认 → T+1 非一字再次封板以涨停价模拟买入 → "
    f"最早 T+2 双阶梯止盈（{LUH_T2_TAKE_PROFIT_TIER1 * 100:.0f}%/"
    f"{LUH_T2_TAKE_PROFIT_TIER2 * 100:.0f}%）或收盘止损 {LUH_T2_STOP_LOSS * 100:.0f}%"
)

LUH_TODAY_JSON = PROJECT_ROOT / "limit_up_today.json"

# 回测连板骑乘：T+2 起最多向后查看的 K 线根数
LUH_BACKTEST_MAX_RIDE_BARS = _env_int_bounded("QUANT_LUH_BT_MAX_RIDE", 30, 5, 120)

# ---------- 回测优化规则（默认开启，QUANT_LUH_BT_LEGACY=1 恢复旧版） ----------
LUH_BACKTEST_LEGACY = os.environ.get("QUANT_LUH_BT_LEGACY", "0") in ("1", "true", "True")
LUH_BT_SLIPPAGE = _env_float("QUANT_LUH_BT_SLIPPAGE", 0.005)
LUH_BT_INDEX_RET_DAYS = int(os.environ.get("QUANT_LUH_BT_INDEX_DAYS", "20"))
LUH_BT_MIN_INDEX_RET = _env_float("QUANT_LUH_BT_MIN_INDEX_RET", 0.0)
LUH_BT_MIN_MARKET_LIMIT_UP = _env_int_bounded("QUANT_LUH_BT_MIN_LIMIT_UP", 50, 0, 500)
LUH_BT_MIN_CONCEPT_LIMIT_UP = _env_int_bounded("QUANT_LUH_BT_MIN_CONCEPT_LU", 3, 0, 50)
LUH_BT_TP_CLOSE_TIER1 = _env_float("QUANT_LUH_BT_TP1", 0.10)
LUH_BT_TP_CLOSE_TIER2 = _env_float("QUANT_LUH_BT_TP2", 0.06)
# T+2 收盘涨停且换手≥该值视为强封单，持有至 T+3（日线代理封单/流通盘）
LUH_BT_STRONG_BOARD_TURNOVER = _env_float("QUANT_LUH_BT_STRONG_TO", 0.01)
# T+1 开盘距理论涨停价上限（代理上午封板）
LUH_BT_T1_OPEN_NEAR_LIMIT = _env_float("QUANT_LUH_BT_T1_OPEN_GAP", 0.02)
# T+1 最低价/收盘价下限（代理开板次数≤1）
LUH_BT_T1_LOW_CLOSE_MIN = _env_float("QUANT_LUH_BT_T1_LOW_RATIO", 0.995)
LUH_BT_STOP_LOSS = _env_float("QUANT_LUH_BT_STOP", 0.07)
# 优化回测 T+1 是否必须再次收盘涨停才买入（默认否：T+1 非一字则开盘价+滑点买）
LUH_BT_REQUIRE_T1_CLOSE_LIMIT = os.environ.get("QUANT_LUH_BT_T1_CLOSE_LIMIT", "0") in (
    "1",
    "true",
    "True",
)
