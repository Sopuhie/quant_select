"""短线模块专用配置（不修改 src/config.py 既有常量）。"""
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


# 逻辑持有交易日数（文案/钉钉）；实际平仓日见 SHORT_SELL_OFFSET
SHORT_HOLDING_DAYS = 1

# 纯日线执行：T 日收盘价买入；未触发止损时的平仓日偏移（1=T+1 收盘，2=T+2 收盘）
SHORT_SELL_OFFSET = _env_int_bounded("QUANT_SHORT_SELL_OFFSET", 1, 1, 2)
# 盘中 -3% 硬止损（用 T+1 最低价模拟）：卖出价 = 买入价 × (1 - 该比例)
SHORT_STOP_LOSS_RATIO = _env_float("QUANT_SHORT_STOP_LOSS", 0.03)

SHORT_HOLD_PLAN = (
    f"T 日收盘确认信号并以收盘价买入 → "
    f"T+1 用最低价模拟 -{SHORT_STOP_LOSS_RATIO * 100:.0f}% 硬止损 → "
    f"未触发则 T+{SHORT_SELL_OFFSET} 收盘平仓"
)
SHORT_TOP_N = _env_int_bounded("QUANT_SHORT_TOP_N", 5, 1, 20)

SHORT_MIN_HISTORY_BARS = int(os.environ.get("QUANT_SHORT_MIN_BARS", "35"))
SHORT_HIST_LIMIT = int(os.environ.get("QUANT_SHORT_HIST_LIMIT", "120"))

SHORT_MIN_MARKET_SCORE = int(os.environ.get("QUANT_SHORT_MIN_MARKET_SCORE", "50"))

SHORT_MA_FAST = 5
SHORT_MA_SLOW = 10

# 5 日涨幅上限（各板块当日涨跌幅上下限在 strategy 内按代码动态计算）
SHORT_MAX_5D_RETURN = _env_float("QUANT_SHORT_MAX_5D_RET", 0.22)

SHORT_VOL_RATIO_5D_MIN = _env_float("QUANT_SHORT_VR5_MIN", 1.25)
SHORT_VOL_RATIO_1D_MIN = _env_float("QUANT_SHORT_VR1_MIN", 1.08)
# 5/1 日量比硬性上限，防停牌复牌畸变污染打分（见 modfy 审计项 2）
SHORT_VOL_RATIO_CLIP_MAX = _env_float("QUANT_SHORT_VOL_RATIO_CLIP", 10.0)

SHORT_KDJ_J_MIN = _env_float("QUANT_SHORT_KDJ_J_MIN", 25.0)
SHORT_KDJ_J_MAX = _env_float("QUANT_SHORT_KDJ_J_MAX", 90.0)
SHORT_KDJ_J_SLOPE_MIN = _env_float("QUANT_SHORT_KDJ_J_SLOPE", 1.5)

# 信号日流动性：成交额(元)与换手率(%) 满足其一即可
SHORT_MIN_AMOUNT = _env_float("QUANT_SHORT_MIN_AMOUNT", 50_000_000.0)
SHORT_MIN_TURNOVER = _env_float("QUANT_SHORT_MIN_TURNOVER", 2.0)

SHORT_EXCLUDE_ST = os.environ.get("QUANT_SHORT_EXCLUDE_ST", "1") not in (
    "0",
    "false",
    "False",
)
SHORT_EXCLUDE_NEAR_LIMIT = os.environ.get("QUANT_SHORT_EXCLUDE_NEAR_LIMIT", "1") in (
    "1",
    "true",
    "True",
)

# 北交所等（与 strategy._exclude_code_prefix 前缀一致）
SHORT_EXCLUDE_BJ = os.environ.get("QUANT_SHORT_EXCLUDE_BJ", "1") not in (
    "0",
    "false",
    "False",
)

SHORT_TODAY_JSON = PROJECT_ROOT / "short_today.json"
