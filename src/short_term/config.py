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


# 持有 1 个交易日：T 日收盘信号 → T+1 买 → T+2 卖
SHORT_HOLDING_DAYS = 1
SHORT_TOP_N = _env_int_bounded("QUANT_SHORT_TOP_N", 20, 1, 20)

SHORT_MIN_HISTORY_BARS = int(os.environ.get("QUANT_SHORT_MIN_BARS", "35"))
SHORT_HIST_LIMIT = int(os.environ.get("QUANT_SHORT_HIST_LIMIT", "120"))

SHORT_MIN_MARKET_SCORE = int(os.environ.get("QUANT_SHORT_MIN_MARKET_SCORE", "50"))

SHORT_MA_FAST = 5
SHORT_MA_SLOW = 10

# 5 日涨幅上限（各板块当日涨跌幅上下限在 strategy 内按代码动态计算）
SHORT_MAX_5D_RETURN = _env_float("QUANT_SHORT_MAX_5D_RET", 0.22)

SHORT_VOL_RATIO_5D_MIN = _env_float("QUANT_SHORT_VR5_MIN", 1.25)
SHORT_VOL_RATIO_1D_MIN = _env_float("QUANT_SHORT_VR1_MIN", 1.08)

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
