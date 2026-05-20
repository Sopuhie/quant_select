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


# 持有 1 个交易日：T 日收盘信号 → T+1 买 → T+2 卖
SHORT_HOLDING_DAYS = 1
SHORT_TOP_N = _env_int_bounded("QUANT_SHORT_TOP_N", 5, 1, 20)

# 扫描所需最少 K 线根数（低于中长线 MIN_HISTORY_BARS）
SHORT_MIN_HISTORY_BARS = int(os.environ.get("QUANT_SHORT_MIN_BARS", "35"))
SHORT_HIST_LIMIT = int(os.environ.get("QUANT_SHORT_HIST_LIMIT", "120"))

# 大盘环境：沪深300 规则分低于此值则当日不出信号
SHORT_MIN_MARKET_SCORE = int(os.environ.get("QUANT_SHORT_MIN_MARKET_SCORE", "50"))

SHORT_MA_FAST = 5
SHORT_MA_SLOW = 10

# 信号日涨幅区间（避免追高与弱势）
SHORT_MIN_DAY_RETURN = float(os.environ.get("QUANT_SHORT_MIN_DAY_RET", "0.008"))
SHORT_MAX_DAY_RETURN = float(os.environ.get("QUANT_SHORT_MAX_DAY_RET", "0.075"))
SHORT_MAX_5D_RETURN = float(os.environ.get("QUANT_SHORT_MAX_5D_RET", "0.22"))

SHORT_VOL_RATIO_5D_MIN = float(os.environ.get("QUANT_SHORT_VR5_MIN", "1.25"))
SHORT_VOL_RATIO_1D_MIN = float(os.environ.get("QUANT_SHORT_VR1_MIN", "1.08"))

SHORT_KDJ_J_MIN = float(os.environ.get("QUANT_SHORT_KDJ_J_MIN", "25"))
SHORT_KDJ_J_MAX = float(os.environ.get("QUANT_SHORT_KDJ_J_MAX", "92"))
SHORT_KDJ_J_SLOPE_MIN = float(os.environ.get("QUANT_SHORT_KDJ_J_SLOPE", "1.5"))

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
SHORT_NEAR_LIMIT_PCT = float(os.environ.get("QUANT_SHORT_NEAR_LIMIT_PCT", "0.09"))

# 默认剔除北交所（8/4 开头）
SHORT_EXCLUDE_BJ = os.environ.get("QUANT_SHORT_EXCLUDE_BJ", "1") not in (
    "0",
    "false",
    "False",
)

SHORT_TODAY_JSON = PROJECT_ROOT / "short_today.json"
