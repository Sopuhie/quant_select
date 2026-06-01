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

# 纯日线执行：T 日收盘确认信号，T+1 开盘限价买入；未触发止损时的平仓日偏移
SHORT_SELL_OFFSET = _env_int_bounded("QUANT_SHORT_SELL_OFFSET", 2, 1, 2)
# 遗留字段：旧版盘中止损比例（现仅用于文档/兼容引用）
SHORT_STOP_LOSS_RATIO = _env_float("QUANT_SHORT_STOP_LOSS", 0.03)
# T+1 收盘价破位止损：收盘价 < 买入价 × (1 - 该比例) 时在 T+1 收盘离场
SHORT_CLOSE_STOP_RATIO = _env_float("QUANT_SHORT_CLOSE_STOP", 0.05)
# T+1 开盘入场：相对信号日收盘价，高开超过该比例则放弃（防追高）
SHORT_ENTRY_MAX_CHASE = _env_float("QUANT_SHORT_ENTRY_MAX_CHASE", 0.01)
# T+1 开盘入场：低开超过该比例则放弃（弱势缺口）
SHORT_ENTRY_MIN_GAP = _env_float("QUANT_SHORT_ENTRY_MIN_GAP", -0.02)

# T+1 盘中冲高动态止盈：最高价涨幅 >= 该比例时，按 (高+收)/2 保守平仓
SHORT_T1_TAKE_PROFIT_PCT = _env_float("QUANT_SHORT_T1_TP", 0.06)
# T+1 收盘强势阈值：>= 该比例则延续至 T+2 趋势骑乘；否则 T+1 收盘即平
SHORT_T1_STRONG_CLOSE_PCT = _env_float("QUANT_SHORT_T1_STRONG", 0.04)

SHORT_HOLD_PLAN = (
    f"T 日收盘确认信号 → T+1 开盘在 "
    f"[{(1 + SHORT_ENTRY_MIN_GAP) * 100:.0f}%, {(1 + SHORT_ENTRY_MAX_CHASE) * 100:.0f}%] "
    f"区间限价买入 → T+1 冲高≥{SHORT_T1_TAKE_PROFIT_PCT * 100:.0f}% 动态止盈 → "
    f"收盘破 -{SHORT_CLOSE_STOP_RATIO * 100:.0f}% 止损 → "
    f"收盘<{SHORT_T1_STRONG_CLOSE_PCT * 100:.0f}% 则 T+1 离场，否则 T+2 趋势骑乘"
)
SHORT_TOP_N = _env_int_bounded("QUANT_SHORT_TOP_N", 5, 1, 20)

SHORT_MIN_HISTORY_BARS = int(os.environ.get("QUANT_SHORT_MIN_BARS", "35"))
SHORT_HIST_LIMIT = int(os.environ.get("QUANT_SHORT_HIST_LIMIT", "120"))

SHORT_MIN_MARKET_SCORE = int(os.environ.get("QUANT_SHORT_MIN_MARKET_SCORE", "60"))
# 短线大盘环境分锚定指数（默认中证1000，更贴近小盘/题材非线性环境）
SHORT_MARKET_INDEX_CODE = str(
    os.environ.get("QUANT_SHORT_MARKET_INDEX", "000852")
).strip().zfill(6)
# 指数 N 日动量 > 该阈值才允许扫描（与 MA20 环境分叠加，过滤贴线放行）
SHORT_MARKET_MOMENTUM_DAYS = int(os.environ.get("QUANT_SHORT_MKT_MOM_DAYS", "5"))
SHORT_MARKET_MOMENTUM_MIN = _env_float("QUANT_SHORT_MKT_MOM_MIN", 0.0)

# 信号日 K 线实体比例下限（防长上影线/炸板）
SHORT_ENTITY_RATIO_MIN = _env_float("QUANT_SHORT_ENTITY_RATIO", 0.7)
# 涨幅非线性打分：满分区间上沿（%），满分区间为 [2%, SHORT_PCT_SCORE_MAX]
SHORT_PCT_SCORE_MAX = _env_float("QUANT_SHORT_PCT_SCORE_MAX", 3.5)

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
