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


# 逻辑持有交易日数（文案）；A 股最早 T+2 卖出，实际至少跨 1 个交割夜
SHORT_HOLDING_DAYS = 2

# 纯日线执行：T 日收盘确认信号，T+1 开盘限价买入；未触发止损时的平仓日偏移
SHORT_SELL_OFFSET = _env_int_bounded("QUANT_SHORT_SELL_OFFSET", 2, 1, 2)
# 遗留字段：旧版盘中止损比例（现仅用于文档/兼容引用）
SHORT_STOP_LOSS_RATIO = _env_float("QUANT_SHORT_STOP_LOSS", 0.03)
# T+1 收盘价破位止损：收盘价 < 买入价 × (1 - 该比例) 时在 T+1 收盘离场
SHORT_CLOSE_STOP_RATIO = _env_float("QUANT_SHORT_CLOSE_STOP", 0.06)
# T+1 开盘入场：相对信号日收盘价，高开超过该比例则放弃（牛股放行通道上限 5.5%）
SHORT_ENTRY_MAX_CHASE = _env_float("QUANT_SHORT_ENTRY_MAX_CHASE", 0.055)
SHORT_ENTRY_MAX_CHASE_HARD_CAP = 0.055
# 微幅高开（≤该比例）按开盘价成交；更高开则按 (open+low)/2 模拟分时低吸
SHORT_ENTRY_DIP_OPEN_THRESHOLD = _env_float("QUANT_SHORT_ENTRY_DIP_OPEN", 0.015)
# T+1 开盘入场：低开超过该比例则放弃（弱势缺口）
SHORT_ENTRY_MIN_GAP = _env_float("QUANT_SHORT_ENTRY_MIN_GAP", -0.02)

# T+1 双阶梯动态止盈
SHORT_T1_TAKE_PROFIT_TIER1_PCT = _env_float("QUANT_SHORT_T1_TP_T1", 0.06)
SHORT_T1_TAKE_PROFIT_TIER2_PCT = _env_float("QUANT_SHORT_T1_TP_T2", 0.05)
SHORT_T1_TAKE_PROFIT_TIER2_LOCK = _env_float("QUANT_SHORT_T1_TP_T2_LOCK", 0.03)
SHORT_T1_TAKE_PROFIT_PCT = SHORT_T1_TAKE_PROFIT_TIER1_PCT
# 平庸股（盘中未达第二阶梯）非对称收盘止损
SHORT_MEDIOCRE_STOP_RATIO = _env_float("QUANT_SHORT_MEDIOCRE_STOP", 0.04)
# T+1 收盘强势阈值：>= 该比例则 T+2 趋势骑乘（收盘 > T+1 收盘）
SHORT_T1_STRONG_CLOSE_PCT = _env_float("QUANT_SHORT_T1_STRONG", 0.04)

SHORT_HOLD_PLAN = (
    f"T 日收盘确认信号 → T+1 非对称买入 "
    f"[{(1 + SHORT_ENTRY_MIN_GAP) * 100:.0f}%, {(1 + SHORT_ENTRY_MAX_CHASE) * 100:.0f}%] "
    f"→ T+2 最早卖出（A股T+1交割）："
    f"T+2 双阶梯止盈/非对称止损；"
    f"T+1 收≥{SHORT_T1_STRONG_CLOSE_PCT * 100:.0f}% 且 T+2 续强则收盘骑乘"
)
SHORT_TOP_N = _env_int_bounded("QUANT_SHORT_TOP_N", 5, 1, 20)

SHORT_MIN_HISTORY_BARS = int(os.environ.get("QUANT_SHORT_MIN_BARS", "35"))
SHORT_HIST_LIMIT = int(os.environ.get("QUANT_SHORT_HIST_LIMIT", "120"))

SHORT_MIN_MARKET_SCORE = int(os.environ.get("QUANT_SHORT_MIN_MARKET_SCORE", "60"))
# 短线大盘环境分锚定指数（默认中证1000，更贴近小盘/题材非线性环境）
SHORT_MARKET_INDEX_CODE = str(
    os.environ.get("QUANT_SHORT_MARKET_INDEX", "000300")
).strip().zfill(6)


def short_market_index_label(index_code: str | None = None) -> str:
    """短线大盘环境分锚定指数的中文名（界面/推送文案）。"""
    code = str(index_code or SHORT_MARKET_INDEX_CODE).strip().zfill(6)
    if code == "000852":
        return "中证1000"
    if code == "000300":
        return "沪深300"
    return code


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
# 换手率黄金区间（%，剔除僵尸股与死亡换手力竭股）
SHORT_TURNOVER_GOLDEN_MIN = _env_float("QUANT_SHORT_TURNOVER_MIN", 3.5)
SHORT_TURNOVER_GOLDEN_MAX = _env_float("QUANT_SHORT_TURNOVER_MAX", 24.0)

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
