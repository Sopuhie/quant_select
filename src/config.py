"""全局配置：路径、数据源、训练与选股参数。"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

# 项目根目录（quant_select/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "stocks.db"
MODEL_PATH = MODELS_DIR / "lgb_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
BEST_LGB_PARAMS_JSON = MODELS_DIR / "best_params.json"
META_STACKER_PATH = MODELS_DIR / "meta_rank_stacker.pkl"
CATBOOST_MODEL_PATH = MODELS_DIR / "cat_ranker.cbm"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 训练标签：预测未来 N 个交易日收益率（收盘价到收盘价）
LABEL_HORIZON_DAYS = 5

# 用于训练/预测的历史 K 线最少条数
MIN_HISTORY_BARS = 60

# 每日选股 TopN（环境变量 QUANT_TOP_N 可调，例如 5、10；默认 3）
def _env_int_bounded(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(str(os.environ.get(name, str(default))).strip())
    except ValueError:
        v = default
    return max(lo, min(hi, v))


TOP_N_SELECTION = _env_int_bounded("QUANT_TOP_N", 3, 1, 50)

# 股票池：all | hs300 | zz500（中证500）
STOCK_POOL = os.environ.get("QUANT_STOCK_POOL", "hs300")

# 排除 ST（名称含 *ST / ST 等）
EXCLUDE_ST = os.environ.get("QUANT_EXCLUDE_ST", "1") not in ("0", "false", "False")

# 排除选股日「上一交易日」收盘价接近涨停（主板约 10%，科创板/创业板请按需调低阈值）
EXCLUDE_NEAR_LIMIT_LAST_BAR = os.environ.get("QUANT_EXCLUDE_NEAR_LIMIT", "1") in (
    "1",
    "true",
    "True",
)
NEAR_LIMIT_PCT_THRESHOLD = float(os.environ.get("QUANT_NEAR_LIMIT_PCT", "0.095"))

# --- 前期涨幅/动量压制（截面清洗前按原始因子比较）---
# ``QUANT_PREV_GAIN_SUPPRESSION=0`` 关闭；为 1/true 时同时作用于：全市场选股/回测/在线预测与「智能诊股」硬提示。
ENABLE_PREV_GAIN_SUPPRESSION = os.environ.get(
    "QUANT_PREV_GAIN_SUPPRESSION", "0"
) in ("1", "true", "True")
MAX_ALLOWED_5D_RETURN = float(os.environ.get("QUANT_MAX_5D_RETURN", "0.12"))
MAX_ALLOWED_20D_RETURN = float(os.environ.get("QUANT_MAX_20D_MOMENTUM", "0.30"))

# 股票池：训练与预测时最多处理的股票数量（AkShare 全市场较慢，可先调小验证流程）
MAX_STOCKS_UNIVERSE = int(os.environ.get("QUANT_MAX_STOCKS", "400"))

# AkShare HTTP 超时（秒）；超时或失败则跳过该股票，避免长时间卡死
AKSHARE_REQUEST_TIMEOUT = float(os.environ.get("QUANT_AK_TIMEOUT", "30"))

# 单次请求失败后的重试次数（网络抖动、东方财富限流、连接被重置时可调大）
AKSHARE_FETCH_RETRIES = max(1, int(os.environ.get("QUANT_FETCH_RETRIES", "5")))

# 重试间隔基数（秒）；AkShare 指数退避时与 2^attempt 相乘，短区间请求时会再封顶
AKSHARE_FETCH_RETRY_SLEEP = float(os.environ.get("QUANT_FETCH_RETRY_SLEEP", "0.8"))

# 自然日跨度不超过该值时，**优先**用 Baostock 拉日线（增量多为 1～数日，可避免东财限流下 AkShare 多次长退避）。
# 设为 0 关闭优先逻辑。全量拉多年历史时跨度大，仍走 AkShare 优先。
BAOSTOCK_FIRST_IF_RANGE_DAYS = max(
    0, int(os.environ.get("QUANT_BAOSTOCK_FIRST_IF_RANGE_DAYS", "99999"))
)

# 每日选股拉 K 线：只取最近若干自然日的数据即可算满窗口因子，减轻接口压力
PREDICT_HISTORY_CALENDAR_DAYS = int(os.environ.get("QUANT_PREDICT_LOOKBACK_DAYS", "800"))

# 选股阶段并发线程数（过大易被东方财富/AkShare 限流，导致「零条有效 K 线」）
PREDICT_FETCH_WORKERS = int(os.environ.get("QUANT_FETCH_WORKERS", "2"))

# AkShare 全部失败时是否用 Baostock 串行兜底（需安装 baostock；非线程安全故全局锁串行）。
# 另：AkShare 若因连接重置/超时等瞬时错误失败，即使本项为 0，``fetch_daily_hist`` 仍会尝试 Baostock 自动兜底。
USE_BAOSTOCK_FALLBACK = os.environ.get("QUANT_BAOSTOCK_FALLBACK", "1") not in (
    "0",
    "false",
    "False",
)

# 本机若配置了失效的 HTTP(S)_PROXY，requests 访问东方财富会全部 ProxyError。
# 默认将 eastmoney 相关域名并入 NO_PROXY / no_proxy，使该主机直连（仍走系统代理访问其他站点）。
# 若必须通过公司代理访问东方财富，请设 ``QUANT_AKSHARE_BYPASS_PROXY_FOR_EASTMONEY=0``。
AKSHARE_BYPASS_PROXY_FOR_EASTMONEY = os.environ.get(
    "QUANT_AKSHARE_BYPASS_PROXY_FOR_EASTMONEY", "1"
) not in ("0", "false", "False")

_EASTMONEY_NO_PROXY_LOCK = threading.Lock()
_EASTMONEY_NO_PROXY_APPLIED = False


def ensure_eastmoney_no_proxy_if_configured() -> None:
    """
    将东方财富相关域名并入 ``NO_PROXY`` / ``no_proxy``，使 requests 访问该域不走系统代理。

    本机若配置了失效的 ``HTTP_PROXY``/``HTTPS_PROXY``，默认会导致 AkShare 访问
    ``push2his.eastmoney.com`` 等全部 ``ProxyError``。关闭本行为请设环境变量
    ``QUANT_AKSHARE_BYPASS_PROXY_FOR_EASTMONEY=0``。
    """
    global _EASTMONEY_NO_PROXY_APPLIED
    if not AKSHARE_BYPASS_PROXY_FOR_EASTMONEY:
        return
    with _EASTMONEY_NO_PROXY_LOCK:
        if _EASTMONEY_NO_PROXY_APPLIED:
            return
        extra = (
            "eastmoney.com",
            ".eastmoney.com",
            "push2his.eastmoney.com",
            "push2.eastmoney.com",
        )
        for key in ("NO_PROXY", "no_proxy"):
            raw = os.environ.get(key, "")
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            seen = set(parts)
            for h in extra:
                if h not in seen:
                    seen.add(h)
                    parts.append(h)
            os.environ[key] = ",".join(parts)
        _EASTMONEY_NO_PROXY_APPLIED = True


def _env_hhmm(name: str, default: str) -> str:
    """解析 ``HH:MM``（24 小时制），非法则退回 default。"""
    raw = str(os.environ.get(name, default)).strip().replace("：", ":")
    if not raw:
        raw = default
    parts = raw.split(":")
    if len(parts) != 2:
        return default
    try:
        h = int(parts[0].strip())
        m = int(parts[1].strip())
    except ValueError:
        return default
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return default
    return f"{h:02d}:{m:02d}"


# Streamlit 内置后台调度触发时刻（本机时间）；自测可设环境变量 QUANT_SCHEDULER_TIME=09:05
SCHEDULER_RUN_AT = _env_hhmm("QUANT_SCHEDULER_TIME", "19:00")

# LightGBM 训练参数（可按机器性能调整）
LGB_PARAMS = {
    "objective": "regression",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "seed": 42,
}

# LambdaRank（LGBMRanker）默认超参；与 Optuna 搜索空间对齐时可覆盖。
# eval_at 勿写入 JSON/merge 字典；且勿传给 LGBMRanker（sklearn 会与内部 params 重复并告警），用库默认 ndcg@k 即可。
LGB_RANKER_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "bagging_freq": 1,
    "seed": 42,
}

# 特征列：量价框架（在 14 项基础因子上扩展流动性/量价协同/分布类因子）
FEATURE_COLUMNS = [
    "factor_bias_5",  # 5日均线乖离
    "factor_bias_10",  # 10日均线乖离
    "factor_bias_20",  # 20日均线乖离（生命线）
    "factor_bias_60",  # 60日均线乖离（牛熊线）
    "factor_ratio_5_20",  # 5日/20日均线距离比
    "factor_ratio_10_60",  # 10日/60日均线距离比
    "factor_return_1d",  # 1日收益
    "factor_return_5d",  # 5日收益
    "factor_momentum_10d",  # 10日动量
    "factor_volume_ratio",  # 量比（相对5日均量）
    "factor_volume_position",  # 量能位置（5日均量相对20日均量）
    "factor_volatility_5d",  # 5日波动
    "factor_volatility_20d",  # 20日波动
    "factor_close_position",  # 收盘位置（资金承接代理）
    # --- 扩展：筹码/流动性/量价结构代理（仍仅依赖 OHLCV；可叠加 config 资金流映射）---
    "factor_amihud_20d",  # Amihud 非流动性（|收益|/成交额代理）
    "factor_pv_corr_10d",  # 价量变化 10 日滚动相关（背离为负）
    "factor_vwap_bias_20d",  # 相对 20 日 VWAP 的偏离（成本带代理）
    "factor_bb_width_20d",  # 布林带宽（波动 regime）
    "factor_drawdown_60d",  # 60 日高点回撤深度（套牢/获利结构代理）
    "factor_shrink_pullback_5d",  # 近 5 日下跌中缩量程度（缩量回调为正）
    # --- 深度优化：资金流 / 筹码 / 技术（正交在截面清洗内完成）---
    "factor_big_order_net_ratio",  # 大单+超大单净流入占比（AkShare 增量库优先，缺省为 OHLCV 代理）
    "factor_north_hold_ratio_chg",  # 北向持股占 A 股比例日变化（增量库；无则 0）
    "factor_chip_profit_ratio",  # 获利盘代理：收盘相对 60 日 VWAP
    "factor_chip_concentration_width",  # 筹码集中度：(95-5分位)-(85-15分位) 宽度 / 价
    "factor_rsi_14",
    "factor_kdj_j",
    "factor_macd_hist",
    # 北向净流入强度（按日全市场相同标量）与 5 日收益交互；无数据时为 0
    "factor_hsgt_flow_interact",
]

# 因子时间序列：EWM 与简单 rolling 的混合权重（越大越偏重近期）
FACTOR_EWM_BLEND = float(os.environ.get("QUANT_FACTOR_EWM_BLEND", "0.35"))
FACTOR_EWM_HALFLIFE_DAYS = max(2.0, float(os.environ.get("QUANT_FACTOR_EWM_HALFLIFE", "10")))

# LambdaRank：对极端负收益样本加权（主动避雷；近似自定义损失）
RANK_SAMPLE_WEIGHT_EXTREME_RET = float(
    os.environ.get("QUANT_RANK_WEIGHT_EXTREME_RET", "2.5")
)
RANK_SAMPLE_WEIGHT_NEG_THRESH = float(
    os.environ.get("QUANT_RANK_WEIGHT_NEG_THRESH", "-0.05")
)
# 历史大幅回撤样本：在极端损失加权之上再乘倍数（与 quant.ranking 可合并）
RANK_DRAWDOWN_WEIGHT_MULT = float(os.environ.get("QUANT_RANK_DD_WEIGHT_MULT", "2.0"))
RANK_DRAWDOWN_WEIGHT_THRESH = float(os.environ.get("QUANT_RANK_DD_WEIGHT_THRESH", "-0.07"))
# Meta 训练：训练集内按交易日 Walk-forward OOF 折数；0 或 1 表示关闭 OOF（与旧版同分布的 in-sample meta）
# OOF 折内并行见 model_trainer.walkforward_oof_base_predictions：QUANT_OOF_MAX_WORKERS、QUANT_OOF_WORKER_OMP_THREADS 等
RANK_META_OOF_FOLDS = int(os.environ.get("QUANT_META_OOF_FOLDS", "0"))

# PSI：验证集 vs 训练集预测分分布稳定性（仅记录告警，不阻断训练）
PSI_SCORE_BINS = max(5, int(os.environ.get("QUANT_PSI_BINS", "12")))
PSI_ALERT_THRESHOLD = float(os.environ.get("QUANT_PSI_ALERT", "0.25"))

# 回测默认摩擦（可被 CLI 覆盖；亦写入 config.json quant.backtest）
BACKTEST_DEFAULT_COMMISSION_RATE = float(
    os.environ.get("QUANT_BACKTEST_COMMISSION", "0.0003")
)
BACKTEST_DEFAULT_STAMP_DUTY_SELL = float(
    os.environ.get("QUANT_BACKTEST_STAMP_SELL", "0.001")
)
# 回测冲击成本（单边）；与 scripts/backtest.py 默认及 quant.backtest.slippage_rate 对齐
BACKTEST_DEFAULT_SLIPPAGE_RATE = float(
    os.environ.get("QUANT_BACKTEST_SLIPPAGE", "0.001")
)
# 涨跌停价与 OHLC 对齐容差（元）
BACKTEST_LIMIT_PRICE_EPS_YUAN = float(
    os.environ.get("QUANT_BACKTEST_LIMIT_EPS", "0.02")
)
# 相邻两根日 K 自然日间隔超过该值视为长期停牌/缺口，禁止买入
BACKTEST_MAX_CALENDAR_GAP_DAYS = int(
    os.environ.get("QUANT_BACKTEST_MAX_CAL_GAP", "6")
)

# 实时监控：请求间隔抖动（秒）、每分钟请求上限、同股同类型信号去重窗口（分钟）
REALTIME_MONITOR_SLEEP_SEC_MIN = float(
    os.environ.get("QUANT_REALTIME_SLEEP_MIN", "1.0")
)
REALTIME_MONITOR_SLEEP_SEC_MAX = float(
    os.environ.get("QUANT_REALTIME_SLEEP_MAX", "3.0")
)
REALTIME_MONITOR_MAX_REQUESTS_PER_MINUTE = int(
    os.environ.get("QUANT_REALTIME_MAX_RPM", "40")
)
REALTIME_SIGNAL_DEDUP_WINDOW_MINUTES = int(
    os.environ.get("QUANT_SIGNAL_DEDUP_MIN", "5")
)
# 连续 API 失败达到该次数后暂停监控（秒）
REALTIME_MONITOR_PAUSE_AFTER_FAILS = int(
    os.environ.get("QUANT_REALTIME_PAUSE_FAILS", "3")
)
REALTIME_MONITOR_PAUSE_SECONDS = float(
    os.environ.get("QUANT_REALTIME_PAUSE_SEC", "300.0")
)

# 形态匹配：DTW 前粗筛数量、量价 DTW 中成交量权重
PATTERN_COARSE_TOP_K = max(20, int(os.environ.get("QUANT_PATTERN_COARSE_TOP_K", "100")))
PATTERN_DTW_VOLUME_WEIGHT = float(os.environ.get("QUANT_PATTERN_VOL_WEIGHT", "1.0"))

# --- 交易员经验硬过滤参数 (留空或为 None 时代表不限制) ---
# 1. 价格范围 (单位: 元)
MIN_PRICE = None  # 例如: 5.0
MAX_PRICE = None  # 例如: 100.0

# 2. 流通市值范围 (单位: 亿元；与库内 mcap 元字段对齐时除以 1e8)
MIN_MCAP = None  # 例如: 30.0 (低于30亿的微盘股不要)
MAX_MCAP = None  # 例如: 500.0 (大于500亿的巨无霸不要)

# 3. 股票热度范围（以今日换手率 % 为代理指标；无换手列时用 volume_ratio_raw 近似，阈值同文档除以 2）
MIN_TURNOVER = None  # 例如: 1.5 (过滤掉无人关注的僵尸股)
MAX_TURNOVER = None  # 例如: 15.0 (过滤掉短期极度亢奋、换手过热的筹码松动股)


def get_experience_thresholds() -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """
    经验硬过滤阈值：优先读取项目根目录 ``config.json`` 中 ``experience_filters``
    （由 Streamlit「任务 C」面板写入）；缺省或解析失败时退回本模块上述常量。
    """
    mp, Mxp, mm, Mxm, mt, Mxt = (
        MIN_PRICE,
        MAX_PRICE,
        MIN_MCAP,
        MAX_MCAP,
        MIN_TURNOVER,
        MAX_TURNOVER,
    )
    try:
        from .config_manager import _normalize_experience_filters_merged, config_manager

        config_manager.reload()
        raw = config_manager.config.get("experience_filters")
        if not isinstance(raw, dict):
            return (mp, Mxp, mm, Mxm, mt, Mxt)
        raw = _normalize_experience_filters_merged(dict(raw))

        def _pick(key: str, cur: float | None) -> float | None:
            if key not in raw:
                return cur
            v = raw[key]
            if v is None:
                return None
            if isinstance(v, str) and not str(v).strip():
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return cur

        return (
            _pick("min_price", mp),
            _pick("max_price", Mxp),
            _pick("min_mcap", mm),
            _pick("max_mcap", Mxm),
            _pick("min_turnover", mt),
            _pick("max_turnover", Mxt),
        )
    except Exception:
        return (mp, Mxp, mm, Mxm, mt, Mxt)

# --- 热门题材交易系统规则 v2.0 核心参数 ---
THEME_MA_SHORT = 20
THEME_MA_LONG = 60
THEME_VOL_RATIO_MIN_5D = 1.5
THEME_VOL_RATIO_MIN_1D = 1.3
THEME_KDJ_J_SLOPE_MIN = 5.0
THEME_KDJ_LEVEL_1 = 100.0
THEME_KDJ_LEVEL_2 = 110.0


def get_quant_config_merged() -> dict[str, Any]:
    """
    读取 ``config.json`` 中 ``quant`` 段并与代码默认值合并（缺省键不回退为 0）。
    避免顶层循环依赖：仅在调用时 import ``config_manager``。
    """
    defaults: dict[str, Any] = {
        "backtest": {
            "commission_rate": BACKTEST_DEFAULT_COMMISSION_RATE,
            "stamp_duty_sell_rate": BACKTEST_DEFAULT_STAMP_DUTY_SELL,
            "slippage_rate": BACKTEST_DEFAULT_SLIPPAGE_RATE,
        },
        "pattern": {
            "coarse_top_k": PATTERN_COARSE_TOP_K,
            "dtw_volume_weight": PATTERN_DTW_VOLUME_WEIGHT,
        },
        "psi": {
            "bins": PSI_SCORE_BINS,
            "alert_threshold": PSI_ALERT_THRESHOLD,
        },
        "ranking": {
            "extreme_loss_weight": RANK_SAMPLE_WEIGHT_EXTREME_RET,
            "extreme_loss_thresh": RANK_SAMPLE_WEIGHT_NEG_THRESH,
            "drawdown_penalty_mult": RANK_DRAWDOWN_WEIGHT_MULT,
            "drawdown_penalty_thresh": RANK_DRAWDOWN_WEIGHT_THRESH,
            "meta_oof_folds": RANK_META_OOF_FOLDS,
        },
    }
    try:
        from .config_manager import config_manager

        config_manager.reload()
        raw = config_manager.config.get("quant")
        if not isinstance(raw, dict):
            return defaults
        out: dict[str, Any] = {}
        for sec, d0 in defaults.items():
            rsec = raw.get(sec)
            if isinstance(rsec, dict) and isinstance(d0, dict):
                merged = dict(d0)
                merged.update({k: v for k, v in rsec.items() if v is not None})
                out[sec] = merged
            else:
                out[sec] = d0
        for sec, v in raw.items():
            if sec not in out:
                out[sec] = v
        return out
    except Exception:
        return defaults
