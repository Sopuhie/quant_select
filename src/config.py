"""全局配置：路径、数据源、训练与选股参数。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# 项目根目录（quant_select/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "stocks.db"
MODEL_PATH = MODELS_DIR / "lgb_model.pkl"
BEST_LGB_PARAMS_JSON = MODELS_DIR / "best_params.json"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 训练标签：预测未来 N 个交易日收益率（收盘价到收盘价）
LABEL_HORIZON_DAYS = 5

# 用于训练/预测的历史 K 线最少条数
MIN_HISTORY_BARS = 60

# 每日选股数量
TOP_N_SELECTION = 3

# 股票池：all | hs300 | zz500（中证500）
STOCK_POOL = os.environ.get("QUANT_STOCK_POOL", "hs300")

# 排除 ST（名称含 *ST / ST 等）
EXCLUDE_ST = os.environ.get("QUANT_EXCLUDE_ST", "1") not in ("0", "false", "False")

# 排除选股日「上一交易日」涨跌幅接近涨跌停（主板约 10%）
EXCLUDE_NEAR_LIMIT_LAST_BAR = os.environ.get("QUANT_EXCLUDE_NEAR_LIMIT", "0") in (
    "1",
    "true",
    "True",
)
NEAR_LIMIT_PCT_THRESHOLD = float(os.environ.get("QUANT_NEAR_LIMIT_PCT", "0.095"))

# 股票池：训练与预测时最多处理的股票数量（AkShare 全市场较慢，可先调小验证流程）
MAX_STOCKS_UNIVERSE = int(os.environ.get("QUANT_MAX_STOCKS", "400"))

# AkShare HTTP 超时（秒）；超时或失败则跳过该股票，避免长时间卡死
AKSHARE_REQUEST_TIMEOUT = float(os.environ.get("QUANT_AK_TIMEOUT", "30"))

# 单次请求失败后的重试次数（网络抖动、接口短暂不可用）
AKSHARE_FETCH_RETRIES = max(1, int(os.environ.get("QUANT_FETCH_RETRIES", "3")))

# 重试间隔基数（秒），实际休眠约 sleep * (attempt + 1)
AKSHARE_FETCH_RETRY_SLEEP = float(os.environ.get("QUANT_FETCH_RETRY_SLEEP", "0.8"))

# 每日选股拉 K 线：只取最近若干自然日的数据即可算满窗口因子，减轻接口压力
PREDICT_HISTORY_CALENDAR_DAYS = int(os.environ.get("QUANT_PREDICT_LOOKBACK_DAYS", "800"))

# 选股阶段并发线程数（过大易被东方财富/AkShare 限流，导致「零条有效 K 线」）
PREDICT_FETCH_WORKERS = int(os.environ.get("QUANT_FETCH_WORKERS", "4"))

# AkShare 全部失败时是否用 Baostock 串行兜底（需安装 baostock；非线程安全故全局锁串行）
USE_BAOSTOCK_FALLBACK = os.environ.get("QUANT_BAOSTOCK_FALLBACK", "0") not in (
    "0",
    "false",
    "False",
)

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

# 特征列名（与 factor_calculator / 本地 K 线训练管线一致）
FEATURE_COLUMNS = [
    "factor_bias_5",
    "factor_bias_10",
    "factor_bias_20",
    "factor_bias_60",
    "factor_ratio_5_20",
    "factor_ratio_10_60",
    "factor_return_1d",
    "factor_return_5d",
    "factor_momentum_10d",
    "factor_volume_ratio",
    "factor_volume_position",
    "factor_volatility_5d",
    "factor_volatility_20d",
    "factor_rsi_14",
    "factor_wr_14",
    "factor_atr_14",
    "factor_size_mcap",
]
