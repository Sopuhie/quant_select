"""LightGBM 训练与模型落盘。"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, LGB_PARAMS, MODEL_PATH
from .data_fetcher import fetch_daily_hist, has_enough_history
from .database import register_model_version
from .factor_calculator import build_stock_panel_features


def calculate_ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
    """Spearman 秩相关，衡量因子与未来收益截面单调性。"""
    m = factor_values.notna() & future_returns.notna()
    if int(m.sum()) < 30:
        return float("nan")
    return float(factor_values[m].corr(future_returns[m], method="spearman"))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def calculate_ir(ic_series: pd.Series) -> float:
    """IR = IC 序列均值 / 标准差。"""
    s = ic_series.dropna()
    if len(s) < 2 or float(s.std()) < 1e-12:
        return float("nan")
    return float(s.mean() / s.std())


def collect_training_samples(
    stock_pairs: list[tuple[str, str]],
    train_end_date: str,
    start_date: str = "20180101",
) -> pd.DataFrame:
    """
    多只股票纵向合并训练样本；只保留 date <= train_end_date 的行。
    """
    parts: list[pd.DataFrame] = []
    end_compact = train_end_date.replace("-", "")
    for code, name in stock_pairs:
        hist = fetch_daily_hist(code, start_date=start_date, end_date=end_compact)
        if not has_enough_history(hist):
            continue
        panel = build_stock_panel_features(hist, code, name)
        panel = panel[panel["date"] <= train_end_date]
        if len(panel) > 0:
            parts.append(panel)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def train_lgbm_regressor(
    df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[LGBMRegressor, dict[str, Any]]:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["label_ret"])
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label_ret"].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[finite]
    y = y[finite]
    if len(y) < 100:
        raise RuntimeError(
            f"有效训练样本过少（{len(y)}），多为异常价或缺失导致；请换截止日期或增大股票池。"
        )
    X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    model = LGBMRegressor(**LGB_PARAMS, n_estimators=500)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )
    pred_val = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    ic_by_feature: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        ic_by_feature[col] = calculate_ic(X_val[col], pd.Series(y_val))
    ic_series = pd.Series(ic_by_feature)
    ir_val = calculate_ir(ic_series)
    metrics = {
        "rmse_val": rmse,
        "n_samples": int(len(y)),
        "ic_mean": float(ic_series.mean()) if ic_series.notna().any() else float("nan"),
        "ir_val": ir_val,
        "ic_by_feature": ic_by_feature,
    }
    return model, metrics


def save_model(model: LGBMRegressor, path: Path | None = None) -> Path:
    p = path or MODEL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    return p


def load_model(path: Path | None = None) -> LGBMRegressor:
    p = path or MODEL_PATH
    return joblib.load(p)


def train_and_register(
    stock_pairs: list[tuple[str, str]],
    train_end_date: str,
    version: str | None = None,
    model_path: Path | None = None,
) -> tuple[LGBMRegressor, dict[str, Any]]:
    if version is None:
        version = "v" + datetime.now().strftime("%Y%m%d.%H%M")
    raw = collect_training_samples(stock_pairs, train_end_date=train_end_date)
    if raw.empty:
        raise RuntimeError("训练样本为空，请检查网络、股票池与日期区间。")
    model, metrics = train_lgbm_regressor(raw)
    save_model(model, model_path)
    register_model_version(
        version=version,
        train_end_date=train_end_date,
        features=list(FEATURE_COLUMNS),
        metrics=_json_safe(metrics),
        set_active=True,
    )
    return model, {"version": version, **metrics, "rows": len(raw)}
