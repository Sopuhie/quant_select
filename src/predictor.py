"""加载模型、全市场打分、写库；更新历史选股收益。"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    AKSHARE_REQUEST_TIMEOUT,
    EXCLUDE_NEAR_LIMIT_LAST_BAR,
    EXCLUDE_ST,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NEAR_LIMIT_PCT_THRESHOLD,
    PREDICT_FETCH_WORKERS,
    TOP_N_SELECTION,
)
from .data_fetcher import (
    compact_start_calendar_days_ago,
    fetch_daily_hist,
    get_risk_st_stock_codes,
    get_stock_pool,
    has_enough_history,
)
from .database import (
    get_active_model_version,
    insert_daily_predictions,
    delete_daily_outputs_for_trade_date,
    insert_daily_selections,
    selection_exists_for_date,
)
from .factor_calculator import compute_factors_for_history
from .model_trainer import load_model


def _normalize_stock_display_name(stock_name: Optional[str]) -> str:
    if not stock_name:
        return ""
    s = str(stock_name).strip()
    s = s.replace("\u3000", "").replace("\xa0", "")
    # 全角星号等统一为 ASCII *（部分数据源用 ＊ 而非 *）
    for ch in ("＊", "\u2217", "\u22c6"):
        s = s.replace(ch, "*")
    return s


def is_st_stock_name(stock_name: Optional[str]) -> bool:
    """名称是否属于 ST / *ST 等（含全角＊、大小写混排）。"""
    n = _normalize_stock_display_name(stock_name)
    if not n:
        return False
    nu = n.upper()
    if any(k in nu for k in ("*ST", "S*ST", "SST")):
        return True
    if nu.startswith("ST"):
        return True
    if re.search(r"\*\s*S\s*T", nu):
        return True
    return False


def is_st_stock_row(stock_code: Any, stock_name: Optional[str]) -> bool:
    """名称规则 + 东方财富风险警示板代码。"""
    if is_st_stock_name(stock_name):
        return True
    code = str(stock_code).strip().zfill(6)
    if len(code) == 6 and code.isdigit() and code in get_risk_st_stock_codes():
        return True
    return False


def filter_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤名称中含 ST 风险标识的股票。
    使用字面子串匹配（regex=False），避免 '*ST' 在正则里有特殊含义。
    """
    if df.empty or "stock_name" not in df.columns:
        return df.copy() if len(df) else df
    st_keywords = ["*ST", "S*ST", "SST", "ST"]
    s = df["stock_name"].map(_normalize_stock_display_name).astype(str)
    hit = pd.Series(False, index=df.index)
    for kw in st_keywords:
        hit |= s.str.contains(kw, na=False, regex=False)
    return df[~hit].reset_index(drop=True)


def filter_predictions(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤打分结果：剔除 ST；可选剔除上一交易日接近涨跌停。
    scores_df 需含 stock_code、stock_name；可选列 pct_prev_day。
    """
    out = scores_df.copy()
    if EXCLUDE_ST:
        if "stock_code" in out.columns:
            mask = ~out.apply(
                lambda r: is_st_stock_row(r.get("stock_code"), r.get("stock_name")),
                axis=1,
            )
            out = out[mask]
        elif "stock_name" in out.columns:
            out = out[~out["stock_name"].map(is_st_stock_name)]
    if (
        EXCLUDE_NEAR_LIMIT_LAST_BAR
        and "pct_prev_day" in out.columns
        and len(out) > 0
    ):
        p = pd.to_numeric(out["pct_prev_day"], errors="coerce").abs()
        out = out[p < NEAR_LIMIT_PCT_THRESHOLD].copy()
    return out.reset_index(drop=True)


def latest_feature_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty or len(df) < 30:
        return None
    fac = compute_factors_for_history(df)
    if fac.empty:
        return None
    last_idx = len(df) - 1
    row = fac.iloc[last_idx]
    if row[FEATURE_COLUMNS].isna().any():
        return None
    out = row[FEATURE_COLUMNS].copy()
    out["date"] = df.iloc[last_idx]["date"]
    out["close"] = float(df.iloc[last_idx]["close"])
    return out


def _fetch_one_stock_features(
    code: str,
    name: str,
    start_compact: str,
    timeout: float,
) -> Optional[dict[str, Any]]:
    """仅拉数据算因子；在主线程里统一 predict，避免多线程共用模型。"""
    hist = fetch_daily_hist(code, start_date=start_compact, timeout=timeout)
    if not has_enough_history(hist):
        return None
    feat = latest_feature_row(hist)
    if feat is None:
        return None
    row: dict[str, Any] = {
        "trade_date": str(feat["date"]),
        "stock_code": code,
        "stock_name": name,
        "close_price": float(feat["close"]),
    }
    for c in FEATURE_COLUMNS:
        row[c] = float(feat[c])
    if len(hist) >= 2:
        c0 = float(hist.iloc[-2]["close"])
        c1 = float(hist.iloc[-1]["close"])
        row["pct_prev_day"] = (c1 / c0 - 1.0) if c0 > 1e-12 else float("nan")
    else:
        row["pct_prev_day"] = float("nan")
    return row


def predict_universe_scores(
    stock_pairs: list[tuple[str, str]],
    model_path: Path | None = None,
    max_workers: int | None = None,
    verbose: bool = True,
) -> tuple[str, pd.DataFrame]:
    """
    对股票池用最新一根 K 线计算因子并预测分数。
    并发拉取 K 线（缩短历史窗口），主线程一次性 predict，避免 sklearn 特征名告警与长时间无反馈。
    返回 (as_of_trade_date, DataFrame columns: stock_code, stock_name, score, close_price)
    """
    workers = int(max_workers if max_workers is not None else PREDICT_FETCH_WORKERS)
    workers = max(1, min(workers, 32))
    start_compact = compact_start_calendar_days_ago()
    timeout = AKSHARE_REQUEST_TIMEOUT
    model = load_model(model_path)

    raw_rows: list[dict[str, Any]] = []
    total = len(stock_pairs)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_fetch_one_stock_features, c, n, start_compact, timeout): (c, n)
            for c, n in stock_pairs
        }
        for fut in as_completed(futs):
            done += 1
            if verbose and done % 40 == 0:
                print(f"拉取进度: {done}/{total}", flush=True)
            try:
                r = fut.result()
            except Exception:
                continue
            if r:
                raw_rows.append(r)
    if not raw_rows:
        raise RuntimeError(
            f"未能拉取到任何有效 K 线（本次股票池共 {total} 只，全部未得到可用因子行）。"
            "已在 AkShare 失败后尝试 Baostock 兜底（默认开启，需 pip install baostock）。\n"
            "仍失败时请检查：① 网络 / 代理 ② "
            "set QUANT_AK_TIMEOUT=45 QUANT_FETCH_RETRIES=5 ③ "
            "set QUANT_FETCH_WORKERS=2 ④ "
            "set QUANT_MAX_STOCKS=100 ⑤ "
            "关闭兜底排查：set QUANT_BAOSTOCK_FALLBACK=0"
        )
    feat_df = pd.DataFrame(raw_rows)
    as_of = str(feat_df["trade_date"].max())
    feat_df = feat_df[feat_df["trade_date"] == as_of].reset_index(drop=True)
    if feat_df.empty:
        raise RuntimeError("过滤到统一交易日后样本为空。")

    feat_df = filter_predictions(feat_df)
    if EXCLUDE_ST:
        feat_df = filter_st_stocks(feat_df)
    if feat_df.empty:
        raise RuntimeError("经 ST/涨跌停过滤后无可用股票，请检查股票池或关闭部分过滤开关。")

    X = feat_df[FEATURE_COLUMNS].astype(np.float64)
    scores = model.predict(X)
    feat_df["score"] = scores.astype(float)
    cols = ["trade_date", "stock_code", "stock_name", "score", "close_price"]
    out = feat_df[cols].copy()
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return as_of, out


def get_full_market_predictions(
    stock_pairs: list[tuple[str, str]],
    model_path: Path | None = None,
    max_workers: int | None = None,
    verbose: bool = True,
) -> tuple[str, pd.DataFrame]:
    """全市场（给定股票池）打分别名，逻辑同 predict_universe_scores。"""
    return predict_universe_scores(
        stock_pairs,
        model_path=model_path,
        max_workers=max_workers,
        verbose=verbose,
    )


def persist_predictions(
    trade_date: str,
    scores_df: pd.DataFrame,
    top_n: int = TOP_N_SELECTION,
) -> None:
    delete_daily_outputs_for_trade_date(trade_date)
    scores_df = filter_predictions(scores_df.copy())
    if EXCLUDE_ST:
        scores_df = filter_st_stocks(scores_df)
    if scores_df.empty:
        raise RuntimeError("写入前过滤结果为空，无法保存预测与选股。")
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
    scores_df["rank_in_market"] = np.arange(1, len(scores_df) + 1)
    pred_rows = scores_df.to_dict("records")
    insert_daily_predictions(
        [
            {
                "trade_date": trade_date,
                "stock_code": r["stock_code"],
                "stock_name": r["stock_name"],
                "score": float(r["score"]),
                "rank_in_market": int(r["rank_in_market"]),
            }
            for r in pred_rows
        ]
    )
    top = scores_df.head(top_n).reset_index(drop=True)
    sel_rows = []
    for i, r in top.iterrows():
        sel_rows.append(
            {
                "trade_date": trade_date,
                "stock_code": r["stock_code"],
                "stock_name": r["stock_name"],
                "rank": int(i) + 1,
                "score": float(r["score"]),
                "close_price": float(r["close_price"]),
                "next_day_return": None,
                "hold_5d_return": None,
            }
        )
    insert_daily_selections(sel_rows)


def run_selection_for_latest(
    max_stocks: int,
    skip_if_exists: bool = True,
    model_path: Path | None = None,
    fetch_workers: int | None = None,
    verbose: bool = True,
    pool_as_of_date: str | None = None,
) -> dict[str, Any]:
    from datetime import datetime

    as_of = pool_as_of_date or datetime.now().strftime("%Y-%m-%d")
    pairs = get_stock_pool(as_of_date=as_of, max_count=max_stocks)
    if not pairs:
        raise RuntimeError("股票列表为空（请检查指数成份接口或网络）。")
    trade_date, df = predict_universe_scores(
        pairs,
        model_path=model_path,
        max_workers=fetch_workers,
        verbose=verbose,
    )
    if len(df) < TOP_N_SELECTION:
        raise RuntimeError(
            f"过滤后仅 {len(df)} 只股票，不足 Top{TOP_N_SELECTION}，可调大股票池或放宽过滤。"
        )
    if skip_if_exists and selection_exists_for_date(trade_date):
        return {"skipped": True, "trade_date": trade_date, "reason": "already_exists"}
    persist_predictions(trade_date, df)
    mv = get_active_model_version()
    return {
        "skipped": False,
        "trade_date": trade_date,
        "n_scored": len(df),
        "model_version": mv["version"] if mv else None,
    }


def backfill_selection_returns(
    max_stocks: int = 800,
) -> int:
    """兼容旧入口：转调统一回填逻辑。"""
    from .return_updater import update_all_returns

    _ = max_stocks
    out = update_all_returns()
    return int(out.get("rows_updated", 0))
