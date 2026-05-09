"""
量化选股系统历史滚动回测（样本外 LightGBM 打分 + 周期性全额换仓）。

换仓规则（默认 holding_days = LABEL_HORIZON_DAYS）：
  每 N 个交易日执行一次：交易日 i 开盘时先卖出全部持仓，再按「上一交易日」收盘后信号
  得到的 TopN 以开盘价等权满仓买入（信号仅使用截至昨日收盘的历史数据，避免未来函数）。

用法（在项目根目录 quant_select/ 下）:
  python scripts/backtest.py --start-date 2024-01-01 --end-date 2024-12-31
  python scripts/backtest.py --holding-days 5 --buy-price open --sell-price open

说明:
  - stocks.db 默认无 OHLCV 表时会并发调用 AkShare/Baostock 拉取行情；亦可自建日线表供 _try_read_ohlc_from_sqlite 命中。
  - 净值序列默认写入 data/backtest_results.csv。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

from src.config import (  # noqa: E402
    DATA_DIR,
    DB_PATH,
    FEATURE_COLUMNS,
    LABEL_HORIZON_DAYS,
    MIN_HISTORY_BARS,
    MODEL_PATH,
    TOP_N_SELECTION,
    STOCK_POOL,
    MAX_STOCKS_UNIVERSE,
)
from src.data_fetcher import fetch_daily_hist, get_stock_pool  # noqa: E402
from src.database import get_active_model_version  # noqa: E402
from src.factor_calculator import compute_factors_for_history  # noqa: E402
from src.model_trainer import load_model  # noqa: E402
from src.predictor import filter_predictions  # noqa: E402

PriceKind = Literal["open", "close"]
TRADING_DAYS_PER_YEAR = 242


def _normalize_date(s: str) -> str:
    return str(s).strip()[:10]


def load_active_model() -> Any:
    """载入磁盘上的 LightGBM 模型（与每日选股一致）。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}（请先 train_model.py 训练并落盘）")
    return load_model(MODEL_PATH)


def _sqlite_table_names(db_path: Path) -> list[str]:
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


def _try_read_ohlc_from_sqlite(
    start_date: str,
    end_date: str,
    db_path: Path,
) -> pd.DataFrame | None:
    """
    若库中存在常见命名的 OHLCV 表则读取；列名兼容 date/trade_date、code/stock_code。
    """
    tables = _sqlite_table_names(db_path)
    candidates = [
        t for t in tables if any(k in t.lower() for k in ("daily", "bar", "ohlc", "kline", "quote"))
    ]
    for t in candidates:
        try:
            with sqlite3.connect(db_path) as conn:
                probe = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 1", conn)
                cols = {c.lower(): c for c in probe.columns}
                dc = cols.get("date") or cols.get("trade_date")
                cc = cols.get("stock_code") or cols.get("code") or cols.get("symbol")
                oc = cols.get("open")
                hc = cols.get("high")
                lc = cols.get("low")
                pc = cols.get("close")
                vc = cols.get("volume")
                nc = cols.get("stock_name") or cols.get("name")
                if not all([dc, cc, oc, hc, lc, pc, vc]):
                    continue
                qcols = [dc, cc, oc, hc, lc, pc, vc]
                if nc:
                    qcols.append(nc)
                sel = ", ".join(qcols)
                sql = f"SELECT {sel} FROM {t} WHERE {dc} >= ? AND {dc} <= ?"
                raw = pd.read_sql_query(sql, conn, params=(start_date, end_date))
        except Exception:
            continue
        if raw.empty:
            continue
        rename = {
            dc: "date",
            cc: "stock_code",
            oc: "open",
            hc: "high",
            lc: "low",
            pc: "close",
            vc: "volume",
        }
        if nc:
            rename[nc] = "stock_name"
        raw = raw.rename(columns=rename)
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        raw["stock_code"] = raw["stock_code"].astype(str).str.strip().str.zfill(6)
        for col in ("open", "high", "low", "close", "volume"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        if "stock_name" not in raw.columns:
            raw["stock_name"] = ""
        raw = raw.dropna(subset=["close"]).sort_values(["stock_code", "date"])
        print(f"已从 SQLite 表 `{t}` 读取 OHLCV：{len(raw)} 行")
        return raw.reset_index(drop=True)
    return None


def _fetch_one_stock(
    code: str,
    name: str,
    start_compact: str,
    end_compact: str,
) -> pd.DataFrame:
    hist = fetch_daily_hist(code, start_date=start_compact, end_date=end_compact)
    if hist.empty or len(hist) < MIN_HISTORY_BARS:
        return pd.DataFrame()
    hist = hist.copy()
    hist["stock_code"] = code
    hist["stock_name"] = name
    return hist


def fetch_backtest_data(
    start_date: str,
    end_date: str,
    stock_pairs: list[tuple[str, str]],
    db_path: Path | None = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    优先从 stocks.db 读取已落库的日线；若无合适表则并发调用 fetch_daily_hist。
    """
    path = db_path or DB_PATH
    buf_days = 420
    cal_start = (pd.Timestamp(start_date) - pd.Timedelta(days=buf_days)).strftime("%Y-%m-%d")
    iso_start = _normalize_date(cal_start).replace("-", "")
    iso_end = _normalize_date(end_date).replace("-", "")

    hit = _try_read_ohlc_from_sqlite(cal_start, end_date, path)
    if hit is not None and not hit.empty:
        codes = set(hit["stock_code"].unique())
        need = {c for c, _ in stock_pairs}
        if len(codes & need) >= max(10, len(need) // 10):
            return hit[hit["stock_code"].isin(need)].reset_index(drop=True)
        print(
            "库内 OHLCV 与当前股票池交集过少，将改为在线拉取补充。"
            "若需纯本地回测，请先批量写入日线或扩大库内股票覆盖。"
        )

    print(
        f"未使用或未命中本地 OHLCV 宽表，正在在线拉取 {len(stock_pairs)} 只股票日线 "
        f"（{iso_start} ~ {iso_end}），请耐心等待…"
    )
    parts: list[pd.DataFrame] = []
    workers = max(1, min(int(max_workers), 16))
    done = 0
    total = len(stock_pairs)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_fetch_one_stock, c, n, iso_start, iso_end): (c, n)
            for c, n in stock_pairs
        }
        for fut in as_completed(futs):
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  行情进度: {done}/{total}", flush=True)
            try:
                df = fut.result()
            except Exception:
                continue
            if df is not None and not df.empty:
                parts.append(df)
    if not parts:
        raise RuntimeError(
            "未能获取任何股票的足够历史 K 线。请检查网络、Baostock/AkShare，"
            "或缩小 --max-stocks / 缩短区间；也可将日线导入 SQLite 供读取。"
        )
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _feature_dict_as_of(
    hist: pd.DataFrame,
    as_of_date: str,
    code: str,
    name: str,
) -> dict[str, Any] | None:
    """截至 as_of_date（含）的切片上计算最后一根 K 线的因子；严禁使用未来数据。"""
    sub = hist[hist["date"] <= as_of_date].reset_index(drop=True)
    if len(sub) < MIN_HISTORY_BARS:
        return None
    fac = compute_factors_for_history(sub)
    if fac.empty:
        return None
    last_i = len(sub) - 1
    row = fac.iloc[last_i]
    if row[list(FEATURE_COLUMNS)].isna().any():
        return None
    out: dict[str, Any] = {c: float(row[c]) for c in FEATURE_COLUMNS}
    out["trade_date"] = as_of_date
    out["stock_code"] = code
    out["stock_name"] = name
    out["close_price"] = float(sub.iloc[last_i]["close"])
    if len(sub) >= 2:
        c0 = float(sub.iloc[-2]["close"])
        c1 = float(sub.iloc[-1]["close"])
        out["pct_prev_day"] = (c1 / c0 - 1.0) if c0 > 1e-12 else float("nan")
    else:
        out["pct_prev_day"] = float("nan")
    return out


def _score_universe_for_date(
    trade_date: str,
    by_code: dict[str, pd.DataFrame],
    universe: list[tuple[str, str]],
    model: Any,
) -> list[str]:
    rows: list[dict[str, Any]] = []
    for code, name in universe:
        hist = by_code.get(code)
        if hist is None:
            continue
        r = _feature_dict_as_of(hist, trade_date, code, name)
        if r:
            rows.append(r)
    if not rows:
        return []
    feat_df = pd.DataFrame(rows)
    feat_df = filter_predictions(feat_df)
    if feat_df.empty:
        return []
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    if feat_df.empty:
        return []
    X = feat_df[FEATURE_COLUMNS].astype(np.float64)
    scores = model.predict(X.values)
    feat_df = feat_df.copy()
    feat_df["score"] = scores.astype(float)
    feat_df = feat_df.sort_values("score", ascending=False).reset_index(drop=True)
    return feat_df.head(TOP_N_SELECTION)["stock_code"].astype(str).tolist()


def _price_on(
    row: pd.Series,
    kind: PriceKind,
) -> float:
    k = "open" if kind == "open" else "close"
    v = float(row[k])
    if not np.isfinite(v) or v <= 0:
        return float("nan")
    return v


def fetch_benchmark_close(
    start_date: str,
    end_date: str,
    bench: str = "000300",
) -> pd.DataFrame:
    """沪深300等指数收盘价序列，用于超额收益；失败则返回空表。"""
    import akshare as ak

    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    df = pd.DataFrame()
    try:
        if bench == "000300":
            sym = "000300"
            tmp = ak.index_zh_a_hist(symbol=sym, period="daily", start_date=s, end_date=e)
            if tmp is not None and not tmp.empty:
                df = tmp
    except Exception:
        pass
    if df.empty:
        try:
            tmp = ak.stock_zh_index_daily_em(symbol="沪深300")
            if tmp is not None and not tmp.empty:
                df = tmp
        except Exception:
            pass
    if df.empty:
        print("警告: 未能获取基准指数行情，跳过超额收益对比。")
        return pd.DataFrame()

    date_col = next(
        (c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"),
        df.columns[0],
    )
    close_col = next(
        (
            c
            for c in df.columns
            if "收盘" in str(c) or str(c).lower() == "close" or str(c) == "收盘"
        ),
        None,
    )
    if close_col is None:
        return pd.DataFrame()
    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
    return out.sort_values("date").reset_index(drop=True)


@dataclass
class HeldShare:
    code: str
    shares: float
    cost_cash: float


@dataclass
class BacktestEngine:
    bars: pd.DataFrame
    trade_dates: list[str]
    model: Any
    initial_cash: float
    holding_days: int
    buy_price: PriceKind
    sell_price: PriceKind
    buy_fee_rate: float
    sell_fee_rate: float
    get_universe: Callable[[str], list[tuple[str, str]]]

    cash: float = field(init=False)
    date_pos: dict[str, dict[str, int]] = field(default_factory=dict)
    by_code: dict[str, pd.DataFrame] = field(default_factory=dict)
    positions: list[HeldShare] = field(default_factory=list)
    nav_records: list[dict[str, Any]] = field(default_factory=list)
    trade_pnls: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def _row(self, d: str, code: str) -> pd.Series | None:
        pos = self.date_pos.get(code, {}).get(d)
        if pos is None:
            return None
        return self.by_code[code].iloc[pos]

    def run(self) -> None:
        self.by_code = {
            c: g.reset_index(drop=True)
            for c, g in self.bars.groupby("stock_code", sort=False)
        }
        codes_list = list(self.by_code.keys())
        self.date_pos = {}
        for c in codes_list:
            g = self.by_code[c]
            self.date_pos[c] = {str(g.iloc[i]["date"]): i for i in range(len(g))}

        hd = max(1, int(self.holding_days))
        for i, d in enumerate(self.trade_dates):
            # 每 holding_days 个交易日全额换仓：T-1 收盘后信号 → T 日开盘先卖清再等额买入 TopN
            if i >= 1 and (i - 1) % hd == 0:
                sig_date = self.trade_dates[i - 1]
                self._sell_all_open(d)
                univ = self.get_universe(sig_date)
                codes = _score_universe_for_date(
                    sig_date, self.by_code, univ, self.model
                )
                self._buy_equal_open(d, codes)

            self._mark_to_market_close(d)

    def _sell_all_open(self, d: str) -> None:
        if not self.positions:
            return
        total_cost = sum(p.cost_cash for p in self.positions)
        proceeds_sum = 0.0
        for pos in self.positions:
            row = self._row(d, pos.code)
            if row is None:
                continue
            px = _price_on(row, self.sell_price)
            if not np.isfinite(px) or px <= 0:
                continue
            gross = pos.shares * px
            fee = gross * self.sell_fee_rate
            proceeds_sum += gross - fee
        self.trade_pnls.append(float(proceeds_sum - total_cost))
        self.cash += proceeds_sum
        self.positions = []

    def _buy_equal_open(self, d: str, codes: list[str]) -> None:
        codes = [c for c in codes if c in self.by_code]
        if not codes or self.cash <= 0:
            return
        per = self.cash / len(codes)
        spent_total = 0.0
        new_holdings: list[HeldShare] = []
        for code in codes:
            row = self._row(d, code)
            if row is None:
                continue
            px = _price_on(row, self.buy_price)
            if not np.isfinite(px) or px <= 0:
                continue
            alloc = min(per, max(0.0, self.cash - spent_total))
            if alloc <= 0:
                break
            gross_target = alloc / (1 + self.buy_fee_rate)
            shares = gross_target / px
            fee = gross_target * self.buy_fee_rate
            cash_use = gross_target + fee
            if cash_use > self.cash - spent_total + 1e-9:
                continue
            spent_total += cash_use
            new_holdings.append(
                HeldShare(code=code, shares=float(shares), cost_cash=float(cash_use))
            )
        self.cash -= spent_total
        self.positions.extend(new_holdings)

    def _mark_to_market_close(self, d: str) -> None:
        mv = 0.0
        for lot in self.positions:
            row = self._row(d, lot.code)
            if row is None:
                continue
            c = float(row["close"])
            if np.isfinite(c) and c > 0:
                mv += lot.shares * c
        nav = self.cash + mv
        self.nav_records.append(
            {
                "trade_date": d,
                "nav": float(nav),
                "cash": float(self.cash),
                "hold_mv": float(mv),
                "n_positions": len(self.positions),
            }
        )


def _compute_metrics(
    nav_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    trade_pnls: list[float],
    rf_annual: float,
) -> dict[str, Any]:
    nav_df = nav_df.sort_values("trade_date").reset_index(drop=True)
    nav = nav_df["nav"].astype(float)
    daily_ret = nav.pct_change().fillna(0.0)
    equity = nav / nav.iloc[0]
    cum_ret = float(equity.iloc[-1] - 1.0)
    n = len(nav_df)
    ann_factor = TRADING_DAYS_PER_YEAR / max(n - 1, 1)
    ann_ret = float((equity.iloc[-1] ** ann_factor) - 1.0) if n > 1 else cum_ret

    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    trough_i = int(dd.idxmin())
    peak_before = equity.iloc[: trough_i + 1].idxmax()
    dd_start = str(nav_df.iloc[int(peak_before)]["trade_date"])
    dd_end = str(nav_df.iloc[trough_i]["trade_date"])

    rf_d = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = daily_ret - rf_d
    std_d = float(daily_ret.std(ddof=1))
    sharpe = (
        float(excess.mean() / std_d * np.sqrt(TRADING_DAYS_PER_YEAR))
        if std_d > 1e-12
        else float("nan")
    )

    win_daily = float((daily_ret.iloc[1:] > 0).mean()) if n > 2 else float("nan")
    win_trade = (
        float(np.mean([1.0 if x > 0 else 0.0 for x in trade_pnls]))
        if trade_pnls
        else float("nan")
    )

    alpha_note = ""
    bench_cum = float("nan")
    excess_cum = float("nan")
    if bench_df is not None and not bench_df.empty:
        b = bench_df.sort_values("date").reset_index(drop=True)
        b = b[(b["date"] >= nav_df["trade_date"].iloc[0]) & (b["date"] <= nav_df["trade_date"].iloc[-1])]
        if len(b) >= 2:
            bench_equity = (b["close"].astype(float) / float(b["close"].iloc[0])).reset_index(drop=True)
            # 对齐交易日：按日期 merge
            m = pd.merge(
                nav_df[["trade_date", "nav"]],
                b.rename(columns={"date": "trade_date", "close": "bench_close"}),
                on="trade_date",
                how="inner",
            )
            if len(m) >= 2:
                be = m["bench_close"].astype(float)
                bench_equity_aligned = be / be.iloc[0]
                strat_equity_aligned = m["nav"].astype(float) / float(m["nav"].iloc[0])
                bench_cum = float(bench_equity_aligned.iloc[-1] - 1.0)
                excess_cum = float(strat_equity_aligned.iloc[-1] - bench_equity_aligned.iloc[-1])
                alpha_note = "（终点对齐超额：策略累计 - 沪深300累计）"

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "max_drawdown": mdd,
        "drawdown_interval": (dd_start, dd_end),
        "sharpe_ratio": sharpe,
        "win_rate_daily": win_daily,
        "win_rate_trades": win_trade,
        "n_trades": len(trade_pnls),
        "benchmark_cumulative_return": bench_cum,
        "excess_cumulative_return": excess_cum,
        "alpha_note": alpha_note,
        "nav_df": nav_df,
        "daily_ret": daily_ret,
    }


def _print_report(m: dict[str, Any]) -> None:
    dd0, dd1 = m["drawdown_interval"]
    print("\n" + "=" * 56)
    print("回测结果摘要")
    print("=" * 56)
    print(f"累计收益率:           {m['cumulative_return'] * 100:>10.2f} %")
    print(f"年化收益率 ({TRADING_DAYS_PER_YEAR} 日): {m['annualized_return'] * 100:>10.2f} %")
    print(f"最大回撤:             {m['max_drawdown'] * 100:>10.2f} %")
    print(f"回撤区间:             {dd0} ~ {dd1}")
    print(f"夏普比率 (Rf≈年化):   {m['sharpe_ratio']:>10.4f}")
    print(f"胜率（按交易日）:     {m['win_rate_daily'] * 100:>10.2f} %" if np.isfinite(m["win_rate_daily"]) else "胜率（按交易日）:          N/A")
    tr = m["win_rate_trades"]
    print(
        f"胜率（按平仓笔数）:   {tr * 100:>10.2f} % ({m['n_trades']} 笔)"
        if np.isfinite(tr)
        else f"胜率（按平仓笔数）:   N/A ({m['n_trades']} 笔)"
    )
    if np.isfinite(m["benchmark_cumulative_return"]):
        print(f"基准累计收益:         {m['benchmark_cumulative_return'] * 100:>10.2f} %")
        print(f"超额收益 (累计):      {m['excess_cumulative_return'] * 100:>10.2f} % {m['alpha_note']}")
    print("=" * 56 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="量化选股历史滚动回测")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument(
        "--holding-days",
        type=int,
        default=LABEL_HORIZON_DAYS,
        help=f"换仓周期（交易日），默认与训练标签 horizon 一致：{LABEL_HORIZON_DAYS}",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--buy-price", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--sell-price", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--buy-fee-rate", type=float, default=0.0001, help="买入单边费率（例万1）")
    parser.add_argument("--sell-fee-rate", type=float, default=0.0011, help="卖出单边费率（例千1.1）")
    parser.add_argument("--rf-annual", type=float, default=0.03, help="无风险利率年化，用于夏普")
    parser.add_argument("--max-stocks", type=int, default=MAX_STOCKS_UNIVERSE)
    parser.add_argument("--pool", type=str, default=STOCK_POOL, help="hs300 | zz500 | all")
    parser.add_argument("--workers", type=int, default=4, help="在线拉行情并发数")
    parser.add_argument("--benchmark", type=str, default="000300", help="基准指数代码（默认沪深300）")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "backtest_results.csv"),
        help="净值曲线输出路径",
    )
    args = parser.parse_args()

    start_date = _normalize_date(args.start_date)
    end_date = _normalize_date(args.end_date)

    mv = get_active_model_version()
    if mv is None:
        print("提示: model_versions 中无 is_active=1 记录，仍使用磁盘模型:", MODEL_PATH)
    else:
        print("当前激活模型版本:", mv.get("version"), "train_end_date=", mv.get("train_end_date"))

    model = load_active_model()

    # 股票池：取回测区间内「尽可能覆盖」——按结束日成分 + 上限
    pairs = get_stock_pool(as_of_date=end_date, pool_type=args.pool, max_count=args.max_stocks)
    if not pairs:
        raise SystemExit("股票池为空，请检查 pool 与网络（AkShare）。")

    bars = fetch_backtest_data(
        start_date,
        end_date,
        pairs,
        db_path=DB_PATH,
        max_workers=args.workers,
    )

    all_days = sorted(bars["date"].unique())
    trade_dates = [d for d in all_days if start_date <= d <= end_date]
    if len(trade_dates) < args.holding_days + 5:
        raise SystemExit("回测区间内有效交易日过少，请放宽日期或检查行情完整性。")

    # 与拉取行情使用同一股票池快照（按 end_date 选成份）。逐日 get_stock_pool(as_of=当日) 在指数源
    # 仅提供「最新变更日」一行成份时，会导致回测前期全日空池、无法建仓。
    def universe_fn(_d: str) -> list[tuple[str, str]]:
        return pairs

    engine = BacktestEngine(
        bars=bars,
        trade_dates=trade_dates,
        model=model,
        initial_cash=args.initial_cash,
        holding_days=args.holding_days,
        buy_price=args.buy_price,
        sell_price=args.sell_price,
        buy_fee_rate=args.buy_fee_rate,
        sell_fee_rate=args.sell_fee_rate,
        get_universe=universe_fn,
    )
    engine.run()

    nav_df = pd.DataFrame(engine.nav_records)
    bench_df = fetch_benchmark_close(start_date, end_date, bench=args.benchmark)
    metrics = _compute_metrics(nav_df, bench_df, engine.trade_pnls, rf_annual=args.rf_annual)
    _print_report(metrics)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nav_out = metrics["nav_df"].copy()
    nav_out["daily_return"] = metrics["daily_ret"].values
    if bench_df is not None and not bench_df.empty:
        nav_out = pd.merge(
            nav_out,
            bench_df.rename(columns={"date": "trade_date", "close": "benchmark_close"}),
            on="trade_date",
            how="left",
        )
    nav_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"净值序列已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()
