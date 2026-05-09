"""
清理 daily_selections：每个交易日仅保留分数最高的 TOP_N 条，并重排 rank=1..n。

适用场景：历史版本中 UNIQUE(trade_date, stock_code) 允许同日多条不同股票，
多次运行 run_daily 会在同一 trade_date 下累积多行，导致界面出现多张「同名名次」卡片。

用法（在 quant_select 目录下）:
  python scripts/clean_duplicates.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DB_PATH, TOP_N_SELECTION


def _fetch_day(conn: sqlite3.Connection, trade_date: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT id, trade_date, stock_code, stock_name, rank, score,
               close_price, next_day_return, hold_5d_return, created_at
        FROM daily_selections
        WHERE trade_date = ?
        ORDER BY score DESC, id DESC
        """,
        conn,
        params=(trade_date,),
    )


def normalize_daily_selections(db_path: Path | None = None) -> dict[str, int]:
    """
    对每个交易日：只保留分数最高的 TOP_N_SELECTION 条，删除其余行，
    并按 score 降序将 rank 设为 1..n。
    """
    path = Path(db_path or DB_PATH)
    conn = sqlite3.connect(str(path))
    try:
        dates = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM daily_selections ORDER BY trade_date",
            conn,
        )["trade_date"].tolist()

        deleted_total = 0
        updated_days = 0

        for trade_date in dates:
            df = _fetch_day(conn, trade_date)
            if df.empty:
                continue

            cap = TOP_N_SELECTION
            keep = df.head(cap)
            keep_ids = set(int(i) for i in keep["id"])
            all_ids = set(int(i) for i in df["id"])
            del_ids = sorted(all_ids - keep_ids)

            if del_ids:
                placeholders = ",".join("?" * len(del_ids))
                conn.execute(
                    f"DELETE FROM daily_selections WHERE id IN ({placeholders})",
                    del_ids,
                )
                deleted_total += len(del_ids)

            remaining = _fetch_day(conn, trade_date)
            if remaining.empty:
                continue

            updated_days += 1
            for new_rank, rid in enumerate(remaining["id"].tolist(), start=1):
                conn.execute(
                    "UPDATE daily_selections SET rank = ? WHERE id = ?",
                    (new_rank, int(rid)),
                )

        conn.commit()
        return {"trade_dates_processed": len(dates), "rows_deleted": deleted_total, "days_updated": updated_days}
    finally:
        conn.close()


def main() -> None:
    print("数据库:", DB_PATH)
    print("=== 清理前（最近若干行）===")
    conn = sqlite3.connect(str(DB_PATH))
    before = pd.read_sql_query(
        """
        SELECT id, trade_date, rank, stock_code, stock_name, score
        FROM daily_selections
        ORDER BY trade_date DESC, rank ASC
        LIMIT 40
        """,
        conn,
    )
    conn.close()
    print(before.to_string(index=False))

    stats = normalize_daily_selections()
    print("\n=== 处理结果 ===")
    print(stats)

    conn = sqlite3.connect(str(DB_PATH))
    after = pd.read_sql_query(
        """
        SELECT id, trade_date, rank, stock_code, stock_name, score
        FROM daily_selections
        ORDER BY trade_date DESC, rank ASC
        LIMIT 40
        """,
        conn,
    )
    conn.close()
    print("\n=== 清理后（最近若干行）===")
    print(after.to_string(index=False))
    print("\n完成。")


if __name__ == "__main__":
    main()
