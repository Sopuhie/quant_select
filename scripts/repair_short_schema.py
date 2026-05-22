"""
修复短线表结构（旧库缺 final_score 或残留无效索引时使用）。

用法:
  python scripts/repair_short_schema.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DB_PATH
from src.short_term.db import ensure_short_term_tables


def main() -> int:
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        ensure_short_term_tables(conn)
        cols = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(short_daily_selections)"
            ).fetchall()
        ]
        print(f"数据库: {path}")
        print("short_daily_selections 列:", ", ".join(cols))
        if "final_score" not in cols:
            print("错误: 仍缺少 final_score 列", file=sys.stderr)
            return 1
        print("短线表结构修复完成。")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
