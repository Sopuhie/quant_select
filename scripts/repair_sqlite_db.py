# -*- coding: utf-8 -*-
"""
检查并修复损坏的 data/stocks.db。

用法:
  python scripts/repair_sqlite_db.py          # 仅检查
  python scripts/repair_sqlite_db.py --repair # 备份损坏库并替换为可读副本
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DB_PATH  # noqa: E402
from src.database import (  # noqa: E402
    _backup_corrupt_db_files,
    repair_sqlite_db,
    sqlite_quick_check_ok,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="检查/修复 stocks.db")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="损坏时备份源库并将 *.db.repaired 覆盖为 stocks.db",
    )
    args = parser.parse_args()

    if not DB_PATH.is_file():
        print(f"数据库不存在: {DB_PATH}")
        return 1

    ok = sqlite_quick_check_ok(DB_PATH)
    print(f"{DB_PATH} ({DB_PATH.stat().st_size} bytes) quick_check={'ok' if ok else 'FAIL'}")
    if ok:
        return 0
    if not args.repair:
        print("库已损坏。请加 --repair 执行修复，或删除 QUANT_DB_NO_AUTO_REPAIR 后重启应用自动修复。")
        return 2

    backup = _backup_corrupt_db_files(DB_PATH)
    print(f"已备份损坏库 -> {backup}")
    repaired = repair_sqlite_db(DB_PATH)
    print(f"已生成修复库 -> {repaired} ({repaired.stat().st_size} bytes)")
    for suffix in ("-wal", "-shm", "-journal"):
        side = Path(str(DB_PATH) + suffix)
        if side.is_file():
            side.unlink()
    shutil.move(str(repaired), str(DB_PATH))
    if not sqlite_quick_check_ok(DB_PATH):
        print("修复后 quick_check 仍失败")
        return 3
    print("修复完成，quick_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
