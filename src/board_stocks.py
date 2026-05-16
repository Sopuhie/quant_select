"""
概念板块→成分股映射（东方财富成份，JSON + 供 database 同步 SQLite）。
"""
from __future__ import annotations

import json
from datetime import datetime

from .config import DATA_DIR

BOARD_STOCKS_PATH = DATA_DIR / "board_stocks.json"


def save_board_mapping(
    mapping: dict[str, list[str]],
    merge: bool = True,
) -> None:
    """
    mapping: {"PLC概念": ["002979", ...], ...}
    merge=True: 与已有数据合并追加成份；merge=False: 对 mapping 中的板块整板覆盖。
    """
    BOARD_STOCKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    old_stock_to_boards: dict[str, list[str]] = {}
    old_boards: dict[str, list[str]] = {}
    if merge and BOARD_STOCKS_PATH.exists():
        try:
            with open(BOARD_STOCKS_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_stock_to_boards = old_data.get("stock_to_boards", {})
            old_boards = old_data.get("boards", {})
        except Exception:
            pass

    merged_boards: dict[str, list[str]] = {
        board: list(dict.fromkeys(codes)) for board, codes in old_boards.items()
    }
    for board, codes in mapping.items():
        fresh = [
            str(c).strip().zfill(6)
            for c in codes
            if str(c).strip().isdigit()
        ]
        if merge:
            existing = list(dict.fromkeys(merged_boards.get(board, [])))
            for c6 in fresh:
                if c6 not in existing:
                    existing.append(c6)
            merged_boards[board] = existing
        else:
            merged_boards[board] = list(dict.fromkeys(fresh))

    stock_to_boards: dict[str, list[str]] = {}
    for board, codes in merged_boards.items():
        for c in codes:
            c6 = str(c).strip().zfill(6)
            if c6 not in stock_to_boards:
                stock_to_boards[c6] = []
            if board not in stock_to_boards[c6]:
                stock_to_boards[c6].append(board)

    total_codes = sum(len(v) for v in merged_boards.values())
    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "eastmoney_akshare",
        "board_count": len(merged_boards),
        "stock_count": len(stock_to_boards),
        "stock_board_entries": total_codes,
        "boards": merged_boards,
        "stock_to_boards": stock_to_boards,
    }
    with open(BOARD_STOCKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    new_entries = sum(len(v) for v in mapping.values())
    print(
        f"[BoardStocks] {'Merged' if merge else 'Replaced'} {len(mapping)} boards "
        f"({new_entries} codes), total {len(merged_boards)} boards, "
        f"{len(stock_to_boards)} stocks → {BOARD_STOCKS_PATH}"
    )


def load_board_mapping() -> dict[str, list[str]]:
    """返回 stock_code → [board_names]。"""
    if not BOARD_STOCKS_PATH.exists():
        return {}
    with open(BOARD_STOCKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("stock_to_boards", {})


def get_stock_boards(stock_code: str) -> list[str]:
    mapping = load_board_mapping()
    return mapping.get(str(stock_code).zfill(6), [])


if __name__ == "__main__":
    m = load_board_mapping()
    print(f"{len(m)} stocks mapped")
