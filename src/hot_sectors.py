"""
从东方财富抓取热门概念板块名称（AkShare / 定时任务 JSON）。
供 Streamlit「热门题材高爆选股」下拉使用。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR

HOT_SECTORS_PATH = DATA_DIR / "hot_sectors.json"


def build_default_tags() -> list[str]:
    return ["科技", "新能源", "电力", "机器人", "半导体"]


def save_tags(tags: list[str], metadata: dict | None = None) -> None:
    HOT_SECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    td = datetime.now().strftime("%Y-%m-%d")
    if metadata and metadata.get("trade_date"):
        td = str(metadata["trade_date"]).strip()[:10]
    data: dict = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": td,
        "source": "eastmoney_akshare",
        "tags": tags,
    }
    if metadata:
        data["metadata"] = metadata
    with open(HOT_SECTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[HotSectors] Saved {len(tags)} tags → {HOT_SECTORS_PATH}")


def load_hot_sectors_meta() -> dict:
    if not HOT_SECTORS_PATH.exists():
        return {}
    try:
        with open(HOT_SECTORS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_tags() -> list[str]:
    if not HOT_SECTORS_PATH.exists():
        return build_default_tags()
    try:
        with open(HOT_SECTORS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        tags = data.get("tags", [])
        return tags if tags else build_default_tags()
    except Exception:
        return build_default_tags()


if __name__ == "__main__":
    print(load_tags())
