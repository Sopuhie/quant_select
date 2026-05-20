# -*- coding: utf-8 -*-
"""板块过滤：与 ``run_daily`` 任务 C 一致（默认剔除 300/301、688）。"""
from __future__ import annotations


def board_allowed(
    code: str,
    *,
    include_300: bool = False,
    include_688: bool = False,
) -> bool:
    """未勾选 ``include_300`` / ``include_688`` 时剔除创业板 300/301 与科创板 688。"""
    c = str(code).strip().zfill(6)
    if not include_300 and (c.startswith("300") or c.startswith("301")):
        return False
    if not include_688 and c.startswith("688"):
        return False
    return True
