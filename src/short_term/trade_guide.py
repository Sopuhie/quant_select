# -*- coding: utf-8 -*-
"""短线选股可操作价位指南：由信号收盘价推算 T+1 买入区间与 T+2 止盈/止损/持有条件。"""
from __future__ import annotations

from typing import Any

from .config import (
    SHORT_CLOSE_STOP_RATIO,
    SHORT_ENTRY_DIP_OPEN_THRESHOLD,
    SHORT_ENTRY_MAX_CHASE,
    SHORT_ENTRY_MIN_GAP,
    SHORT_MEDIOCRE_STOP_RATIO,
    SHORT_SELL_OFFSET,
    SHORT_T1_STRONG_CLOSE_PCT,
    SHORT_T1_TAKE_PROFIT_TIER1_PCT,
    SHORT_T1_TAKE_PROFIT_TIER2_LOCK,
    SHORT_T1_TAKE_PROFIT_TIER2_PCT,
)
from .execution import (
    mediocre_stop_trigger_price,
    stop_loss_trigger_price,
)


def _px(v: float) -> float:
    return round(float(v), 2)


def _pct_str(v: float) -> str:
    return f"{float(v) * 100:.1f}%"


def _levels_for_buy(buy: float) -> dict[str, float]:
    buy = float(buy)
    return {
        "ref_buy": _px(buy),
        "tp_tier1_high": _px(buy * (1.0 + SHORT_T1_TAKE_PROFIT_TIER1_PCT)),
        "tp_tier2_high": _px(buy * (1.0 + SHORT_T1_TAKE_PROFIT_TIER2_PCT)),
        "tp_tier2_sell": _px(buy * (1.0 + SHORT_T1_TAKE_PROFIT_TIER2_LOCK)),
        "stop_mediocre_close": _px(mediocre_stop_trigger_price(buy)),
        "stop_strong_close": _px(stop_loss_trigger_price(buy)),
        "t1_strong_close": _px(buy * (1.0 + SHORT_T1_STRONG_CLOSE_PCT)),
    }


def build_trade_action_guide(
    signal_close: float,
    *,
    sell_offset: int | None = None,
) -> dict[str, Any]:
    """
    由信号日收盘价生成 T+1 买入与 T+2 卖出/持有操作指南（绝对价格，可直接挂单参考）。

    说明：T+2 各阈值按「参考买入价」推算；实际成交以 T+1 真实买入价为准，
    可在拿到买入价后对阈值同比例缩放（或重新调用本函数传入预估买入价）。
    """
    sig = float(signal_close)
    if sig <= 0:
        return {"valid": False, "reason": "invalid_signal_close"}

    offset = int(sell_offset if sell_offset is not None else SHORT_SELL_OFFSET)
    offset = max(1, min(2, offset))

    open_lo = sig * (1.0 + float(SHORT_ENTRY_MIN_GAP))
    open_hi = sig * (1.0 + float(SHORT_ENTRY_MAX_CHASE))
    dip_hi = sig * (1.0 + float(SHORT_ENTRY_DIP_OPEN_THRESHOLD))

    ref = _levels_for_buy(sig)
    ref_lo = _levels_for_buy(open_lo)
    ref_hi = _levels_for_buy(open_hi)

    t1_lines = [
        f"有效开盘 { _px(open_lo):.2f}~{_px(open_hi):.2f} 元"
        f"（信号收 {_px(sig):.2f} 的 {_pct_str(SHORT_ENTRY_MIN_GAP)}~{_pct_str(SHORT_ENTRY_MAX_CHASE)}）",
        f"放弃：< {_px(open_lo):.2f} 或 > {_px(open_hi):.2f} 元",
        f"微高开 ≤{_px(dip_hi):.2f} 元 → 买价=开盘价",
        f"高开 {_px(dip_hi):.2f}~{_px(open_hi):.2f} 元 → 买价=(开盘+当日最低)/2",
    ]

    t2_lines = [
        f"【参考买价 {_px(sig):.2f} 元】T+2 最早可卖（T+1 不可卖）",
        f"① 冲高≥{_px(ref['tp_tier1_high']):.2f} 元（+{_pct_str(SHORT_T1_TAKE_PROFIT_TIER1_PCT)}）"
        f" → 动态止盈≈(高+收)/2",
        f"② 冲高≥{_px(ref['tp_tier2_high']):.2f} 元（+{_pct_str(SHORT_T1_TAKE_PROFIT_TIER2_PCT)}）"
        f" → 锁定卖 {_px(ref['tp_tier2_sell']):.2f} 元（+{_pct_str(SHORT_T1_TAKE_PROFIT_TIER2_LOCK)}）",
        f"③ 盘中最高<{_pct_str(SHORT_T1_TAKE_PROFIT_TIER2_PCT)} 且收盘<{_px(ref['stop_mediocre_close']):.2f} 元"
        f" → 平庸止损（-{_pct_str(SHORT_MEDIOCRE_STOP_RATIO)}）",
        f"④ 曾冲高≥{_pct_str(SHORT_T1_TAKE_PROFIT_TIER2_PCT)} 且收盘<{_px(ref['stop_strong_close']):.2f} 元"
        f" → 强势止损（-{_pct_str(SHORT_CLOSE_STOP_RATIO)}）",
    ]
    if offset >= 2:
        t2_lines.append(
            f"⑤ 持有骑乘：T+1 收≥{_px(ref['t1_strong_close']):.2f} 元（+{_pct_str(SHORT_T1_STRONG_CLOSE_PCT)}）"
            f" 且 T+2 收 > T+1 收 → T+2 收盘价卖"
        )
    t2_lines.append("⑥ 其余未触发 → T+2 收盘价卖")

    range_note = (
        f"若买价在 {_px(open_lo):.2f}~{_px(open_hi):.2f} 元之间，"
        f"止盈1触发高 {_px(ref_lo['tp_tier1_high']):.2f}~{_px(ref_hi['tp_tier1_high']):.2f}，"
        f"平庸止损收<{_px(ref_lo['stop_mediocre_close']):.2f}~{_px(ref_hi['stop_mediocre_close']):.2f}"
    )

    summary_short = (
        f"T+1开 {_px(open_lo):.2f}~{_px(open_hi):.2f} | "
        f"T+2止盈≥{_px(ref['tp_tier1_high']):.2f}/锁{_px(ref['tp_tier2_sell']):.2f} | "
        f"止损<{_px(ref['stop_mediocre_close']):.2f}"
    )

    return {
        "valid": True,
        "signal_close": _px(sig),
        "sell_offset": offset,
        "t1": {
            "open_valid_lo": _px(open_lo),
            "open_valid_hi": _px(open_hi),
            "dip_open_hi": _px(dip_hi),
            "skip_below": _px(open_lo),
            "skip_above": _px(open_hi),
            "lines": t1_lines,
        },
        "t2_ref": ref,
        "t2_ref_lo": ref_lo,
        "t2_ref_hi": ref_hi,
        "t2": {
            "lines": t2_lines,
            "range_note": range_note,
        },
        "summary_short": summary_short,
        "summary_text": "\n".join(
            ["【T+1 买入】"] + [f"  · {ln}" for ln in t1_lines]
            + ["", "【T+2 卖出/持有】"] + [f"  · {ln}" for ln in t2_lines]
            + ["", f"  · {range_note}"]
        ),
        "display": {
            "T+1开盘区间": f"{_px(open_lo):.2f}~{_px(open_hi):.2f} 元",
            "T+1放弃条件": f"<{_px(open_lo):.2f} 或 >{_px(open_hi):.2f} 元",
            "T+2止盈触发": (
                f"高≥{_px(ref['tp_tier1_high']):.2f} 动态止盈；"
                f"高≥{_px(ref['tp_tier2_high']):.2f} 锁{_px(ref['tp_tier2_sell']):.2f}"
            ),
            "T+2止损线": (
                f"平庸收<{_px(ref['stop_mediocre_close']):.2f}；"
                f"强势收<{_px(ref['stop_strong_close']):.2f}"
            ),
            "T+2持有条件": (
                f"T+1收≥{_px(ref['t1_strong_close']):.2f} 且 T+2收>T+1收 → 骑乘"
                if offset >= 2
                else "SELL_OFFSET=1：未触发止盈/止损则 T+2 收盘卖"
            ),
            "操作要点": summary_short,
        },
    }
