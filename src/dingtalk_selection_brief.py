"""
钉钉 Top3 推送附带的「简评速览」Markdown 模板（由信号日截面数据自动生成）。
"""
from __future__ import annotations

import re
from typing import Any

from .config import MARKET_REGIME_MIN_SCORE, MAX_ALLOWED_5D_RETURN
from .database import query_df
from .market_regime import compute_market_regime_score


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.1f}%"


def _md_line(text: str) -> str:
    """钉钉 Markdown：行尾两空格强制换行。"""
    return f"{text.rstrip()}  "


def _first_reason_line(reason: str, max_len: int = 18) -> str:
    s = str(reason or "").strip()
    if not s:
        return ""
    line = re.split(r"[；;\n]", s)[0].strip()
    line = re.sub(r"^\d+\.\s*", "", line)
    line = re.sub(r"\[factor_[^\]]+\]", "", line).strip()
    if len(line) > max_len:
        return line[: max_len - 1] + "…"
    return line


def _fmt_score(score: float | None) -> str:
    if score is None:
        return "—"
    return f"{float(score):.2f}"


def _stock_meta_lines(s: dict[str, Any]) -> list[str]:
    """指标拆成 1～2 行短句，避免 | 分列在手机上错位。"""
    mkt_rank = s.get("rank_in_market")
    score = s.get("score")
    close = s.get("close_price")
    row1: list[str] = []
    if mkt_rank is not None:
        row1.append(f"排名 #{mkt_rank}")
    if score is not None:
        row1.append(f"分 {_fmt_score(score)}")
    if close is not None:
        row1.append(f"收 {float(close):.2f}")
    row2: list[str] = []
    if s.get("ret5") is not None:
        row2.append(f"5日 {_pct(s.get('ret5'))}")
    if s.get("ret20") is not None:
        row2.append(f"20日 {_pct(s.get('ret20'))}")
    if s.get("day_chg") is not None:
        row2.append(f"信号日 {_pct(s.get('day_chg'))}")
    out: list[str] = []
    if row1:
        out.append(" · ".join(row1))
    if row2:
        out.append(" · ".join(row2))
    return out


def _market_regime_label(score: int | None) -> str:
    if score is None:
        return "未知"
    thr = int(MARKET_REGIME_MIN_SCORE)
    if score >= thr + 15:
        return "偏强"
    if score > thr:
        return "中性"
    if score == thr:
        return "中性偏弱（卡线）"
    return "偏弱（或已熔断）"


def _stock_verdict(
    *,
    rank: int,
    mkt_rank: int | None,
    ret5: float | None,
    ret20: float | None,
    day_chg: float | None,
) -> tuple[list[str], list[str], str]:
    pros: list[str] = []
    risks: list[str] = []
    if mkt_rank == 1:
        pros.append("模型全市场排名第 1，截面置信度最高")
    elif mkt_rank is not None and mkt_rank <= 10:
        pros.append(f"全市场排名第 {mkt_rank}，融合分靠前")
    elif mkt_rank is not None and mkt_rank > 15:
        risks.append(f"截面排名第 {mkt_rank}，三只中模型把握偏弱")

    if ret5 is not None:
        if ret5 <= 0.03:
            pros.append("近 5 日涨跌温和，更接近「蓄势」形态")
        elif ret5 >= 0.09:
            risks.append(f"近 5 日已涨 {_pct(ret5)}，偏趋势延续/追涨")
        if ret5 >= float(MAX_ALLOWED_5D_RETURN) * 0.9:
            risks.append(
                f"5 日涨幅接近系统可选上限 {MAX_ALLOWED_5D_RETURN * 100:.0f}%"
            )

    if ret20 is not None and ret20 >= 0.15:
        risks.append(f"近 20 日累计 {_pct(ret20)}，中期涨幅不小")

    if day_chg is not None and day_chg >= 0.04:
        pros.append(f"信号日单日 {_pct(day_chg)}，短线动能偏强")
    elif day_chg is not None and day_chg <= -0.03:
        risks.append(f"信号日单日 {_pct(day_chg)}，当日偏弱")

    if rank == 1 and (ret5 or 0) >= 0.08:
        one = "模型首选，但偏追涨，宜轻仓+紧止损"
    elif rank == 1:
        one = "模型首选，可主攻但需设止损"
    elif mkt_rank is not None and mkt_rank > 15:
        one = "偏防守/补涨角色，仓位宜小于 Rank1"
    elif (ret5 or 0) >= 0.08:
        one = "短线动能股，小仓参与，不宜与新仓重叠行业"
    else:
        one = "截面过关，可按计划仓位参与"

    if not pros:
        pros.append("已通过量化截面筛选进入 Top3")
    return pros, risks, one


def _portfolio_notes(selections: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    hot = sum(
        1
        for s in selections
        if (s.get("ret5") or 0) >= 0.08
    )
    if hot >= 2:
        notes.append(f"组合内 {hot} 只近 5 日涨幅≥8%，整体追高风险偏高")
    weak = [
        s
        for s in selections
        if (s.get("rank_in_market") or 999) > 15
    ]
    if weak:
        names = "、".join(str(s.get("stock_name") or s.get("stock_code")) for s in weak)
        notes.append(f"{names} 截面排名靠后，宜降低仓位或二选一")
    notes.append("单票设止损；行业/财报/公告请自行复核")
    return notes


def enrich_selections_for_brief(
    signal_trade_date: str,
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """为每只标的补充 rank_in_market、5/20 日收益、信号日涨跌幅等。"""
    td = str(signal_trade_date).strip()[:10]
    if not td or not selections:
        return [dict(s) for s in selections]

    codes = [str(s.get("stock_code", "")).strip().zfill(6) for s in selections]
    rank_map: dict[str, int] = {}
    try:
        ph = ",".join("?" * len(codes))
        df = query_df(
            f"""
            SELECT stock_code, rank_in_market
            FROM daily_predictions
            WHERE trade_date = ? AND stock_code IN ({ph})
            """,
            (td, *codes),
        )
        for _, row in df.iterrows():
            c = str(row["stock_code"]).strip().zfill(6)
            if row.get("rank_in_market") is not None:
                rank_map[c] = int(row["rank_in_market"])
    except Exception:
        pass

    out: list[dict[str, Any]] = []
    for s in selections:
        row = dict(s)
        row["signal_trade_date"] = td
        code = str(row.get("stock_code", "")).strip().zfill(6)
        row["rank_in_market"] = rank_map.get(code)
        row["ret5"] = None
        row["ret20"] = None
        row["day_chg"] = None
        try:
            kdf = query_df(
                """
                SELECT date, close FROM stock_daily_kline
                WHERE stock_code = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 21
                """,
                (code, td),
            )
            if len(kdf) >= 2:
                # 按 date DESC：closes[0]=信号日，closes[1]=上一交易日
                closes = kdf["close"].astype(float).tolist()
                row["day_chg"] = closes[0] / closes[1] - 1
                if len(closes) >= 6:
                    row["ret5"] = closes[0] / closes[5] - 1
                row["ret20"] = closes[0] / closes[-1] - 1
        except Exception:
            pass
        out.append(row)
    return out


def build_selection_brief_markdown(
    signal_trade_date: str,
    selections: list[dict[str, Any]],
    *,
    market_score: int | None = None,
) -> str:
    """生成「简评速览」Markdown 正文（钉钉：行尾两空格 + 换行）。"""
    td = str(signal_trade_date).strip()[:10]
    enriched = enrich_selections_for_brief(td, selections)

    if market_score is None:
        try:
            market_score = int(compute_market_regime_score(td))
        except Exception:
            market_score = None

    mkt_s = str(market_score) if market_score is not None else "—"
    regime = _market_regime_label(market_score)

    lines: list[str] = [
        _md_line("📊 简评速览"),
        _md_line(f"大盘 {mkt_s} 分 · {regime}（熔断线 {MARKET_REGIME_MIN_SCORE}）"),
        "",
    ]

    for s in enriched:
        try:
            rank = int(s.get("rank") or 0)
        except (TypeError, ValueError):
            rank = 0
        code = str(s.get("stock_code", "")).strip().zfill(6)
        name = str(s.get("stock_name", "")).strip()
        mkt_rank = s.get("rank_in_market")
        pros, risks, one = _stock_verdict(
            rank=rank,
            mkt_rank=mkt_rank,
            ret5=s.get("ret5"),
            ret20=s.get("ret20"),
            day_chg=s.get("day_chg"),
        )
        lines.append(_md_line(f"【{rank}】{name} {code}"))
        for meta in _stock_meta_lines(s):
            lines.append(_md_line(f"　{meta}"))
        reason = _first_reason_line(str(s.get("selection_reason") or ""))
        if reason:
            lines.append(_md_line(f"　因子 {reason}"))
        for p in pros[:1]:
            lines.append(_md_line(f"　✅ {p}"))
        for r in risks[:2]:
            lines.append(_md_line(f"　⚠️ {r}"))
        lines.append(_md_line(f"　💡 {one}"))
        lines.append("")

    lines.append(_md_line("📋 组合提示"))
    for note in _portfolio_notes(enriched):
        lines.append(_md_line(f"　· {note}"))
    lines.append("")

    return "\n".join(lines)
