"""
一键全套量化工作流：按顺序执行行情同步 → 训练 → 每日选股 → 滚动回测 → 收益回填。
供 Streamlit「一键」按钮与定时调度调用；日志写入 ``system_logs``。
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from src.database import get_connection, insert_system_log

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pipeline_dingtalk_push(_emit: Callable[[str], None]) -> list[str]:
    """全套流程成功后：按库内最新 trade_date 触发一次钉钉（与 run_daily --skip-dingtalk 配对）。"""
    buf: list[str] = []

    def log(line: str) -> None:
        buf.append(line)
        _emit(line)

    try:
        from src.dingtalk_notifier import maybe_push_daily_selections
    except ImportError as exc:
        log(f"[Pipeline] 钉钉模块不可用，跳过推送: {exc}\n")
        return buf

    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM daily_selections WHERE trade_date IS NOT NULL"
            ).fetchone()
    except Exception as exc:
        log(f"[Pipeline] 读取选股日期失败，跳过钉钉: {exc}\n")
        return buf

    td = row[0] if row else None
    if not td:
        log("[Pipeline] daily_selections 无记录，跳过钉钉推送。\n")
        return buf

    td_str = str(td).strip()[:10]
    log(f"[Pipeline] 全套流程已完成，正在钉钉推送选股（trade_date={td_str}）...\n")
    try:
        ok = maybe_push_daily_selections(td_str)
    except Exception as exc:
        log(f"[Pipeline] 钉钉推送异常: {exc}\n")
        return buf

    if ok:
        log(f"[Pipeline] 钉钉推送已成功（{td_str}）。\n")
    else:
        log(
            "[Pipeline] 钉钉未发送或未成功（"
            f"{td_str}）。若未配置 Webhook 或未启用推送属正常；"
            "否则请查看运行 Streamlit 的终端日志。\n"
        )
    return buf


def is_today_trading_day() -> bool:
    """今日是否为 A 股交易日（新浪日历；失败时退回工作日近似）。"""
    from src.utils import is_a_share_trading_day

    return is_a_share_trading_day(datetime.now().strftime("%Y-%m-%d"))


def run_complete_pipeline(
    task_name: str = "自动全套工作流",
    *,
    log_consumer: Callable[[str], None] | None = None,
) -> bool:
    """
    顺序执行 A→E；任一步非零退出码则中止。
    ``log_consumer``：若提供则将每行输出回调（例如 Streamlit 占位符刷新）；否则打印到 stdout。
    """
    python_exe = sys.executable
    steps: list[dict[str, object]] = [
        {
            "name": "1. 增量行情与行业同步",
            "args": [python_exe, str(PROJECT_ROOT / "scripts" / "update_local_data.py")],
        },
        {
            "name": "2. 双排序模型重训 (LGBM+XGB)",
            "args": [python_exe, str(PROJECT_ROOT / "train_model.py")],
        },
        {
            "name": "3. 每日智能选股",
            "args": [python_exe, str(PROJECT_ROOT / "run_daily.py")],
        },
        {
            "name": "4. 历史滚动回测",
            "args": [python_exe, str(PROJECT_ROOT / "scripts" / "backtest.py")],
        },
        {
            "name": "5. 历史选股收益率回填",
            "args": [python_exe, str(PROJECT_ROOT / "scripts" / "update_returns.py")],
        },
    ]

    def _emit(line: str) -> None:
        if log_consumer is not None:
            log_consumer(line)
        else:
            sys.stdout.write(line)
            sys.stdout.flush()

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_log_accumulator: list[str] = []
    success = True
    run_daily_skipped_non_trading = False

    header = f"[Pipeline] 全套工作流开始 | {start_time} | cwd={PROJECT_ROOT}\n"
    _emit(header)
    full_log_accumulator.append(header)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.platform == "win32":
        env.setdefault("PYTHONUTF8", "1")

    train_end = os.environ.get("QUANT_TRAIN_END_DATE", "").strip()
    for step in steps:
        step_name = str(step["name"])
        cmd = list(step["args"])  # type: ignore[arg-type]
        if (
            train_end
            and len(cmd) >= 2
            and Path(cmd[1]).resolve() == (PROJECT_ROOT / "train_model.py").resolve()
        ):
            cmd = [*cmd, "--train-end-date", train_end[:10]]
        if (
            len(cmd) >= 2
            and Path(cmd[1]).resolve() == (PROJECT_ROOT / "run_daily.py").resolve()
        ):
            cmd = [*cmd, "--skip-dingtalk"]

        banner = f"\n—— 正在执行: {step_name} ——\n"
        _emit(banner)
        full_log_accumulator.append(banner)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            err = f"[Pipeline][ERROR] 无法启动子进程 {cmd}: {exc}\n"
            _emit(err)
            full_log_accumulator.append(err)
            success = False
            break

        assert proc.stdout is not None
        step_logs: list[str] = []
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                _emit(line)
                step_logs.append(line)

        returncode = proc.wait()
        step_log_str = "".join(step_logs)
        if (
            len(cmd) >= 2
            and Path(cmd[1]).resolve() == (PROJECT_ROOT / "run_daily.py").resolve()
            and "QUANT_RUN_DAILY_SKIPPED=non_trading_day" in step_log_str
        ):
            run_daily_skipped_non_trading = True
        summary = f"\n=== {step_name} 结束 (exit={returncode}) ===\n"
        full_log_accumulator.append(summary)
        full_log_accumulator.append(step_log_str)
        _emit(summary)

        if returncode != 0:
            fail = f"[Pipeline][FAIL] {step_name} 失败，Pipeline 中止。\n"
            _emit(fail)
            full_log_accumulator.append(fail)
            success = False
            break
        _emit(f"[Pipeline][OK] {step_name} 完成。\n")

    if success:
        if run_daily_skipped_non_trading:
            skip_msg = (
                "[Pipeline] 选股步骤因非交易日已跳过写入，"
                "本次不在流水线末尾触发钉钉推送。\n"
            )
            _emit(skip_msg)
            full_log_accumulator.append(skip_msg)
        else:
            full_log_accumulator.extend(_pipeline_dingtalk_push(_emit))

    status = "SUCCESS" if success else "FAILED"
    try:
        insert_system_log(
            task_name=task_name,
            status=status,
            parameters="Sequential Pipeline [A->B->C->D->E]",
            log_output="".join(full_log_accumulator),
        )
    except Exception as ex:
        _emit(f"写入 system_logs 失败: {ex}\n")

    if not success:
        try:
            from src.dingtalk_notifier import maybe_push_pipeline_failure_alert

            excerpt = "".join(full_log_accumulator)
            maybe_push_pipeline_failure_alert(task_name, excerpt)
        except Exception:
            pass

    return success


if __name__ == "__main__":
    ok = run_complete_pipeline(task_name="CLI pipeline")
    raise SystemExit(0 if ok else 1)
