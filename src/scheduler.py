"""
交易日本地时间 20:00 触发全套工作流（依赖 ``schedule`` 库）。
仅在进程内启动一条守护线程；重复调用 ``start_background_scheduler`` 无效。
"""
from __future__ import annotations

import threading
import time

import schedule

from src.pipeline import is_today_trading_day, run_complete_pipeline

_lock = threading.Lock()


def scheduled_job() -> None:
    from datetime import datetime

    print(
        f"[A-Quant] 定时任务触发: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    if is_today_trading_day():
        print("[A-Quant] 今日为交易日，启动自动全套工作流...", flush=True)
        ok = run_complete_pipeline(task_name="定时全套工作流(20:00)")
        if ok:
            print("[A-Quant] 定时工作流执行成功。", flush=True)
        else:
            print("[A-Quant] 定时工作流中途失败，请查看 system_logs。", flush=True)
    else:
        print("[A-Quant] 今日非交易日，跳过工作流。", flush=True)


def _scheduler_thread_alive() -> bool:
    return any(
        getattr(t, "name", "") == "QuantSchedulerThread" and t.is_alive()
        for t in threading.enumerate()
    )


def start_background_scheduler() -> None:
    """注册每日 20:00 任务并启动守护线程（幂等；热重载下依赖线程名防重复）。"""
    with _lock:
        if _scheduler_thread_alive():
            return
        schedule.clear()
        schedule.every().day.at("20:00").do(scheduled_job)

    def _loop() -> None:
        while True:
            schedule.run_pending()
            time.sleep(30)

    t = threading.Thread(target=_loop, daemon=True, name="QuantSchedulerThread")
    t.start()
    print(
        "[A-Quant] 后台调度已启动：每个交易日 20:00（本机时间）自动执行全套工作流。",
        flush=True,
    )
