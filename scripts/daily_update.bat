@echo off
REM 定时回填收益：在「任务计划程序」中新建任务，每日 17:00 触发，
REM 操作选择「启动程序」，程序填本文件路径：
REM   D:\Projects\Aquant\quant_select\scripts\daily_update.bat
REM （可选）条件：仅交流电源 / 唤醒时运行等按需勾选。
chcp 65001 >nul
cd /d D:\Projects\Aquant\quant_select

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [警告] 未找到 .venv，使用当前环境 PATH 中的 python。
)

python scripts\update_returns.py
echo 收益更新完成
REM 任务计划程序定时运行时请去掉下一行 pause，否则任务会一直等待按键。
pause
