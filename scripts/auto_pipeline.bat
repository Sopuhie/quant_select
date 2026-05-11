@echo off
:: 支持 UTF-8 编码，防止中文日志乱码
chcp 65001 >nul
title A-Quant Lite 一键量化流水线

echo ===================================================
echo 🚀 开始执行量化选股系统全管线任务（自动化触发）
echo 时间: %date% %time%
echo ===================================================

:: 1. 切换到项目根目录
cd /d "%~dp0.."

:: 2. 激活虚拟环境
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [警告] 未找到本地 .venv，将尝试使用系统默认 Python。
)

:: 3. 步骤一：收益率回填 (更新历史数据的收益，以便给后续回测和复盘提供数据支撑)
echo [1/5] 🔄 正在回填历史收益率...
python update_returns.py
if %errorlevel% neq 0 (
    echo ❌ [失败] 收益率回填异常，流程中断！
    exit /b %errorlevel%
)
echo [OK] 收益率回填完成。
echo ---------------------------------------------------

:: 4. 步骤二：同步行情数据 (任务 A)
echo [2/5] 📊 正在运行本地行情同步 (增量同步)...
python scripts\update_local_data.py
if %errorlevel% neq 0 (
    echo ❌ [失败] 行情同步异常，流程中断！
    exit /b %errorlevel%
)
echo [OK] 行情同步完成。
echo ---------------------------------------------------

:: 5. 步骤三：重新训练选股模型 (任务 B)
echo [3/5] 🎯 正在重新训练 LightGBM 选股模型...
python train_model.py
if %errorlevel% neq 0 (
    echo ❌ [失败] 模型训练异常，流程中断！
    exit /b %errorlevel%
)
echo [OK] 模型重训完成。
echo ---------------------------------------------------

:: 6. 步骤四：执行每日智能选股 (任务 C)
echo [4/5] 📡 正在生成今日/最新交易日选股预测打分...
:: 此处默认启用创业板和科创板，您可以根据配置按需调整参数 --include-300 --include-688
python run_daily.py --include-300 --include-688
if %errorlevel% neq 0 (
    echo ❌ [失败] 每日选股预测异常，流程中断！
    exit /b %errorlevel%
)
echo [OK] 每日智能选股完成！
echo ---------------------------------------------------

:: 7. 步骤五：运行历史滚动回测 (任务 D)
echo [5/5] 📈 正在运行策略历史滚动回测...
python scripts\backtest.py
if %errorlevel% neq 0 (
    echo ❌ [失败] 策略回测异常，流程中断！
    exit /b %errorlevel%
)
echo [OK] 历史滚动回测完成！
echo ---------------------------------------------------

echo ===================================================
echo 🎉 [SUCCESS] 一键全管线量化任务全部成功执行完成！
echo ===================================================

:: 如果是通过双击手动运行，保留窗口查看结果；如果是定时任务静默执行，建议去掉 pause
if "%1"=="--scheduled" (
    exit /b 0
) else (
    pause
)