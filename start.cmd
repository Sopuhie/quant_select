@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo [已激活] .venv
) else (
    echo [提示] 未找到 .venv，将使用 PATH 中的 python。建议先: python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
)

:menu
echo.
echo ========================================
echo   量化选股 - 测试启动菜单
echo ========================================
echo   1  Streamlit 复盘     ^(streamlit run app.py^)
echo   2  每日选股           ^(run_daily，200 只，8 线程，默认覆盖当日记录^)
echo   3  训练模型           ^(示例截止 2024-12-31，200 只^)
echo   4  每日选股 ^(已存在则跳过，--skip-if-exists^)
echo   5  回填收益           ^(scripts\update_returns.py^)
echo   6  钉钉推送           ^(按最近交易日推送库内 Top 选股，需已配置 webhook^)
echo   0  退出
echo ========================================
set /p choice=请输入数字后回车: 

if "%choice%"=="1" goto app
if "%choice%"=="2" goto daily
if "%choice%"=="3" goto train
if "%choice%"=="4" goto daily_force
if "%choice%"=="5" goto returns
if "%choice%"=="6" goto dingtalk
if "%choice%"=="0" goto eof
echo 无效选择，请重试。
goto menu

:app
streamlit run app.py
goto after_run

:daily
python run_daily.py --max-stocks 200 --workers 8
goto after_run

:daily_force
python run_daily.py --max-stocks 200 --workers 8 --skip-if-exists
goto after_run

:train
python train_model.py --train-end-date 2024-12-31 --max-stocks 200
goto after_run

:returns
python scripts\update_returns.py
goto after_run

:dingtalk
python -c "from src.dingtalk_notifier import maybe_push_daily_selections; from src.utils import get_last_trading_date; td=get_last_trading_date(); print('trade_date=', td); print('pushed=', maybe_push_daily_selections(td))"
goto after_run

:after_run
echo.
pause
goto menu

:eof
exit /b 0
