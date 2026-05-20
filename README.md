# 量化选股系统（个人版）

基于 Python、SQLite、LightGBM 与 Streamlit 的 A 股日频选股与复盘工具。不涉及实盘下单。

## 环境

- Python 3.9+
- 依赖见 `requirements.txt`

```bash
cd quant_select
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## 流程

1. **训练模型**（需联网拉取 AkShare 数据，耗时与 `--max-stocks` 成正比）：

```bash
python train_model.py --train-end-date 2024-12-31
```

可选：`--max-stocks 200`、`--version v1.0.0`。

2. **每日选股**（生成当日或最近交易日的 Top3 与全市场打分）：

```bash
python run_daily.py
```

若该交易日已有记录会跳过；使用 `--force` 强制覆盖写入。

3. **Web 复盘**：

```bash
streamlit run app.py
```

## 定时任务

- **Windows**：任务计划程序定时执行 `python d:\path\to\quant_select\run_daily.py`
- **Linux/macOS**：crontab 在收盘后执行同上命令

也可在后台常驻 `schedule` 轮询，本项目未强制绑定实现方式。

## 配置

- `src/config.py`：数据库路径、特征列、LightGBM 参数、`MAX_STOCKS_UNIVERSE` 等。
- 环境变量 `QUANT_MAX_STOCKS`：训练/回测/每日选股默认股票池上限（默认 **400**）。
- `QUANT_RUN_DAILY_MAX_STOCKS`：仅选股；设为 `0` 表示不限制（全库）；未设时与 `QUANT_MAX_STOCKS` 一致。
- `QUANT_META_OOF_FOLDS`：Meta Ridge 的 Walk-forward OOF 折数（默认 **3**，设 `0` 关闭）。
- `QUANT_PREV_GAIN_SUPPRESSION`：前期涨幅硬过滤（默认 **开启**；设 `0` 关闭）。

审核整改明细见 [docs/AUDIT_FIXES.md](docs/AUDIT_FIXES.md)。

钉钉密钥建议用环境变量（勿提交 `config.json`）：

```bash
set QUANT_DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=...
set QUANT_DINGTALK_SECRET=SEC...
```

配置模板：复制 `config.json.example` 为 `config.json` 后填写。

### 回测

```bash
# 默认：时点可得股票池（PIT）+ 与 run_daily 一致的风控链
python scripts/backtest.py --start-date 2025-01-01 --end-date 2025-12-31

# 拒绝样本内回测（train_end_date 须早于 start_date）
python scripts/backtest.py --enforce-sample-out ...

# 滚动重训 walk-forward（较慢）
python scripts/walkforward_backtest.py --start-date 2025-01-01 --end-date 2025-12-31
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest tests/test_audit_fixes.py -q
```

## 数据说明

- 数据库：`data/stocks.db`（首次运行自动创建表结构）
- 模型：`models/lgb_model.pkl`
- 数据源：AkShare 免费 A 股日线（前复权）

## 免责声明

仅供学习与研究，不构成投资建议。历史表现不代表未来收益。
