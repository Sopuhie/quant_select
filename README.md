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
- 环境变量 `QUANT_MAX_STOCKS` 可覆盖默认股票池上限。

## 数据说明

- 数据库：`data/stocks.db`（首次运行自动创建表结构）
- 模型：`models/lgb_model.pkl`
- 数据源：AkShare 免费 A 股日线（前复权）

## 免责声明

仅供学习与研究，不构成投资建议。历史表现不代表未来收益。
