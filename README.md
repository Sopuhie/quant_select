# quant_select — A 股量化选股与复盘系统（个人版）

基于 **Python + SQLite + AkShare 日线** 的本地量化研究工具，包含两条相对独立的选股链路：

| 模块 | 说明 | 入口 |
|------|------|------|
| **中长线智能选股** | LightGBM / XGB / CatBoost + Meta 融合，输出 TopN 推荐 | `train_model.py` / `run_daily.py` |
| **短线规则选股** | 纯日线技术共振 + 模拟买卖，默认 Top 5 | `scripts/run_short_daily.py` |
| **Web 控制台** | Streamlit 一键同步、训练、选股、复盘 | `streamlit run app.py` |

> **免责声明**：本项目仅供学习与研究，不涉及实盘下单，不构成任何投资建议。

---

## 功能概览

### 中长线（机器学习）

- 本地 `stock_daily_kline` 日线库（AkShare / 增量同步）
- 技术因子 + 可选基本面/资金流辅助表
- 训练多模型与 Meta Ridge 排序，每日产出 `daily_selections` TopN
- 大盘环境分（沪深300 vs MA20）熔断与经验风控硬过滤
- 历史回测、Walk-forward 滚动重训回测
- 钉钉推送 Top 推荐（可选）

### 短线（规则引擎，纯日线）

- **仅使用日线 OHLCV**，无分时、无盘口
- T 日收盘确认信号；**T+1 非对称限价买入**（相对信号收盘 [-2%, +5.5%]）
- **A 股 T+1 交割**：T+1 只买不卖，**最早 T+2** 走双阶梯止盈/非对称止损
- T+1 收≥4% 且 T+2 续强 → T+2 收盘骑乘；否则 T+2 收盘平
- 温和共振：量比 / MACD / KDJ **4 项满足 3 项** 即可
- `final_score` 非线性打分排序，默认输出 **Top 5**
- 历史信号日下拉复盘，自动补齐 T1/T2 开收盘价与 **T1买→T2卖涨跌幅**

### 控制台（Streamlit）

- 任务 A：全量/增量行情同步  
- 任务 B：模型训练  
- 任务 C：每日智能选股（可勾选创业板/科创板）  
- 短线规则 Tab：扫描、历史复盘、规则说明  
- 热门题材规则选股、形态相似度扫描等  

---

## 环境要求

- **Python 3.9+**（Windows / Linux / macOS）
- 可访问 AkShare 数据源（训练、同步行情时需要）
- 磁盘：全 A 日线库约数 GB 级（视同步范围而定）

```bash
cd quant_select
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # 可选，跑 pytest
```

复制配置模板：

```bash
copy config.json.example config.json    # Windows
# cp config.json.example config.json    # Linux/macOS
```

---

## 快速开始

### 1. 初始化数据库

首次运行任意脚本或 `streamlit run app.py` 时会自动执行 `init_db()`，创建主库表与短线专用表。

- 默认库路径：`data/stocks.db`（见 `src/config.py` 的 `DB_PATH`）

### 2. 同步本地行情（建议先做）

```bash
python scripts/update_local_data.py
# 全 A（较慢）
python scripts/update_local_data.py --all-stocks
```

或在 Streamlit → **任务 A** 中点击「运行本地行情同步」。

### 3. 训练中长线模型

```bash
python train_model.py --train-end-date 2024-12-31
# 调试时可缩小股票池
python train_model.py --train-end-date 2024-12-31 --max-stocks 200
```

产出：`models/lgb_model.pkl`、`meta_rank_stacker.pkl` 等。

### 4. 每日中长线选股

```bash
python run_daily.py
python run_daily.py --force              # 覆盖当日已有记录
python run_daily.py --include-300 --include-688   # 纳入创业板/科创板
```

产出：`today.json`、`daily_selections` 表。

### 5. 短线规则选股

```bash
python scripts/run_short_daily.py --force
python scripts/run_short_daily.py --trade-date 2026-05-21 --skip-dingtalk
python scripts/run_short_daily.py --include-300 --include-688
```

产出：`short_today.json`、`short_daily_selections`、`short_order_tracker`。

### 6. 打开 Web 复盘

```bash
streamlit run app.py
# 或 Windows：双击 start.cmd → 选 1
```

---

## 项目结构

```
quant_select/
├── app.py                      # Streamlit 主界面
├── train_model.py              # 模型训练入口
├── run_daily.py                # 中长线每日选股
├── today.json                  # 最近一次中长线 TopN 摘要
├── short_today.json            # 最近一次短线 TopN 摘要
├── config.json.example         # 钉钉 / 经验风控 / 回测费率模板
├── data/
│   └── stocks.db               # SQLite 主库（自动创建）
├── models/                     # 训练产物
├── docs/
│   └── AUDIT_FIXES.md          # 审核整改说明
├── scripts/
│   ├── update_local_data.py    # 行情同步
│   ├── update_returns.py       # 中长线推荐事后收益回填
│   ├── backtest.py             # 历史回测
│   ├── walkforward_backtest.py # 滚动重训回测
│   ├── run_short_daily.py      # 短线每日扫描
│   ├── update_short_review.py   # 短线 T1/T2 复盘价 + 订单重估
│   └── repair_short_schema.py  # 旧库短线表结构修复
└── src/
    ├── config.py               # 全局配置
    ├── database.py             # 主库建表与读写
    ├── predictor.py            # 预测与风控链
    ├── pipeline.py             # 一键全套工作流
    ├── scheduler.py            # 定时任务（可选）
    └── short_term/             # 短线独立模块
        ├── strategy.py         # 扫描与打分
        ├── execution.py        # 纯日线模拟买卖
        ├── runner.py           # 跑批入口
        ├── db.py               # 短线表与落库
        ├── review_prices.py    # T1/T2 复盘价
        ├── history_review.py   # 历史复盘查询
        └── rules_doc.py        # 界面规则说明文案
```

---

## 数据库说明

### 中长线相关（节选）

| 表名 | 用途 |
|------|------|
| `stock_daily_kline` | 全市场日线 OHLCV（核心数据源） |
| `daily_selections` | 每日 TopN 推荐及事后收益 |
| `daily_predictions` | 全市场打分（可选） |
| `model_versions` | 模型版本登记 |
| `system_logs` | 控制台/流水线日志 |

### 短线专用

| 表名 | 用途 |
|------|------|
| `short_daily_selections` | 信号日选股结果（含 `final_score`、T1/T2 价、T1买T2卖涨跌幅等） |
| `short_order_tracker` | 模拟订单（买入价、平仓价、止损标记、`exit_reason`） |

`init_db()` 会自动创建/迁移上述表结构（旧库可通过 `python scripts/repair_short_schema.py` 修复）。

---

## 短线规则模块（详细）

完整规则说明见 Streamlit **短线规则选股** Tab →「📋 短线选股规则说明（复盘对照）」，源码：`src/short_term/rules_doc.py`（与 `config.py` 阈值自动同步）。

### 大盘环境（扫描前）

- 锚定 **中证1000**（`000852`）：环境分 ≥ 60（收盘>MA20→60 分）且 **5 日动量 > 0**
- 不满足则当日不扫描

### 选股硬性门槛（`strategy.py`）

- 趋势：收盘 > MA5 且 MA5 ≥ MA10，且收阳（收>开）
- 实体比 ≥ 0.7；5 日涨幅 ≤ 22%；近涨停/ ST / 北交所 / 停牌 / 涨停日剔除
- 流动性：成交额 ≥ 5000 万 **或** 换手 ≥ 2%；且换手 ∈ **[3.5%, 24%]**
- **量价背离否决**：涨幅 > 3% 且 1 日量比 < 1.0 → 剔除
- **温和共振**：量比 / MACD / KDJ 共 4 项至少 3 项

### 规则得分（排序 Top N，默认 5）

- 涨幅非线性 20%（2%~3.5% 满分）、5 日量比 30%、MACD 20%、J 斜率 15%
- 光头强阳（实体≥98%）+25 分；J≥88 扣 30 分

### 模拟执行（`execution.py`）

| 阶段 | 逻辑 |
|------|------|
| T+1 入场 | 相对信号收盘 [-2%, +5.5%]；微高开按开盘价，否则 (open+low)/2 |
| T+2 止盈 | 冲高 ≥6%：(高+收)/2；≥5%：锁定 +3%（T+1 不可卖） |
| T+2 止损 | 平庸股 -4%；曾冲高≥5% 则 -6% |
| 持仓 | T+1 收≥4% 且 T+2 续强 → 收盘骑乘；否则 T+2 收盘平 |

回测：`python scripts/short_term_backtest.py`

### 界面复盘

Streamlit → **短线规则选股** Tab：

- 运行扫描、推送钉钉  
- **历史选股复盘**：按信号日查看选股表（含 T1/T2 价、T1买T2卖涨跌幅）  
- **短线选股规则说明**：与代码同步的规则文档  

打开某日复盘时，会自动从 `stock_daily_kline` 补齐 T1/T2 开收盘价。

---

## 中长线配置要点

### `src/config.py` 与环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `QUANT_MAX_STOCKS` | 训练/回测/选股股票池上限 | 400 |
| `QUANT_RUN_DAILY_MAX_STOCKS` | 仅 `run_daily`；`0` = 全库 | 同左 |
| `QUANT_TOP_N` | 中长线 TopN | 3 |
| `QUANT_META_OOF_FOLDS` | Meta OOF 折数；`0` 关闭 | 3 |
| `QUANT_PREV_GAIN_SUPPRESSION` | 前期涨幅硬过滤 | 开启 |
| `QUANT_SCHEDULER_TIME` | 定时全套工作流 HH:MM | 20:00 |

审核与风控细节见 [docs/AUDIT_FIXES.md](docs/AUDIT_FIXES.md)。

### 短线环境变量（`src/short_term/config.py`）

| 变量 | 含义 | 默认 |
|------|------|------|
| `QUANT_SHORT_TOP_N` | 短线输出条数 | 5 |
| `QUANT_SHORT_MIN_MARKET_SCORE` | 大盘环境分下限 | 60 |
| `QUANT_SHORT_MARKET_INDEX` | 锚定指数 | 000852 |
| `QUANT_SHORT_SELL_OFFSET` | 2=强势骑乘分支，1=全走 T+2 收盘 | 2 |
| `QUANT_SHORT_CLOSE_STOP` | 强势股收盘止损 | 0.06 |
| `QUANT_SHORT_MEDIOCRE_STOP` | 平庸股收盘止损 | 0.04 |
| `QUANT_SHORT_ENTRY_MAX_CHASE` | T+1 最高追高 | 0.055 |
| `QUANT_SHORT_VOL_RATIO_CLIP` | 量比上限 | 10 |
| `QUANT_SHORT_MIN_BARS` | 最少历史 K 线根数 | 35 |

---

## 钉钉通知

在 `config.json` 中配置 `dingtalk`，或使用环境变量（**勿提交密钥到 Git**）：

```bash
set QUANT_DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=...
set QUANT_DINGTALK_SECRET=SEC...
```

- 中长线：`run_daily` 成功后推送 Top3 简洁模板  
- 短线：`notification.send_short_on_success` 控制是否推送  

---

## 回测

```bash
# 默认：PIT 股票池 + 与 run_daily 一致的风控
python scripts/backtest.py --start-date 2025-01-01 --end-date 2025-12-31

# 强制样本外（train_end 须早于 start）
python scripts/backtest.py --start-date 2025-06-01 --enforce-sample-out

# Walk-forward 滚动重训（耗时长）
python scripts/walkforward_backtest.py --start-date 2025-01-01 --end-date 2025-06-30 --retrain-every 60
```

---

## 定时与一键流水线

- **内置调度**：`src/scheduler.py`，默认每个交易日 `20:00` 执行 `run_complete_pipeline()`（可用 `QUANT_SCHEDULER_TIME` 修改）  
- **全套流水线**：`python -m src.pipeline`（同步 → 训练 → 选股 → 回测 → 回填 → 可选钉钉）  
- Streamlit 内 **「启动一键全套工作流」** 等价调用  

---

## 常用脚本速查

| 命令 | 作用 |
|------|------|
| `python scripts/update_local_data.py` | 增量/全量同步日线 |
| `python scripts/update_returns.py` | 回填中长线推荐事后收益 |
| `python scripts/update_short_review.py` | 回填短线 T1/T2 价并重估 HOLDING 订单 |
| `python scripts/repair_short_schema.py` | 修复旧库短线表缺少新列 |
| `python -m pytest tests/ -q` | 运行单元测试 |

---

## 测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -q

# 分项示例
pytest tests/test_audit_fixes.py -q
pytest tests/test_short_execution.py tests/test_short_term.py -q
```

---

## 常见问题

**Q：短线扫描结果为 0？**  
- 检查是否已同步行情；信号日是否为最新 K 线日  
- 大盘环境分是否低于 50（熔断）  
- 是否过严（可临时勾选纳入 300/688 做对比）  

**Q：T1/T2 价格为 empty？**  
- 信号日太新时，T+1/T+2 交易日 K 线尚未入库；等 `update_local_data` 后再打开复盘或执行 `update_short_review.py`  

**Q：`no such column: final_score`？**  
- 执行 `python scripts/repair_short_schema.py` 后重试  

**Q：与实盘差异？**  
- 本项目为 **日频研究工具**，短线止损用日线 low 近似盘中 -3%，一字跌停场景已做开盘/收盘降级，仍无法替代真实盘口成交。  

---

## 许可证与免责

仅供个人学习、策略研究与复盘，不对任何投资损失负责。使用 AkShare 等第三方数据请遵守其服务条款。
