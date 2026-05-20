# 审核问题整改清单

基于 2026-05 深度审核，按优先级跟踪。✅ = 已在代码中落实。

## P0 — 统计可信度

| # | 问题 | 状态 | 改动说明 |
|---|------|------|----------|
| 1 | Meta 默认 in-sample stacking | ✅ | `RANK_META_OOF_FOLDS` 默认 **3**；设 `QUANT_META_OOF_FOLDS=0` 可恢复旧行为 |
| 2 | 技术因子 `bfill` 前视 | ✅ | `factor_calculator` 技术面仅 `ffill`，去掉 `bfill` |
| 3 | 回测未对齐打分前硬风控 | ✅ | `scripts/backtest.py` 增加 `apply_pre_score_hard_risk_filters` |
| 4 | run_daily 全库 vs 训练/回测 400 不一致 | ✅ | 默认 `resolve_run_daily_max_stocks()` → `MAX_STOCKS_UNIVERSE`；`QUANT_RUN_DAILY_MAX_STOCKS=0` 全库 |

## P1 — 推荐质量与运维

| # | 问题 | 状态 | 改动说明 |
|---|------|------|----------|
| 5 | 前期涨幅压制默认关闭 | ✅ | `QUANT_PREV_GAIN_SUPPRESSION` 默认 **开启**（5日12%/20日30%） |
| 6 | 融合权重与熔断数据源分裂 | ✅ | `compute_market_regime_blend_weights` 优先本地 `index_daily` |
| 7 | 选股拉取失败静默 skip | ✅ | `predict_universe_scores` 汇总并打印前 8 条失败原因 |

## P2 — 文档与测试

| # | 问题 | 状态 | 改动说明 |
|---|------|------|----------|
| 8 | 缺少自动化测试 | ✅ | `tests/test_audit_fixes.py` |
| 9 | README 过简 | ✅ | README 增加审计相关环境变量 |
| 10 | 回测局限未说明 | ✅ | `backtest.py` 模块 docstring 补充 |
| 11 | selection_reason 因果误导 | ✅ | docstring 标明「统计相关」 |

## 第二轮整改（续）

| # | 问题 | 状态 | 改动说明 |
|---|------|------|----------|
| A | 回测成分股期末快照前视 | ✅ | 回测默认 **PIT universe**；`--fixed-universe-snapshot` 恢复旧行为 |
| B | 回测固定模型样本内 | ✅ 部分 | `--enforce-sample-out`；`scripts/walkforward_backtest.py` 滚动重训 |
| C | 无指数数据默认放行 | ✅ | `MARKET_REGIME_MISSING_DATA_SCORE` 默认 **50**（熔断） |
| D | 钉钉密钥 | ✅ 文档 | 已支持 `QUANT_DINGTALK_WEBHOOK` / `SECRET`；新增 `config.json.example` |
| E | walk-forward | ✅ | `python scripts/walkforward_backtest.py`；Streamlit「系统控制台」任务 D2 +「历史回测」Tab |

## 仍待迭代

| # | 问题 | 建议 |
|---|------|------|
| F | PE/市值/换手历史 PIT | 独立时点表或按公告日写入 |
| G | 指数成份股逐日历史 | 接入历史成分库，替换 PIT 近似 |

## 关键环境变量

```bash
# Meta OOF（默认 3）
set QUANT_META_OOF_FOLDS=3

# 涨幅压制（默认开启；关闭设 0）
set QUANT_PREV_GAIN_SUPPRESSION=1

# 训练/回测/选股共用 universe 上限（默认 400）
set QUANT_MAX_STOCKS=400

# 每日选股上限（默认同上；0=全库）
set QUANT_RUN_DAILY_MAX_STOCKS=400
```

## 验证命令

```bash
pip install -r requirements-dev.txt
pytest tests/test_audit_fixes.py -q
python run_daily.py --only-data

# 样本外单次回测（要求 train_end < start）
python scripts/backtest.py --start-date 2025-06-01 --enforce-sample-out

# Walk-forward（耗时长，建议先 --retrain-every 80）
python scripts/walkforward_backtest.py --start-date 2025-01-01 --end-date 2025-06-30 --retrain-every 60
```
