# Changelog

## [v0.6.0] - 2026-07-09

### Added

- **模拟盘启动脚本**: `scripts/run_simulation.py` — 自动交易循环：
  - daily_inference 信号生成 → Broker(simulate=True) 开/平仓 → RiskManager 风控 → OrderManager 审计日志 → 状态持久化
- **Broker 价格缓存**: `_sim_latest_prices` — 模拟模式使用真实行情定价，不再回退到 100.0 占位价
- **有效风控**: 仓位计算自动受限于单笔风险上限（2%/8% = 25% equity），浮点边界 epsilon 保护
- **Windows 编码兼容**: `src/__init__.py` 自动接管 stdout/stderr UTF-8，无需 `PYTHONIOENCODING`

### Fixed

- **signals.py**: 移除 `generate_exit_signals` 中从未使用的 `min_hold` 参数（回测中 min_hold 约束由执行层保证）
- **broker.py**: `get_symbol_price` 新增 `side` 参数，实盘模式买单用 ask、卖单用 bid
- **order_manager.py**: Broker 私有方法 `_sim_fill_order` / `_sim_close_position` 改为公共接口调用
- **risk.py**: 浮点比较 `max_loss/equity > max_loss_per_trade_pct` 增加 `1e-10` epsilon 防边界误拒

### Docs

- **TO_DO_LIST.md**: 新增 — 模拟盘最小链路 Sprint 计划与并行改造任务
- **审查报告.md**: 新增 + 合并实验记录与验证数据（原"当前记录与更新计划.md"精华数据已迁入）
- **使用指南.md**: 全面重写 — 分类命令、删除 Jupyter、补充模拟盘与执行层说明
- **交易逻辑说明.md**: 修复因子分类（MAX_DD_60 移至波动率类）
- **README.md**: 版本更新至 v0.6.0，路线图同步
- **文档结构精简**: 删除 docs/当前记录与更新计划.md（内容合并至审查报告.md + TO_DO_LIST.md）

---

## [v0.5.0] - 2026-07-06

### Added

- **因子池扩展至 13 个**: ULCER_INDEX, MAX_DD_60, CHAIKIN_MF 正式集成到 `assembly.py` / `config.py` / `FEATURE_COLS`（共 39 个特征：13 base + 13 rank + 13 zscore）
- **模型有效性验证（全套 5 项）**:
  - Rolling OOS 子区间回测（3/3 子区间全部夏普 > 10 通过）
  - 随机入场信号基线检验（p=0.00，模型超越全部 50 次随机）
  - Permutation Test（p=0.00，模型超越全部 100 次打乱）
  - 费用模型验证（vectorbt 计算正确，~1.6% 损耗）
  - 安全杠杆计算（保守建议 ≤ 7x）
- **文档同步**: 5 个文档全面更新至 13 因子 / 39 特征 / 118 标的 / 验证结果
- **Model Registry**: `src/models/registry.py` — 从独立 `models/` 目录加载已训练模型
- **模型存储重构**: 独立 `models/` 目录（V1_28stock_fwd21, V2_118stock_fwd10），manifest + 模型文件单独版本管理

### Changed

- **版本号**: 0.3.0 → 0.5.0
- **回测费用模型**: 确认 vectorbt `cash_sharing` 模式下费用计算正确（`pf.fees()` 报告为 0 是显示问题，实际费用已正确计入收益）
- **实盘就绪评估**: 新增完整评估章节，6 维度评分表 + 分阶段路径

### Removed

- 旧"待加入因子"计划章节（因子已全部集成完毕）
- 过时的"Fees=0 问题"风险项

### Fixed

- docs/当前记录与更新计划.md 章节编号修复（两个"五"）
- README/docs 中因子数/标的数/预期输出与实际同步
- docs/交易频率影响分析.md 中调仓频率、交易成本、最短持仓与实际同步
- scripts/daily_inference.py: 缺少 add_cross_sectional_features 和 clip_outliers 步骤，导致推理时特征数（16）与模型训练时（30）不匹配；兼容新旧 manifest 格式（features vs feature_cols），使用 model.feature_name() 作为最终特征列来源
- src/models/registry.py: load_manifest() 缺少 encoding="utf-8"，导致中文 Windows (GBK) 下读取 manifest.json 失败

---

## [v0.4.0] - 2026-07-05

### Added

- 回测验证通过（vectorbt, 年化 64%~81%, 夏普 4.9~5.8）
- 扩大标的池至 118 只全链路训练+回测
- 因子探索：BB_WIDTH, HIGH_LOW_RATIO 集成
- 文档全面整理合并

---

## [v0.3.0] - 2026-07-05

### Added

- 模型训练（LightGBM + Walk-Forward, 30 特征含横截面, 10 个基础因子）
- 工程化重构 P3/P4（pyproject.toml + 子包结构 + factors/ 拆分 + src/ 布局）

---

## [v0.2.0] - 2026-07-04

### Added

- IO 层协议化（DataBackend 协议 + yfinance/AV/MT5 后端）
- 标的池迁移至 universe.txt
- 工程化重构 P2 里程碑

---

## [v0.1.0] - 2026-07-04

### Added

- 数据采集 + 因子计算 MVP
- 配置中心（config.py + python-dotenv）
- 工程化重构 P1 里程碑