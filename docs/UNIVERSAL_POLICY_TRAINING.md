# 单一通用策略与训练样本设计

## 运行时架构

提交推理路径现在始终实例化 `UniversalPolicy`。DUT 名称、结构签名和闭集神经分类结果不再决定策略；DMA/SPI 等旧策略仅保留为离线教师代码，不参与运行时路由。

运行时决策为：

```text
spec → Semantic IR ┐
coverage meta → bin targets ├→ universal option + semantic program → action vector
coverage history ───────────┘
```

## 样本结构 v2

每条 JSONL 样本包括：

- `family_split`：根据角色、目标类型、寄存器和依赖结构生成的家族哈希，用于整家族留出；
- `features`：字段角色、目标类型、缺失目标权重、IR 复杂度、覆盖进度；
- `option`：configure/control/temporal/recovery；
- `semantic_program`：按角色和同角色 ordinal 表示的动作赋值、归一化值、边界类别与持续周期；
- `action_sequence`：用于审计的原始动作；
- `new_bin_indices`、`cycles`、`reward_per_cycle`。

语义程序而不是原始动作索引作为主要训练目标，因此不同动作维度的 DUT 可以共享样本。

## 模型与安全门禁

第一层 option-value 模型采用离线单位周期收益拟合，并与在线 UCB 混合。模型只有满足以下条件才会被运行时加载：

- 至少三个结构家族；
- 四个 option 每类至少 20 条正收益样本。

第一版公开数据共 5108 条，但正收益仅 10 条，分布为 `[5, 0, 0, 5]`，因此模型被自动拒绝，不参与运行时决策。低训练误差不能绕过该门禁。

## 纯通用基线

30k 评估结果：

| DUT | 专家版 | 未门禁稀疏模型 | 当前门禁基线 |
|---|---:|---:|---:|
| dma_xfer_public | 100% | 42.39% | 34.78% |
| spi_master_public | 100% | 4.17% | 4.17% |
| spi_xfer_public | 100% | 2.33% | 2.33% |
| cache_ctrl_validation | 100% | 68.00% | 73.33% |
| branch_predictor_validation | 100% | 37.88% | 37.88% |
| watchdog_safety_validation | 59.32% | 61.02% | 59.32% |

未门禁模型只用于诊断，当前运行时采用最后一列。结果说明现有四宏生成器无法表达 SPI/DMA 所需的长配置程序；偶然提高 watchdog 不能抵消其训练证据不足。下一阶段训练重点必须是语义程序和参数解码，不能只训练 option 分类。

## 数据隔离

验证集仅用于表格中的评估，不写入 `training/universal_public_v1.jsonl`。训练采集器当前只接受 `PUBLIC_DUTS`，并按结构家族哈希支持整家族留出。
