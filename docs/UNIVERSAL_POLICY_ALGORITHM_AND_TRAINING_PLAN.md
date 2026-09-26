# 单一通用覆盖策略：算法设计与训练方案

## 1. 文档目的

本文定义后续算法的目标架构、训练样本、训练流程、数据隔离规则、部署方式与验收标准。核心决策是：提交运行时不再识别 DUT 类型并选择专用策略，而是让所有 DUT 共享同一个通用策略模型。

旧 DMA、SPI、cache、branch 等策略不会进入在线推理路径。它们只作为离线教师产生高质量轨迹，用于训练通用模型理解“配置、启动、等待、故障、恢复”等可迁移的事务结构。

本文严格区分：

- **当前已实现**：已经存在于工作区并通过测试的能力；
- **计划实现**：训练完成前仍需开发的模块；
- **验收条件**：不能仅凭训练损失宣布完成，必须满足的覆盖与泛化指标。

## 2. 赛题约束与设计原则

赛题在初始化时提供 DUT spec、covergroup 和 coverage metadata；逐周期提供累计二值 `coverage_state`、当前步数和总预算。`predict()` 必须输出固定维度动作向量。

评测阶段禁止修改模型权重，但允许根据覆盖反馈改变规划状态。因此采用以下边界：

- 模型训练全部在提交前离线完成；
- 推理时模型权重只读；
- 在线允许更新 UCB 统计、episode 状态、动作队列和短期信用；
- 不依赖评测 DUT 名称；
- 不将验证集轨迹加入训练数据；
- 对所有动作执行范围、复位极性、前置依赖和保留位约束。

最终目标不是记住公开 DUT 的完整脚本，而是学习：

```text
当前结构语义 + 未覆盖目标 + 最近试验结果
                    ↓
下一段通用语义程序及其参数
```

## 3. 为什么不能只训练 DUT 分类器

原混合专家系统在公开 DUT 上覆盖率高，但它学习的是“DUT 属于哪个已知类”，不是“缺失 bin 需要什么事务”。遇到训练分布之外的 watchdog 类 DUT 时，分类器无法提供有效程序，覆盖率明显下降。

简单扩大分类类别仍有三个问题：

1. 新 DUT 总可能落在所有已知类别之外；
2. 不同 DUT 可以共享局部事务结构，却不属于同一协议族；
3. 动作空间维度和字段顺序不同，原始动作向量不能直接跨 DUT 共享。

因此训练单位必须从“DUT 类型”改为“语义事务程序”。

## 4. 总体架构

```text
DUT spec ─────────→ Semantic IR encoder ───────┐
                                               │
coverage_meta ────→ Coverage target encoder ───┼→ Universal policy
                                               │       │
coverage history → History/episode encoder ────┘       │
                                                       ↓
                                             semantic program decoder
                                                       │
                                                       ↓
                                               constraint executor
                                                       │
                                                       ↓
                                                DUT action vector
```

模型不输出某个专家编号，而是输出层次化决策：

```text
option
  + program length
  + field-role assignments
  + values/tokens/register addresses
  + pulse/hold cycles
  + repeat count
  + reset/recovery decision
```

在线 UCB 与 episode 管理器作为模型外的安全探索层：模型提供先验分数，UCB 保证未尝试 option 仍有机会执行，停滞时强制执行组合事务和恢复。

## 5. 输入表示

### 5.1 Semantic IR

当前已经解析：

- 动作字段名称、顺序和角色；
- 位宽、上下界、枚举；
- 寄存器地址和描述；
- 高/低有效复位；
- 比较约束；
- requires、write-condition、before、after 依赖；
- 最小/最大周期窗口。

每个动作字段转换为一个 token，建议特征为：

| 特征 | 说明 |
|---|---|
| role | write/read/address/data/request/event/fault/reset 等 |
| ordinal | 同角色字段中的序号，避免依赖原始绝对索引 |
| width | 归一化位宽 |
| minimum/maximum | 合法范围 |
| active_low | 复位或控制极性 |
| enum summary | 枚举数量、边界和语义 embedding |
| register relation | 是否关联寄存器地址或数据字段 |
| dependency degree | 入边、出边和时序依赖数量 |

字段名称文本可以作为辅助 embedding，但模型不能只靠名字；训练时会随机重命名字段，迫使模型使用结构、范围和关系。

### 5.2 Coverage target IR

每个 bin 转换为目标 token：

- coverpoint/bin 名称；
- basic、boundary、condition、cross、sequential、temporal 等类型；
- signal 列表；
- value、range 或 cross 目标值；
- sequence 标识；
- stage、difficulty、source；
- 当前是否已覆盖；
- 最近尝试次数与最近一次相关收益；
- 建议 option 分布。

覆盖输入必须保留 bin 身份，不能只输入总覆盖率。缺失目标采用集合编码，使模型支持 26～120 甚至更长的可变 bin 数。

### 5.3 历史与 Episode

历史编码只保留有限窗口，避免推理成本随步数增长：

- 最近 8～16 个语义程序；
- 每个程序的周期、参数摘要和新增 bin 集合；
- 三步延迟信用；
- 当前 episode 年龄；
- 最近新增覆盖距今周期；
- 当前参数探索层级；
- 剩余预算比例。

## 6. 输出：通用语义程序

### 6.1 Option 层

第一层选择以下通用 option，而不是 DUT 专家：

- `configure`：寄存器/字段配置与回读；
- `control`：enable/disable、request、start、单拍脉冲；
- `temporal`：等待、持续推进、窗口扫描、状态积累；
- `recovery`：fault escalation、flush、reset、新 episode。

后续可细分为更多原语，但所有原语必须根据 IR 构造，不能匹配 DUT 名称。

### 6.2 Program 层

每个 option 解码为可变长度程序：

```text
step 0: assign(write_enable#0=1, register_address#0=2, data_lane=...)
step 1: hold(idle, 4 cycles)
step 2: pulse(event#0, token=0x55)
step 3: pulse(event#0, token=0xAA)
step 4: hold(advance#0=1, 16 cycles)
step 5: pulse(recovery#0)
```

使用 `role + ordinal` 表示字段，而不是绝对动作索引。约束执行器在最后一步把语义程序映射回当前 DUT 的动作向量。

### 6.3 参数层

参数不能全部作为普通回归量处理。建议采用混合输出：

- 寄存器地址：从 IR 合法地址集合中分类选择；
- 枚举字段：分类；
- 普通整数：边界类别分类 + 桶内偏移；
- token/key：候选集合分类 + 新值生成；
- cycle：离散候选 `{1,2,3,4,5,8,12,16,24,32,64,...}`；
- repeat：小范围分类；
- reset_after：二分类。

这种表示比直接预测 32-bit 数值稳定，也能自然覆盖边界 bin。

## 7. 训练样本格式

当前 JSONL schema v2 已包含：

```text
schema_version
source_id
family_split
features
option
semantic_program
action_sequence
cycles
new_bin_indices
direct_reward
reward_per_cycle
```

其中 `semantic_program` 的每个 assignment 包含：

- field role；
- 同角色 ordinal；
- normalized value；
- min/max/interior 边界类别；
- 持续周期。

后续 schema v3 需要增加：

- 程序开始前的完整 missing-target mask；
- 每个新增 bin 的目标描述；
- episode ID 和终止原因；
- teacher/student 来源；
- 行为策略概率，用于离线 RL 重要性修正；
- 程序是否被约束执行器修改；
- failure reason，例如非法依赖、提前 reset、超预算。

## 8. 训练数据来源

### 8.1 专家轨迹蒸馏

旧专家只在离线采集阶段运行。采集器记录每周期动作和 bin 变化，并围绕新增 bin 截取因果窗口：

```text
[上一次新增覆盖之后的动作 ... 本次新增覆盖]
```

然后完成三步转换：

1. 原始动作向量 → role/ordinal 语义动作；
2. 连续相同动作 → run-length program step；
3. 多个 step → configure/control/temporal/recovery 程序。

专家名称不作为训练特征。模型只能看到 IR、目标和语义程序。

需要特别保留无收益轨迹，但比例必须受控。建议每个正样本配 2～4 个相同结构附近的 hard negative，而不是保留数千个重复零收益动作。

### 8.2 合成 DUT 课程

只有三个公开 DUT 不足以训练泛化模型。需要自动生成大量小型合成状态机，随机化：

- 字段名称与顺序；
- 动作维度和位宽；
- 寄存器地址；
- 枚举编码；
- enable 极性；
- key/token 值和顺序；
- 计时窗口；
- fault 累计阈值；
- reset/recovery 行为；
- FSM 深度与不可逆状态；
- coverage value/range/cross/sequence 组合。

每个模板必须同时生成：

- spec；
- coverage metadata；
- 可执行模拟器；
- 可证明可达的参考程序；
- 随机重命名和参数化变体。

建议第一阶段至少覆盖 8 类结构模板、每类 50 个结构变体、每个变体 8～16 个参数种子。训练/验证/测试按结构模板或图同构族划分，不能把同一模板换名字后同时放进训练和测试。

### 8.3 通用策略自博弈

模型达到基本可用后，在训练 DUT 和合成 DUT 上执行探索：

- 收集模型成功但教师未覆盖的新程序；
- 收集高置信失败作为 hard negative；
- 对成功程序做参数扰动和最短化；
- 将长程序切分为前置配置、触发和恢复段，建立延迟信用。

### 8.4 验证集隔离

cache、branch、watchdog 验证集只用于最终泛化评估：

- 不写入 JSONL；
- 不用于超参数搜索；
- 不用于早停；
- 不根据其缺失 bin 名称手写训练模板；
- 每个大版本最多做一次正式评估，避免人工过拟合。

## 9. 训练流程

### 阶段 A：行为克隆预训练

目标是先学会输出合法、完整的语义程序。

损失建议：

```text
L_BC = L_option
     + λ_len L_program_length
     + λ_role L_field_role
     + λ_value L_parameter
     + λ_cycle L_duration
     + λ_stop L_end_token
```

采用 teacher forcing，并对正收益样本加权。非法字段、非法地址和违反依赖的输出在训练中加入额外约束损失。

### 阶段 B：覆盖目标条件化

同一个 IR 下，根据不同 missing-bin mask 训练不同程序，使模型真正使用目标信息。

可采用 contrastive loss：有效程序应更接近它命中的目标，远离未命中的目标。

```text
L_target = -log exp(sim(program, hit_targets)) /
                 Σ exp(sim(program, candidate_targets))
```

### 阶段 C：离线强化学习

行为克隆完成后，再优化覆盖效率。建议使用保守离线方法，而不是直接 Q-learning：

- reward：新增 bin 的难度加权和除以周期；
- delayed reward：向最近三个前置程序按 `0.5^distance` 回传；
- terminal bonus：完成 hardest/sequence/cross 目标；
- penalty：非法动作、无效重复、提前 reset、超长无收益程序。

建议目标：

```text
r = Σ difficulty_weight(new_bin)
    / max(1, cycles)
    - λ_invalid * invalid_count
    - λ_repeat * redundant_program
```

离线数据没有可靠反事实，因此优先选择 IQL 或带行为约束的 actor-critic；option-value 头只作为先验，在线 UCB 保留探索权。

### 阶段 D：合成环境强化与蒸馏

在可无限生成的合成 DUT 上进行强化学习，然后把策略蒸馏成提交使用的小模型。大模型负责训练期搜索，小模型负责逐周期推理。

### 阶段 E：程序压缩

对成功轨迹做 delta debugging：依次删除或缩短 step，只保留仍能命中目标的最短程序。压缩后的轨迹能减少模型学习无关动作，也提升单位周期奖励。

## 10. 模型结构建议

第一版不建议直接使用大型 Transformer。可以采用约 0.5～2M 参数的小型结构：

- Field encoder：共享 MLP；
- Target encoder：共享 MLP + masked mean/max pooling；
- Dependency encoder：2 层小型 GNN 或 attention；
- History encoder：GRU；
- Option head：4 类 logits；
- Program decoder：2～4 层自回归 Transformer/GRU；
- Parameter heads：地址、枚举、边界、周期分别输出。

模型必须支持 mask：不存在的角色、非法寄存器地址和越界值在 softmax 前置为不可选。

若模型复杂度影响逐周期时间，可只在宏边界运行模型；宏内部动作由队列展开。赛题每周期预算约 44 ms，而当前 NumPy 路径远低于 1 ms，因此宏边界运行小模型有充分余量。

## 11. 在线推理算法

初始化阶段：

1. 解析 Semantic IR；
2. 解析 CoverageTargetIR；
3. 加载冻结模型；
4. 构建合法字段、地址、枚举和依赖 mask；
5. 初始化 UCB、episode 和历史缓存。

每个宏边界：

1. 计算具体新增 bin 集合；
2. 更新动作程序与新增 bin 的信用；
3. 更新缺失目标集合；
4. 编码 IR、目标、历史和预算；
5. 模型输出 option 与语义程序候选；
6. UCB 在模型先验上加入探索 bonus；
7. 约束执行器修复或拒绝非法程序；
8. 将语义程序编译为动作队列；
9. 在后续周期逐个弹出动作。

Episode 终止条件：

- 达到最小 episode 长度后出现新覆盖；
- 超过耐心窗口无新增覆盖；
- 进入已知不可逆状态；
- 剩余预算不足以完成当前程序。

终止后优先执行 recovery/reset，并切换参数簇。

## 12. 当前状态与诊断结果

当前已经完成：

- 单一 `UniversalPolicy` 在线入口；
- 结构化 IR 与 bin 目标；
- exact new-bin/action-sequence 记录；
- 延迟信用和 episode；
- schema v2 语义程序样本；
- option-value 训练脚本；
- 训练质量门禁。

第一版公开采集得到 5108 条记录，但只有 10 条正收益，四类 option 的正样本分布为 `[5, 0, 0, 5]`。因此模型被标记为 `approved=False`，当前运行时不会加载。

当前门禁后的 30k 结果：

| DUT | 覆盖结果 |
|---|---:|
| dma_xfer_public | 32/92，34.78% |
| spi_master_public | 5/120，4.17% |
| spi_xfer_public | 2/86，2.33% |
| cache_ctrl_validation | 55/75，73.33% |
| branch_predictor_validation | 25/66，37.88% |
| watchdog_safety_validation | 35/59，59.32% |

这说明当前瓶颈不是 option 分类，而是程序表达与正样本质量。没有程序解码器之前，继续扩大 option 模型没有意义。

## 13. 训练质量门禁

模型必须同时满足以下条件才能进入运行时：

### 数据门禁

- 至少 8 个合成结构家族；
- 至少 3 个真实公开结构家族；
- 每个 option 至少 500 条正收益程序；
- 正/负样本比例不低于 1:8；
- family holdout 中不存在结构泄漏；
- 验证集样本数严格为 0。

### 离线门禁

- 合法动作率 100%；
- 未见字段重命名下 option 准确率不下降超过 5%；
- 未见结构家族的正收益 program recall 达到设定阈值；
- reward/cycle 优于无模型 UCB 基线；
- 模型校准后，高置信失败率受控。

### RTL 门禁

建议按阶段验收：

1. 公开 DUT 平均覆盖率 ≥70%；
2. 公开 DUT 平均覆盖率 ≥90%；
3. 三个公开 DUT 恢复 100%；
4. 家族留出合成集平均覆盖率 ≥85%；
5. 验证集只在大版本结束时测试一次，并单独报告。

在公开 DUT 未恢复到至少 90% 前，不应替换可提交的稳定版本。

## 14. 消融实验

至少保留以下实验：

| 编号 | 配置 | 目的 |
|---|---|---|
| A | 规则 UCB，无模型 | 通用基线 |
| B | option 模型，无 program decoder | 验证只学宏选择是否足够 |
| C | BC program decoder | 衡量专家蒸馏收益 |
| D | BC + target conditioning | 衡量 bin 目标映射 |
| E | BC + offline RL | 衡量覆盖效率优化 |
| F | 去掉 delayed credit | 衡量前置配置信用 |
| G | 去掉 episode/reset | 衡量不可逆状态恢复 |
| H | 去掉合成 DUT | 衡量合成课程对泛化的贡献 |

报告指标应包括最终覆盖率、覆盖 AUC、首次命中 hardest bin 的周期、reward/cycle、非法动作率和推理延迟。

## 15. 实施顺序

### M1：教师轨迹转语义程序

- 实现专家轨迹窗口截取；
- RLE 压缩；
- role/ordinal 翻译；
- 正负样本平衡；
- 程序最短化。

验收：公开专家轨迹能够被语义程序重放，并保持原覆盖结果。

### M2：合成 DUT 生成器

- 先实现 register/FSM、key/token、timeout/recovery 三类模板；
- 随机字段名、顺序、位宽和阈值；
- 自动生成参考程序与 coverage metadata。

验收：至少 1000 个结构/参数实例，参考程序全部可达。

### M3：行为克隆程序模型

- 训练 option、长度、角色、参数和周期 head；
- 加入合法 mask；
- 做结构家族留出。

验收：合法动作率 100%，公开 DUT 平均覆盖率达到 70%。

### M4：目标条件化与离线 RL

- missing-bin mask 条件化；
- 难度加权 reward；
- delayed credit；
- IQL/保守 actor-critic。

验收：公开平均覆盖率达到 90%，合成家族留出达到 85%。

### M5：压缩与提交验证

- 模型量化或 NumPy 化；
- 宏边界推理缓存；
- 30k/50k 多种子回归；
- 消融与最终报告。

验收：公开 DUT 恢复 100%，P99 推理时间满足赛题限制，无在线权重更新。

## 16. 风险与应对

### 风险 1：专家蒸馏退化成记忆协议

应对：随机重命名、字段重排、角色化动作、结构家族留出；训练特征中不包含 DUT 名称。

### 风险 2：正收益过于稀疏

应对：以新增 bin 为中心截取窗口；使用参考程序生成正样本；平衡采样；程序最短化；合成环境主动生成难例。

### 风险 3：长时序信用错误

应对：保存具体新增 bin、三步延迟信用、episode 级 return，并通过删除实验验证前置动作是否必要。

### 风险 4：模型输出非法动作

应对：结构化 mask、约束执行器、范围裁剪和保留位清零；非法程序不直接发送给 DUT。

### 风险 5：训练集覆盖高但新 DUT 失败

应对：按结构家族而不是随机样本划分；保留完全未见的合成模板；严格隔离验证 DUT。

## 17. 最终定义

本方案中的“单一通用策略”并不意味着只有一个扁平神经网络直接输出动作。它表示：

- 所有 DUT 使用同一个模型、同一套 option、同一种语义程序表示；
- 不根据 DUT 类型切换人工脚本；
- 专家知识只通过离线样本蒸馏进入统一模型；
- 在线差异来自输入的 IR、coverage target 和反馈历史；
- 最终动作由同一个约束执行器编译。

只有当模型在未见结构家族上仍能组合出有效程序，且公开 DUT 不再依赖硬编码专家恢复覆盖时，才能认为真正实现了泛化。
