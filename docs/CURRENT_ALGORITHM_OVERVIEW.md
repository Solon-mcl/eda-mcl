# 当前算法整理（代码快照：2026-09-26，M7、家族路由与神经覆盖控制器移除后）

本文是**从代码出发的现状梳理**，说明提交路径上实际跑的是什么。版本沿革与消融开关的语义见
`docs/ALGORITHM_VARIANTS.md`，逐阶段进度见 `docs/IMPLEMENTATION_PROGRESS.md`。

## 1. 一句话概括

**规格语义 IR + 覆盖目标 IR 驱动的事务程序搜索器**：初始化时把 DUT 的 Markdown 规格和
`coverage_meta.json` 编译成结构化 IR，推理时用「4 个宏的 UCB 调度 + 按角色生成合法事务程序 +
覆盖增量/周期成本反馈搜索」逐周期产出动作。全程不读 DUT 名称，不做 DUT 家族路由，在线不更新任何模型权重。

## 2. 决定算法形状的赛题硬约束

| 约束 | 对算法的影响 |
|---|---|
| `predict(coverage_state, step, max_steps) -> action`，逐周期调用 | 决策必须在宏边界算完，宏内部退化成动作队列出队 |
| 评测期禁止修改模型权重，但允许改规划状态 | 所有自适应都在 UCB 计数、候选统计、episode 状态上，无梯度更新 |
| 反馈只有**累计二值 bin 向量**，看不到内部信号 | 奖励只能取「程序前后新增 bin 集合」，信用回传靠衰减近似 |
| 5~10 万 cycles / DUT，AUC 计入评分 | 不能等到后期才探索，长序列与兜底候选都要给早期机会 |

## 3. 提交入口与调用链

```
评测脚本
  └─ InferenceInterface(dut_spec_path, covergroup_path)      app/inference/__init__.py
       ├─ DeepSeekPlanner.plan()        # 可选，仅此处一次，默认禁用
       └─ UniversalPolicy(_GenericPolicy)
            ├─ CoverageMacroScheduler  # 宏级 UCB
            ├─ EpisodeManager          # 停滞/收益触发新 episode
            ├─ JointCandidateRanker    # 联合候选排序（M4）
            ├─ GenericSequenceSearch   # 通用程序搜索（M8/M8.1）
            └─ UniversalOptionModel    # 线性 option 先验（当前 approved=False，不生效）
```

文件中已不含任何 DUT 家族路由或专家策略：`_SpiPolicy` / `_DmaPolicy` 旧专家、
`neural_router` 家族路由 MLP、M7 的 branch/cache/watchdog 专用状态序列均已于 2026-09-26
移除，见第 9 节。公开 DUT 目录下的 `inference_interface.py` 只在实验工具选
random/greedy 基线时加载。

## 4. 初始化阶段：规格 → IR（全部一次性完成）

### 4.1 Semantic IR（`semantic_ir.py`）

从 spec 文本正则抽取，**只认通用接口约定，不认 DUT 名字**：

| 产物 | 内容 |
|---|---|
| `fields` | 动作字段名/顺序/维度；每字段一个**角色**（约 40 种：`write_enable`、`register_address`、`register_data`、`instance_select`、`data_lane`、`address_lane`、`pc_lane`、`operation`、`mode`、`context_id`、`privilege`、`scope`、`recovery`、`request`、`advance`、`event`、`fault`、`stall`、`ready`、`length`、`mask`、`selector`、`padding` …） |
| 每字段数值域 | `minimum/maximum/width`、`active_low`（`_n` 后缀或"低有效"）、枚举表 |
| `registers` | `Register N (NAME)` 与 Markdown 表格两种形式；寄存器**位域** `[11:10]=TMOD` 解析成 `RegisterFieldIR` |
| `packed_fields` | `data[15:0] = len` 这类打包布局 |
| `constraints` | 数值比较与链式区间 |
| `dependencies` | `requires` / `write_condition` / `after` / `before` |
| `timing_constraints` | "A 之后 B 需 N~M 周期" |

**角色判定的规则**：拿「字段名 + 该名字在说明书里出现的所有句子」当文本，按一条
**从上到下、命中即停**的优先级链匹配。命名容错是泛化的第一道门槛，因此表里同时收录
全名（`write_enable`）、全文短形式（`we` / `wr` / `re` / `rd` / `cs_n` / `rdy` / `vld`）、
无编号形式（`addr` / `adr` / `data` / `din` / `dout` / `payload`）与业务同义词
（`grant` / `qid` / `chan` / `port` / `last` / `eop` / `keep` / `clk_en`）。

**依赖抽取只保留可执行的那部分**：从散文里抽出的 `X requires Y` 会先过滤 —— 至少一侧
必须是已声明的动作字段。实测表明，未过滤前 7 个 DUT 抽出的依赖 **100% 是自然语言噪音**
（例如 `accesses requires the`），过滤后全部归零。

### 4.2 Coverage Target IR（`coverage_targets.py`）

读同目录 `coverage_meta.json`，把每个 bin 展平成一个 `CoverageTargetIR`：
coverpoint 名、bin 名、类型、signal 列表、value/range/cross 值、`seq` 标签、difficulty、stage、`macro_hints`。

- **一致性校验**：`total_bins` 与实际展开数不符 → 整体返回空，不带着错位索引继续跑。
- **`macro_hints`**：由 kind/名称推断该 bin 该由哪个宏去打（configure/control/temporal/recovery）。
- **`missing_target_weights`**：把当前未命中的 bin 按 difficulty（easy 1.0 → hardest 2.0）折算成 4 个宏的先验权重，归一化后喂给宏调度器。

### 4.3 覆盖依赖图（`coverage_dependency.py`）

bin ↔ 动作字段的带证据边：signal 与字段**精确同名** → `direct`(1.0)；词元重合 → `inferred`(0.55)；
角色语义匹配 → `inferred`(0.30)。只用于**引导探索与候选排序**，且未映射的目标仍保留在轮换里，
保证不完整的图不会让可达 bin 永久隐形。

### 4.4 联合候选编译（`joint_candidates.py`）

只对 `kind == cross` 的目标做：把 cross 的各个条件反查寄存器位域，编译成**同一次合法寄存器写**。

- 需要至少 2 个条件被成功映射，否则放弃（留给普通探索）。
- 无法解析的条件记进 `unresolved_conditions`，**不伪装成已满足**。
- 含 hold/state/done 类条件时 `hold_cycles` 从 64 提到 128。

### 4.5 通用候选池（`_build_generic_sequence_candidates()`）

按 IR 能力生成原生候选族（不读 DUT 名）：

| 类别 | 候选族 |
|---|---|
| 语义原语 | `semantic_qualifier`、`semantic_pairwise`、`semantic_handshake`、`semantic_queue`、`semantic_interrupt`、`semantic_lock`、`semantic_credit`、`semantic_power`、`semantic_fault_recovery`、`semantic_reset_recovery` |
| 流与边界 | `role_stream`、`enum_transition`、`persistent_boundaries`、`register_boundary`、`timed_pulse`、`ordered_token` |
| 时序恢复 | `control_recovery` |
| 地址/内存 | `address_repeat`、`address_scale_search`、`context_transition`、`context_alias`、`scope_recovery` |
| 能力兜底 | `capability_field_sequence`、`capability_register_sequence`（无时序元数据、原生构造器够不着时用） |

有显式 `seq`/temporal 目标且能生成原生候选 → `generic_sequence_immediate = True`，立即进入搜索；
否则兜底候选只在停滞时逐个投放。

### 4.6 候选池的两道保险（泛化关键）

| 机制 | 触发条件 | 作用 |
|---|---|---|
| **零候选兜底** | 候选池为空 | 对每个非填充字段做合法值扫描 + 与请求/使能锚点配对。保证「候选数 ≥ 1」是硬保证 —— 上一版在 4 个假想 spec 里有 3 个候选数为 0，整条搜索链直接失效 |
| **组合候选** | 候选池 < `EDA_COMBINATORIAL_MIN_POOL`（默认 8） | 按 `{角色} × {值策略} × {时长}` 生成，并额外覆盖**角色未识别出来**的字段。覆盖度由说明书暴露的角色决定，而不是由人工枚举的候选族决定 |

组合候选**只在候选池贫瘠时启用**，这是实测结论：在候选池已经充足的 DUT 上叠加组合候选
会稀释周期预算（DMA 82→71、SPI 72→69），因为搜索层是「未执行的候选优先」，一次性候选
会挤掉有效候选的重试机会。

## 5. 在线循环

### 每个周期（`_GenericPolicy.predict`）

1. 更新覆盖统计：`covered`、`covered_bins`、`_last_gain_step`、`_last_coverage_gain_step`。
2. **队列非空 → 直接出队**，不做任何搜索计算（绝大多数周期走这条）。
3. 队列空（宏边界）→ 走下一节的调度。

### 每个宏边界

```
缺失目标 + 目标权重
   ↓
CoverageMacroScheduler.select()          选宏（UCB + 停滞 campaign）
   ↓
EpisodeManager.observe()                 收益/停滞 → 新 episode，强制插一个 recovery
   ↓
按宏挑候选目标（先联合候选、再依赖图置信度、再 index）
   ↓
① M8/M8.1 通用候选  →  ② M4 联合候选排序  →  ③ M3/M5/M6 模板事务
   ↓
程序编译成 (action, cycles) 队列，attach 到宏调度器做信用归因
```

## 6. 宏调度层（`generic_planner.py`）

四个宏：`configure` / `control` / `temporal` / `recovery`。

| 机制 | 规则 |
|---|---|
| 冷启动 | 四个宏各先跑一次 |
| campaign | 停滞时插入 3 种宏组合轮换：`(configure,temporal,control)` / `(control,temporal,recovery)` / `(configure,control,recovery)` |
| 停滞判定 | `patience = clamp(max_steps // 40, 128, 2048)` |
| UCB 打分 | `reward/cost + 0.08·sqrt(ln(total+1)/counts) + phase_bonus + 0.06·target_weight + 0.08·归一化模型Q` |
| 延迟信用 | 新 bin 奖励向最近 3 个宏按 `0.5^distance` 回传（覆盖率常滞后 1~2 个宏才出现） |
| Episode | 达到最小长度后有新覆盖 → `coverage_gain`；超 patience 无覆盖 → `stagnation`。两者都开新 episode 并 `force_next("recovery")` |

## 7. 事务生成层：配置型 DUT 的主力（M3 → M6）

| 里程碑 | 做什么 | 关键实现 |
|---|---|---|
| **M3** | 合法寄存器事务模板 | 按 spec 寄存器表生成「禁用 → 配置 → 使能 → 数据写入 → 等待」；所有 `_n` 输入默认去使能（这是 SPI 之前进不了传输态的主因） |
| **M3b** | cross bin 联合候选 | 协议 × TMOD、协议 × effective DFS 等编译进**同一次**合法写；修复了宏游标不推进、`SER` 首词匹配失败 |
| **M4** | 候选结果/成本自适应排序 | `JointCandidateRanker`：`(hit_rate + 0.35·sqrt(ln(total+1)/attempts), yield_rate, -target_index)`；结构门控仅当候选数 1~8 时生效，重试预算 3（≤8 个候选）/ 2 |
| **M5** | 字段选择完整事务 + 并发 campaign | 适用「资源选择 + 字段选择 + 写使能 + 数据 + 请求」接口：逐字段配置 → 安全长度覆写 → 请求上升沿 → 观察窗口；并发 campaign 先配多资源再连续请求；mode×burst 二维扫描；零长度/超长边界先写命中再覆写成可执行长度（防止资源被永久占用） |
| **M6** | 黑盒接受窗口鲁棒性 | 每个 selector 字段写**保持 16 周期**的有界幂等重试，跨过 RTL 内部 busy/hold 的写拒绝窗口（Python 模型与 RTL 的相位分歧由此修复） |

M5/M6 都由**接口结构门控**触发（是否同时存在 instance_select / register_address / write_enable /
request / data 字段），不满足的接口自动保持上一版行为，不读 DUT 名。

## 8. 通用搜索层：M8 与 M8.1

### M8（`generic_sequence_search.py`）

| 环节 | 规则 |
|---|---|
| 候选来源 | 第 4.5 节的原生族 + 能力兜底族；产生覆盖的 token/token 对/程序再派生时距与重复变体 |
| 选择 | 未执行候选优先（按 `priority`）；之后 `rate + 0.05·sqrt(ln(total+1)/attempts)`，对应 UCB |
| 重试 | 每个候选最多 2 次 |
| 激活 | 原生候选 → 立即；兜底候选 → 停滞 `clamp(max_steps//10, 1024, 4096)` 后投放，1~2 个探针后冷却 |
| 预算保护 | 最后 1/6 预算停止探索，回退既有策略；单程序周期受剩余预算 10% 限制 |
| 耗尽 | 候选用完后回退 M6 调度 |

### M8.1 轨迹学习

从**实际产生新 bin 的 M3~M6 完整事务**里抠出动作程序，派生出：重复 2 次、1/4/16 周期间隔、
请求保持 2/4 周期、至多两个字段邻域变体、相邻成功事务的正反顺序组合。

- 上限：32 条源轨迹，单派生程序 ≤ 2048 周期，最多 1024 个候选。
- **收益准入**（关键防退化机制）：当前 DUT 的原生/能力 M8 探针必须**先直接新增 bin**，
  学习变体才被允许执行；否则只留作影子候选。默认 2 次无收益熔断，每个直接收益 +2 次额度。
  （依据：早期无准入实验曾把 SPI xfer 从 55/86 打到 54/86。）

## 9. 已移除的家族专用组件（2026-09-26）

三块与「不读 DUT 名、不做家族路由」直接冲突的实现已整体移除：

| 组件 | 原内容 | 移除理由 |
|---|---|---|
| M7 状态序列模板 | branch/cache/watchdog 专用序列，含写死的密钥扫描与 tag 偏移计算 | DUT 家族专用知识；默认关闭时对默认路径零贡献 |
| `_SpiPolicy` / `_DmaPolicy` | 旧专家教师（`InferenceInterface` 从不选用） | 同样按 DUT 家族写死 |
| `neural_router.py` + 权重 + 训练/测试脚本 | 按规格文本把 DUT 路由到 DMA/SPI 家族 | 推理路径从未引用它，属家族路由概念本身 |

**验证**：移除后 7 个 DUT 的 5k local 回归与移除前**逐位一致**（覆盖率与归一化 AUC 完全相同），
包含 M7 原本专治的 branch 66/66、cache 56/75、watchdog 47/59。记录见
`results/m7_removal_regression_5k.json`，移除前完整状态见 git 提交 `3a53a95`。

## 10. 学习型组件与 LLM 的当前状态

| 组件 | 结构 | 现状 |
|---|---|---|
| `UniversalOptionModel` | 线性 `features → 4 option`，特征为 IR 角色计数、目标类型计数、难度权重与预算进度 | **当前 `approved=False`，运行时返回全 0，不生效** |
| `DeepSeekPlanner` | OpenAI 兼容客户端，`__init__` 一次调用，JSON schema 校验 + 范围/维度/有限性校验 | **默认禁用**（需 `DEEPSEEK_ENABLED=1` + key）。失败/超时/不合法响应一律降级为纯本地策略；逐周期路径不碰网络 |

### 已移除：神经覆盖控制器（2026-09-26）

`CoverageSetController`（Deep Sets：每 bin 13 维 → `phi 13→24`+ReLU → 均值池化(全量 ‖
未覆盖)+3 维全局 → `rho 51→32→4`，输出 4 个宏 Q 值）连同 `coverage_controller.npz`、
`coverage_q_controller.npz` 与 4 个配套脚本一并删除。两条理由：

1. **它是死信号**：输出的宏 Q 值经 `macro_scores=` 传进 `_GenericPolicy.predict()` 后，
   函数体从未读取该参数；宏打分里的 `learned_bonus` 读的是 `UniversalOptionModel`
   （当前恒为 0）。即每 32 周期做一次前向，对决策零影响。
2. **宏语义与通用策略冲突**：它的 4 个输出是 `basic/boundary/cross/temporal`，而通用
   策略的 4 个宏是 `configure/control/temporal/recovery`，两套语义无法直接对齐。

移除后 `predict()` 不再做周期性前向，覆盖率逐位不变（见第 13 节）。

## 11. 安全与合法性约束层

- `_base_action()`：active-low 输入与 `ready` 默认置 1，其余 0。
- `_sanitize()`：① 按 `requires`/`write_condition` 依赖**自动置位前置字段**；② 按 IR 逐字段 `clamp`；
  ③ `padding` 强制清零。所有来源（模板 / 搜索 / LLM 计划）都过这一层。
- `_apply_active_target()`：把当前目标 bin 的 signal/value 按同名/去前缀映射到动作字段。
- 未处理访问的 step 要把「命中/未命中」类信号置**哨兵值**，不能用 0。
- 32 位高边界值按 4 个字节 lane 精确发送，规避 `float32` 标量在 `2^24` 以上的舍入。

## 12. 开关与默认值

| 环境变量 | 命令行 | 默认 |
|---|---|---|
| `EDA_JOINT_CANDIDATES` | `--joint-candidates on/off` | **开**（M3b） |
| `EDA_ADAPTIVE_JOINT_RANKING` | `--adaptive-joint-ranking on/off` | **开**（M4） |
| `EDA_FIELD_TRANSACTION_TEMPLATES` | `--field-transaction-templates on/off` | **开**（M5） |
| `EDA_ROBUST_FIELD_WRITES` / `EDA_FIELD_WRITE_REPEATS` | `--robust-field-writes` / `--field-write-repeats` | **开** / 16（M6） |
| `EDA_GENERIC_SEQUENCE_SEARCH` | `--generic-sequence-search on/off` | **开**（M8） |
| `EDA_GENERIC_TRACE_LEARNING` | `--generic-trace-learning on/off` | **开**（M8.1） |
| `EDA_COMBINATORIAL_CANDIDATES` | —— | **开**（泛化加固） |
| `EDA_COMBINATORIAL_MIN_POOL` | —— | 8（候选池低于此值才补组合候选） |

即：**默认提交配置 = M8.1**。实验 JSON 每条记录都写 `algorithm_variant`，不靠文件名推断配置。

## 13. 实测结果（当前快照）

30k cycles，`results/planning_m8_1_final_*`：

| DUT | local | Verilator |
|---|---|---|
| dma_xfer_public | 85/92（AUC 0.882） | **92/92**（AUC 0.949） |
| spi_master_public | 75/120（AUC 0.583） | 75/120（AUC 0.583） |
| spi_xfer_public | 55/86（AUC 0.591） | 55/86（AUC 0.591） |

5k 短跑，`results/semantic_matrix_*`（含留出验证 DUT）：

| DUT | 结果 | AUC |
|---|---|---|
| branch_predictor_validation | **66/66** | 0.900 |
| tlb_mmu_validation | **78/78** | 0.890 |
| cache_ctrl_validation | 56/75 | 0.493 |
| watchdog_safety_validation | 47/59 | 0.602 |
| dma / spi_master / spi_xfer | 82/92 / 72/120 / 47/86 | 0.698 / 0.497 / 0.459 |

所有运行**非法动作数均为 0**。

**移除 M7 与家族路由后的回归**（`results/m7_removal_regression_5k.json`）：7 个 DUT 与上表
5k 数字逐位一致，证明这些组件原本不在默认路径上参与决策。

**移除神经覆盖控制器后的回归**（`results/coverage_controller_removal_regression_5k.json`）：
同样 7 个 DUT 逐位一致，`predict()` P99 仍低于 1.3 ms。

### 解析泛化指标（`tools/check_parsing_generalization.py`）

用 `synthesis/spec_corpus.py` 的 26 个合成 spec（覆盖寄存器 / 流式 / 队列 / 仲裁 / 地址翻译 /
状态机 6 类接口风格，以及全名 / 缩写 / 无编号 / 无意义命名 4 类命名风格，含 4 个退化用例）：

| 指标 | 加固前 | 加固后 |
|---|---|---|
| 字段角色召回 | 79.3%（69/87） | **100%（87/87）** |
| 候选产出达标率 | 50.0%（13/26） | **100%（26/26）** |
| 候选数为 0 的条目 | 13/26 | **0/26** |
| 平均候选数 | 6.9 | 48.6 |

同一批改动下，7 个自带 DUT 的 5k local 回归**逐位一致**（含 103 个字段的角色判定零变化）。

## 14. 已知边界

1. **公开包与赛题原文的 DUT 对不上**：题面是 AES / SPI 主设备 / 温控，仓库里是 DMA / SPI master / SPI xfer。当前所有数字只能当开发证据，正式镜像到手后必须重审计重跑。
2. **SPI 覆盖率偏低**（75/120、55/86），主要卡在需要精确时序的中断/abort-recovery/窗口类 bin，靠的是兜底候选而非专用序列。
3. **DMA local 85/92 vs RTL 92/92**：Python 模型与 RTL 在 hold/写接受时序上存在差异，以 RTL 结果为准。
4. **M8 的"通用性"是候选生成与反馈规则不依赖已知家族**，不等于已验证对任意未见接口有效；TLB 上的 78/78 是唯一的第四类结构证据，样本仍偏少。
5. **推理路径已不含任何 DUT 家族判定**：M7 序列、旧专家策略与家族路由 MLP 均已移除。若将来需要「性能上界」对照，可从 git 提交 `3a53a95` 取回。
6. **能力仍取决于说明书的结构化程度**：寄存器地图只被 SPI 系用上（DMA 的 spec 里连 `0x` 都没有），
   联合候选因此在 DMA 上为 0。角色判定已对命名容错，但"地址 + 位域"这类信息若只以散文出现，
   仍无法恢复。
