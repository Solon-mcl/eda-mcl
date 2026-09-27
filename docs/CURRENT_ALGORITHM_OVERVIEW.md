# 当前算法整理（代码快照：2026-09-27，泛化加固一轮之后）

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

**描述文本要先去掉声明行才可用**。`action = [...]` 那一行会列出**全部**字段名，若把它算作
字段的"描述"，每个字段的语义文本里就塞着整张信号表，任何基于描述的判断都会失真（实测：
启用「描述含 asid 即判 context_id」后，TLB 的有效覆盖从 78 崩到 6）。现在分两层取文本：

| 层 | 取法 | 用途 |
|---|---|---|
| `description` | 所有**提到**该字段的行 | 数值域、位宽、枚举、低有效标记 |
| `keyed_description` | 只有**以该字段为主键**的行（`- \`x\`: …` 或 `\| \`x\` \| …`），**支持区间主键**（`- \`vpn0..vpn3\`: …` 属于区间内每个成员） | 角色判定的第二来源 |

**角色判定有两级**：先按名字走上面那条优先级链；只有落到 `scalar` 时才用 `keyed_description`
再判一次。这样名字无信息（`f0..f15`）但散文仍然描述功能的说明书也能恢复角色 —— 而名字判得
出来时行为完全不变（7 个自带 DUT 的 103 个字段角色零变化）。

**lane 分组由 IR 角色驱动，不再用第二套正则**。同一个缺陷犯过两次：`va0` 角色判成地址通道
却进不了 `addr_lanes`（候选池 72 → 24）；描述恢复的角色同样进不去。现在成员资格只看角色，
数字后缀只决定字节序，`_lane_indices(..., role=...)` 一处定义。

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
| **组合候选** | 候选池 < 8，**或**（有寄存器地图 且 联合候选路径不可用） | 按 `{角色} × {值策略} × {时长}` 生成，并额外覆盖**角色未识别出来**的字段。覆盖度由说明书暴露的角色决定，而不是由人工枚举的候选族决定 |

第二个条件是 2026-09-27 加的：**有寄存器地图，但联合候选为空或超过 M4 排序器的上限**，
意味着结构化事务路径恰好在该用的地方用不上，此时生成候选是补位而不是抢预算。实测只改变了
spi_xfer（**47 → 56，+9 bins**），DMA（没有寄存器地图）、SPI master（排序器可用）与四个
验证 DUT 全部逐位不变。

**为什么必须是硬门控、而不是在搜索层调平衡**：把这批候选放进富池子，四种搜索策略都试过，
全部变差或无效 ——

| 搜索层做法 | 结果 |
|---|---|
| 与原生候选混在同一池 | DMA 82 → 71、SPI master 72 → 69 |
| 只给「首次试验」一个周期预算 | DMA → 72、cache 56 → 43、watchdog 47 → 39 |
| 作为更低的搜索层级（原生试满才轮到它） | DMA → 71、SPI master → 64 |
| 重试优先给已产生覆盖的候选 | **完全无变化**（448 → 448），已删除该规则 |

结论：在候选本来就够的 schema 上，多出来的候选是**净亏**，与排序无关。所以门控留在供给侧。

### 4.7 在线结构探测（`EDA_INTERFACE_PROBE`，默认关）

`field_selected_interface` 是一个**静态**判断：五个角色（资源选择 / 寄存器地址 / 写使能 /
请求 / 数据）齐全才启用字段事务模板。在线探测是它的替代方案：开局花一次有界的预算，走一遍
寄存器写扫（地址 0..3 × 值 {0, 0xFFFFFFFF}，再脉动一次可用的触发类角色），按**实际覆盖反馈**
决定是否撤销静态门控。

在 7 个自带 DUT 上的实测（探测对 spim / spi_xfer / watchdog 三个包是 armed 的）：

| 后端 | 结果 |
|---|---|
| local 5k | bin 数 7/7 不变；spi_master AUC 0.4967 → **0.5117**、spi_xfer 0.2674 → **0.2698**（前置覆盖的功劳） |
| verilator 30k | **变差**：spi_xfer 62 → 51、spi_master 75 → 74 |

同时，**"判定有收益就撤销门控"这条分支在所有自带 DUT 上都返回 `no_gain`**，一次都没触发 ——
它要撤销门控去启用的，正是那个按 DMA 结构特化的模板。因此默认关闭，只保留开关与这组证据：
RTL 是权威口径，local 的 AUC 小涨不足以抵消 RTL 的 -11 bins。这属于「DUT 家族知识必须靠
实测准入」的同一条纪律。

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
| `LLMEnricher` | 同上传输层，构造期一次请求，产出**可校验的结构化假设** | **默认禁用**（需 `EDA_LLM_ENRICH=1` + key）。见 §10.1 |

### 10.1 LLM 富化：模型提假设，算术与覆盖率裁决（2026-09-27）

局部解析器是关键字驱动的，它能从命名和散文里恢复角色，但**不能推理**一个没见过的接口。
实测缺口恰好都是这一类：spi_master 的 16 个 cross bin **只编译出 7 个**、4 个 sequential
bin 全缺、残留的角色召回缺口是关键字表没预料到的散文。`LLMEnricher` 只针对这三件事提问。

**三道边界（都是硬性的）**

| 边界 | 做法 |
|---|---|
| 网络 | 只在构造期发一次请求；`predict()` 永不碰网络（离线测试断言 opener 未被调用） |
| 范围 | 逐项校验：字段必须已声明、角色必须在 `SEMANTIC_ROLES`(38) 内、地址必须在寄存器地图内、值 ≤ 2³²−1、序列步长与总长有上限。**胡说的寄存器到不了 DUT** |
| 淘汰 | 候选级假设交给在线搜索，按覆盖率两次无收益即淘汰 |

**两类假设、两种命运（实测）**

| 假设 | 去处 | 结果 |
|---|---|---|
| `joint_writes` | 走**已有的、预算感知的**联合候选路径 | **spi_master 5k：72 → 81/81/83（3 seed，均值 +9.67）；30k：见 §13** |
| `sequences` | 进**扁平候选池**（被扫描而非被预算） | **spi_xfer 5k：56 → 38/46/47（3 seed，均值 −12.3）** |

消融（只开 joint）：spi_master **+9.67**，spi_xfer **+0.00**。结论：**增益全来自 joint，
伤害全来自序列**，故 `EDA_LLM_SEQUENCES` 默认关闭并把这次测量记在代码旁边。

**一条设计教训（我踩过）**：LLM 的 joint 候选最初是**先合并、后算门控**，于是 spi_master 的
联合候选从 7 变 13，**越过了 M4 排序器的 ≤8 阈值**、又触发了我自己那条"联合路径不可用"的
生成候选门控 —— 一个假设偷偷改掉了两个已标定的决定，第一次 A/B 测出来是 **−1**。
改成**门控先算、LLM 候选后追加**（与生成候选族同一模式）后变成 **+9**。

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
| `EDA_COMBINATORIAL_MIN_POOL` | —— | 8（候选池低于此值才补组合候选；设很大可强制"总是补"用于 A/B） |
| `EDA_INTERFACE_PROBE` | —— | **关**（在线结构探测，见下） |
| `EDA_INTERFACE_PROBE_MIN_ROLES` | —— | 3（探测所需的最少接口角色数） |
| `EDA_LLM_ENRICH` | —— | **关**（LLM 富化，见 §10.1；需 key） |
| `EDA_LLM_SEQUENCES` | —— | **关**（序列假设，实测有害，见 §10.1） |
| `EDA_LLM_MODEL` | —— | **`deepseek-flash`**（端点只提供 `deepseek-flash` 与 `deepseek-v4-pro`；默认值由实测选定，不是按参数规模） |
| `EDA_LLM_THINKING` | —— | 关（实测更差且引入截断/超时，见 §13 的思考模式消融） |
| `EDA_LLM_THINKING_RESERVE` | —— | 6144（开思考时额外预留的 token，思考 token 从同一预算里扣） |
| `EDA_LLM_ENRICH_CACHE` | —— | 未设（把模型**原始响应**缓存到指定文件；有缓存时不需 key 也不联网，校验照常运行） |
| `EDA_LLM_JOINT_MAX_HOLD` | —— | 64（joint 假设的 `hold_cycles` 上限 —— 它是纯空转等待，直接吃预算） |
| `EDA_LLM_TEMPERATURE` | —— | 0 |
| `EDA_LLM_ENRICH_MAX_TOKENS` | —— | 8192（撞上限即判 `truncated`） |
| `DEEPSEEK_ENABLED` | —— | **关**（旧的程序规划器；与 `EDA_LLM_ENRICH` 相互独立） |

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

所有运行**非法动作数均为 0**（JSON 字段名是 `invalid_actions`）。

**2026-09-27 泛化加固一轮之后**（`results/description_role_regression_5k.json`，5k local）：

| DUT | 加固前 | 加固后 |
|---|---|---|
| dma_xfer_public | 82/92 | 82/92 |
| spi_master_public | 72/120 | 72/120 |
| **spi_xfer_public** | 47/86 | **56/86** |
| cache_ctrl_validation | 56/75 | 56/75 |
| branch_predictor_validation | 66/66 | 66/66 |
| watchdog_safety_validation | 47/59 | 47/59 |
| tlb_mmu_validation | 78/78 | 78/78 |
| **合计** | 448 | **457（+9）** |

**移除 M7 与家族路由后的回归**（`results/m7_removal_regression_5k.json`）：7 个 DUT 与上表
5k 数字逐位一致，证明这些组件原本不在默认路径上参与决策。

**移除神经覆盖控制器后的回归**（`results/coverage_controller_removal_regression_5k.json`）：
同样 7 个 DUT 逐位一致，`predict()` P99 仍低于 1.3 ms。

### LLM 富化实测（`results/llm_enrichment_trial.json`，deepseek-flash）

消融：**只开 joint** vs **joint+序列** vs 基线。

用 `EDA_LLM_ENRICH_CACHE` 固定同一份假设后测（此前每次运行重新采样，数字不可复现）：

| DUT | 后端/规模 | 基线 | 开 joint | Δ | AUC |
|---|---|---|---|---|---|
| **spi_master** | local 5k | 72/120 | 81 / 79 | **+9 / +7** | 0.4967→0.5525 |
| **spi_master** | local 30k | 75/120 | 84 / 76 | +9 / +1 | 0.5829→0.6750 |
| **spi_master** | **verilator 30k** | 75/120 | **84/120** | **+9** | 0.5829→**0.6750** |
| spi_xfer | local 5k ×3 | 56/86 | 54/86 | **−2** | 0.2674→0.2977 |
| spi_xfer | local 30k ×3 | 62/86 | 60/86 | **−2** | 0.5841→0.6198 |
| dma / cache / branch / watchdog / tlb | local 5k | — | 不变 | **+0** | — |

三条结论：

1. **增益全部来自 `joint_writes`**；`sequences` 实测有害（spi_xfer −12.3），默认关。
2. **5k 的增益稳定（+7~+9），30k 的增益取决于抽到哪份响应**：同一 prompt 三次采样给出
   5k +9/+7、30k +1/+9/+13。端点即使 temperature=0 也不确定，所以缓存是让结果可复现的前提。
3. **spi_xfer 用不了这条路**：它拿到的 8 条假设**技术上是对的**（手工重放策略排出的动作序列
   确实命中目标 bin 69/63/74），但模型挑的是「**本地编译器编不出来**的目标」，不是
   「**这一轮还没覆盖**的目标」—— 它 8 个目标里有 3 个基线早就覆盖了，所以程序基本冗余，
   却照样占用联合程序预算。**结论：spi_xfer 不开富化。**

**零回归**：富化接线但关闭时，7 个自带 DUT 5k local **逐位一致**。

### 思考模式消融（同一 prompt，spi_master local 5k ×3 seed）

| 配置 | 覆盖率 | Δ | 状态 | AUC |
|---|---|---|---|---|
| 基线（关富化） | 72/72/72 | — | — | 0.4967 |
| **思考关** | **82/81/81** | **+10/+9/+9（均值 +9.33）** | ok/ok/ok | 0.5550/0.5792/0.5792 |
| 思考开，预算 8192 | 83/72/83 | +11/**0**/+11（+7.33） | ok/**truncated**/ok | 0.5925/0.4967/0.5925 |
| 思考开，预算 12288 | 79/72/72 | +7/**0**/**0**（+2.33） | ok/**超时**/**超时** | 0.5342/0.4967/0.4967 |

**思考模式不用，三个理由**：

1. **假设数没变多** —— 完成的那些 seed 依旧只有 6 条 joint 写，与思考关时完全相同。
2. **更差且更不稳** —— 均值 +2.33 vs +9.33；同一个 seed（260923）两边都完成时，思考开只有 +7，思考关有 +10。
3. **引入两个静默失败模式**。思考 token 从**同一个 `max_tokens` 预算**里扣，所以 8192 的上限一开思考就截断（`completion_tokens` 正好等于上限，JSON 残缺 → 0 假设）。给足预算后不再截断，但 45 s 超时又被打爆。

修了其中确定是缺陷的那部分（`EDA_LLM_THINKING_RESERVE`，默认 6144），但思考仍然默认关 —— 理由 1 和 2 与预算无关。

**顺带得到一个真实环境下的 fail-closed 证据**：两个超时的 seed，覆盖率**正好等于关富化时的 72/120（AUC 0.4967）**。这正是离线测试断言的性质（模型不可达 → 策略与无提示完全一致），这次是在真实端点上观测到的。

### 解析泛化指标（`tools/check_parsing_generalization.py`）

用 `synthesis/spec_corpus.py` 的 26 个合成 spec（覆盖寄存器 / 流式 / 队列 / 仲裁 / 地址翻译 /
状态机 6 类接口风格，以及全名 / 缩写 / 无编号 / 无意义命名 4 类命名风格，含 4 个退化用例）：

| 指标 | 加固前 | 加固后 |
|---|---|---|
| 字段角色召回 | 79.3%（69/87） | **100%（87/87）** |
| 候选产出达标率 | 50.0%（13/26） | **100%（26/26）** |
| 候选数为 0 的条目 | 13/26 | **0/26** |
| 平均候选数 | 6.9 | 48.6（默认门控）／61.0（强制总是补组合候选的 A/B 配置） |

同一批改动下，7 个自带 DUT 的 5k local 回归**逐位一致**（含 103 个字段的角色判定零变化）。

### 命名无关性（`tools/check_name_invariance.py`）

固定电路、覆盖模型与 harness，**只换 spec 文本** —— 落差即解析造成的：

变体由 `synthesis/make_spec_variants.py` 生成（可复现）：

| spec 变体 | 覆盖率 | 落差 |
|---|---|---|
| 原始 spec | 78/78 | 基准 |
| 散文式改写（字段名不变） | 78/78 | **0** |
| 短名与业务同义词（`valid→vld`、`vpn→va`、`priv→lvl`、`pad→rsv` …） | 77/78 | **-1** |
| 无意义**信号名**、散文保留（现实情形） | 78/78 | **0** |
| 无意义名且散文一起被改写（对抗性下界） | 76/78 | -2 |

**"无意义信号名但散文保留"是从 -24 收敛到 0 的**：靠的正是 §4.1 的第二级角色判定
（`keyed_description`）+ 角色驱动的 lane 分组。现实中未公开内部信号名的 DUT 仍然会写明
"little-endian virtual page number""privilege level"，这一格才是这类 DUT 的真实刻度。

修的过程中发现一个真 bug：**角色判定与 lane 分组用的是两套独立正则** —— `va0` 被正确判成
地址通道，却因为 `_lane_indices` 没跟上而进不了地址分组，于是所有地址类候选消失
（候选池 72 → 24）。对齐两套正则后，短名变体从 **-16 恢复到 -1**；改成角色驱动后，这类
"判对了却用不上"的缺陷整类消失。

### 覆盖率缺口诊断（30k RTL，2026-09-27）

**先说结论：DMA 已满覆盖（92/92），「低」的只有 SPI 这一对。** 而且缺的 bin **不是不可达** —— 
我用手写脚本实测：**spi_master 缺失的 45 个里有 26 个（58%）被 27 次「写该值 → 跑一次传输」的
组合拿下了**，其中包括最复杂的 `hold_last_frame`（需要多帧 + 最后一帧 + HOLD_SS 驻留）。

| DUT | 30k RTL | 缺失 | 缺的是什么 |
|---|---|---|---|
| dma_xfer_public | **92/92** | 0 | —— |
| spi_master_public | 75/120 | 45 | 34 个「配置值 ∧ 传输进行中」、4 个 HOLD_SS 驻留、3 个 RX 真收到数据、3 个 abort→恢复、1 个多帧最后一帧 |
| spi_xfer_public | 62/86 | 24 | 15 个「配置值 ∧ 传输进行中」、6 个 abort→恢复、2 个 RX 真收到数据、1 个跨帧完成 |

**`_active` 是主开关**：覆盖模型对绝大多数 bin 要求
`active and signal == value`（`_active` = FSM 不在 IDLE/SLEEP）。只写寄存器、不启动传输，
一分覆盖率都拿不到。

四个具体原因，按影响排序：

**① 派生信号没有对应字段 —— spi_master 的 `protocol`。** 它的解码是
`frf = CTRLR0[7:6]`、`scph = CTRLR0[8]`，然后

| bin 名 | 实际需要 |
|---|---|
| `protocol_mode.spi0` | `frf=0 且 scph=0` |
| `protocol_mode.spi1` | **`frf=0 且 scph=1`**（靠 bit 8，不是 FRF 字段） |
| `protocol_mode.ssp` | `frf=1`（且传输需要 `ss_in_n=0`） |
| microwire | `frf=2`（`protocol` 值为 3，**没有对应 bin**） |

把 `1`/`2` 写进 FRF **并不产生** "spi1"/"ssp"。由于没有一个字段叫 `protocol`，位域反查
编译不出它 → `protocol_x_*` 目标永远留在 unresolved → **spi1/ssp 模式从不运行** →
挂在它们 FSM 路径上的 `fsm_state.hold_ss`、`hold_ss_cnt_boundary`（4）、`hold_last_frame`、
`mwpop`、`wait_ready`、`clear_ready` 全部连带缺失。这一条单独解释了约 20 个 bin。
（spi_xfer 没有这个问题：它的 `protocol` 就是 `frf` 本身，所以它的 6 个 `protocol_x_tmod`
本地就编译出来了。）

**② 程序不组合「配置值 + 传输」。** 策略的程序是单一用途的：一个程序做值扫描，另一个程序
配好 + 使能 + 推数据 + 等待。而覆盖模型要的是**两者同时成立**。实测把这两件事写在一起
（`dfs=3/4/8/16/31`、`tmod=1/2/3`、`ser=2/8`、`mwcr=4/7` 各配一次传输）就能拿下
`eff_dfs_boundary` 5 个、`tmod.tmod_1`、`protocol_x_eff_dfs_boundary` 3 个、
`protocol_x_tmod.ssp_tmod2/3` 等。

**③ 需要刺激本身产生时序模式 —— 这一类是真的能力缺口。**

| 需求 | 现状 |
|---|---|
| RX 真收到数据（`rx_data_boundary` 3、`risr_boundary` 3、`rx_fifo_level_boundary.rx_full`） | 要按 sclk 节拍逐位驱动 `rxd`（要 `0x55`/`0xAA` 就得逐位交替）。策略把 `rxd` 当静态标量，**没有「逐周期串行模式」这个概念** |
| abort→恢复（`seq_a` 3 + spi_xfer 6） | 要在传输中途翻转 `ss_in_n` 极性触发 abort，再跑一次完整传输 |
| `mst_collision`（DCO L 冲突） | 需要特定的片上冲突条件 |

**④ 值域边界没覆盖到**：`eff_cfs_boundary.cfs_12/15`（`ctrlr0[19:16]`）、
`mwcr_boundary.mwcr_full`、`ss_sel.slave_2` 属于这类，机制上同 ②。

**要点**：`eff_dfs = dfs`（**没有夹取**），所以 `dfs_3/4/8/16/31` 都是直接可写的；
只有 `eff_cfs = max(cfs_raw, spec_cfs_min)` 有夹取（本地 `spec_cfs_min=8`），
因此 `cfs_12/15` 可写、而「CFS 被夹到隐藏最小值」那个 condition bin 需要 `cfs_raw < 8`。

**缺口的性质判断**：②④ 是**策略的组合能力**问题（可修，代价是程序更长、更吃预算）；
③ 是**动作词汇**问题（当前宏不会生成逐周期串行模式和多阶段 abort 序列，需要新的程序族）；
① 是**解析/编译**问题（派生信号无法反查，需要把「信号 = 多个位域的组合」编译出来）。

## 14. 已知边界

1. **公开包与赛题原文的 DUT 对不上**：题面是 AES / SPI 主设备 / 温控，仓库里是 DMA / SPI master / SPI xfer。当前所有数字只能当开发证据，正式镜像到手后必须重审计重跑。
2. **SPI 覆盖率偏低**（75/120、55/86），主要卡在需要精确时序的中断/abort-recovery/窗口类 bin，靠的是兜底候选而非专用序列。
3. **DMA local 85/92 vs RTL 92/92**：Python 模型与 RTL 在 hold/写接受时序上存在差异，以 RTL 结果为准。
4. **M8 的"通用性"是候选生成与反馈规则不依赖已知家族**，不等于已验证对任意未见接口有效；TLB 上的 78/78 是唯一的第四类结构证据，样本仍偏少。
5. **推理路径已不含任何 DUT 家族判定**：M7 序列、旧专家策略与家族路由 MLP 均已移除。若将来需要「性能上界」对照，可从 git 提交 `3a53a95` 取回。
6. **能力仍取决于说明书的结构化程度**：寄存器地图只被 SPI 系用上（DMA 的 spec 里连 `0x` 都没有），
   联合候选因此在 DMA 上为 0。角色判定已对命名容错，但"地址 + 位域"这类信息若只以散文出现，
   仍无法恢复。
7. ~~描述文本被动作声明行污染~~ —— **已修**（2026-09-27）：先剥离 `action = [...]` 占用行，再分
   `description` / `keyed_description` 两层取文本，见 §4.1。
8. **搜索层的"探索/利用失衡"被证伪，不是当前瓶颈**：四种改法（混池、首次试验周期预算、分层、
   有效候选优先重试）全部测过，三种更差、一种完全无变化（详见 §4.6 表）。真正的约束是
   **周期预算本身**——生成候选与结构化路径抢的就是同一份预算，所以只能靠供给侧的结构门控。
   下一步若要提升，方向不是搜索策略，而是**减少候选的周期成本**（例如更短的探测程序、更早
   判定无效并中断）。
9. **在线结构探测默认关闭**：local 5k 上 AUC 小涨、verilator 30k 上 spi_xfer 掉 11 bins；
   且它要撤销门控去启用的模板本身是按 DMA 结构写的，违反"不引入家族专用知识"。见 §4.7。
10. **泛化证据的强度（2026-09-27 审计）—— 这是最该盯着的一条**：
    「不依赖家族知识」已做到且可验证；「对任意未见 DUT 有效」**尚无证据**。
    7 个 DUT 全部来自本仓库，合成语料与 spec 变体都是本项目自己写的 —— 等于用自定义的
    "风格多样性"测自己。**从未在任何独立来源的未见 DUT 上跑过**，闭环缺一个外部样本。
    审计中还查出两处家族知识残留并已清除（描述规则里的字面短语 `"tlb fence"`、
    `kind_indices` 里的 `"branch_kind"` 死别名）；**仍有一处未决**：`deepseek_planner.py`
    的 `KNOWN_FAMILIES = {dma, spi_master, spi_xfer, generic}` 与 prompt 的 `family_hint`
    取值表（默认关闭，未启用）。
11. **LLM 的收益只在一个 DUT 上被证实，而且幅度依赖抽到哪份响应**：spi_master 5k +7~+9、
    RTL 30k +9（固定一份采样；另一份采样 30k 只有 +1，更早一份是 +13）。其余 6 个 DUT **全部 +0**。
    spi_xfer 是**负的（−2）**，原因不是假设错，而是**目标选错了**（见 §13 第 3 条）。
    正确读法：**对「有寄存器地图、位域反查有缺口、且这批 bin 还没被其它机制覆盖」的 DUT 有效** —— 
    条件比原先写的更苛刻。`sequences` 类假设实测有害（−12.3），默认关。
    端点不确定 ⇒ 任何结论都必须用 `EDA_LLM_ENRICH_CACHE` 固定假设后才可复现。
12. **LLM 依赖网络与延迟**：评分环境若不可达，富化会退化为纯本地策略（离线测试断言此时候选池与
    无提示时完全相同），所以开着不会变差；但没有网络就等于这条能力不存在。这一条已在**真实端点**
    上观测到 —— 超时的跑次覆盖率正好等于关富化时的 72/120。
    默认模型是 **`deepseek-flash`**（由实测选定，端点另提供 `deepseek-v4-pro`）；
    **思考模式实测更差**（均值 +2.33 vs +9.33），且思考 token 与输出共享 `max_tokens`，
    开它会引入截断与超时两个静默失败模式，故默认关。
13. **剩余缺口的机制是"驱动不到 / 写不准"，不是"认不出电路"**（30k verilator 实测缺 bin 分型）：

    | DUT | sequential | cross | boundary | basic | condition |
    |---|---|---|---|---|---|
    | spi_master | **4/4（100% 全缺）** | 11/16（69%） | 19/47（40%） | 8/27 | 3/26 |
    | spi_xfer | 7/10（70%） | 7/20（35%） | 7/30（23%） | 2/18 | 1/8 |

    缺的名字集中在两类：`fsm_state.hold_ss / mwpop / wait_ready / clear_ready`（**多周期协议
    中间态**，要把 DUT 驱动到特定状态）与 `protocol_mode.spi1 / ssp`、`tmod.tmod_1`、
    `eff_dfs_boundary.dfs_*`、`baud_div_boundary.baudr_1`（**寄存器字段写入精度**，必须写进
    特定位域且取到边界值）。后一类正是联合候选该解决的，但 spi_master 16 个 cross bin 只
    编译出 7 个联合候选 —— **位域反查只覆盖了一部分条件**，这是下一步最具体的着力点。
