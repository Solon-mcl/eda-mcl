# 新算法执行进度

算法入口、累积版本与旧教师策略的关系见
`docs/ALGORITHM_VARIANTS.md`。实验 JSON 使用 `algorithm_variant` 标识实际
运行的版本，避免只凭文件名推断配置。

## 当前完成

### M0：接口与数据契约审计

- 新增 `tools/audit_interface_contract.py`，自动盘点本地公开/验证 DUT 的动作维度、字段数、metadata bin 总数、解析目标数和超出 `float32` 精确整数范围的覆盖值。
- `coverage_meta.json` 作为 covergroup 同目录的可选旁路输入；声明 bin 数、展开 bin 数或运行时状态长度不一致时，目标条件化安全失效，不继续使用错位索引。
- 已确认赛题文档列出的公开 DUT（AES、SPI 主设备、温控）与当前仓库公开目录（DMA、SPI master、SPI xfer）不一致。当前仓库实验只能作为开发证据。

### M1：统一评测基座

- `tools/run_experiments.py` 现在从 cycle 0 的初始覆盖开始记录，每 `interval` 个真实 cycle 采样。
- 输出官方风格覆盖曲线、积分 AUC、归一化 AUC、逐 bin 首次命中周期、初始化耗时、predict 均值/P99 和非法动作数。
- 100-cycle 本地 smoke 已验证曲线端点、AUC 范围和动作合法性。

### M2：覆盖依赖图原型

- 新增 `app/inference/coverage_dependency.py`。
- 图边区分 `direct` 与 `inferred`：只有 coverage signal 与 action field 精确同名才属于直接证据；词法和字段角色匹配保持较低置信度。
- 通用策略优先选择具有可控字段证据的未覆盖目标，同时保留未映射目标，避免不完整依赖图永久排除可达 bin。
- 依赖图已接入 `UniversalPolicy` 的目标选择，但尚未接入联合 cross 候选生成和经验边更新。

### M3：第一批合法事务模板

- 补充 `reg_wdata/register_data` 语义角色与 32 位上下界解析；支持寄存器地址范围（如 `0x18~0x3B`）。
- 所有 `_n` 输入在基础动作中默认去使能，避免外部低有效输入被误断言。这个修复是 SPI 从未进入传输状态的主要原因。
- 对存在 enable、control 和 data-port 寄存器的设计，按寄存器表生成“禁用→配置→使能→数据写入→等待”的完整事务。模板依据当前 spec 的寄存器名称和访问属性实例化，不按 DUT 名称路由。

### M3b：cross bin 联合候选

- 新增 `app/inference/joint_candidates.py`，从覆盖目标和寄存器位域 IR 编译联合约束候选。目前支持 `protocol × tmod` 与 `protocol × effective DFS`，并将协议、传输模式和数据位宽合并到同一次合法寄存器事务中。
- `semantic_ir.py` 可从寄存器描述提取位域，例如 `[4:0]=DFS, [7:6]=FRF, [8]=SCPH, [11:10]=TMOD`。协议映射和 DFS 编码均保留为显式条件，无法解析的条件会记录为 unresolved，不伪装成已满足。
- 联合程序沿用“禁用→联合配置→从设备选择→使能→数据写入→等待”模板，并记录候选名、完成次数、程序前后覆盖增量。修复了联合程序未推进宏游标以及 `SER` 首词匹配失败导致未选择从设备的问题。
- `EDA_JOINT_CANDIDATES=0/1` 支持同快照消融；当前默认开启。没有可编译联合候选的 DUT 自动回退到 M3，不改变调度。

### M4：候选结果/成本自适应排序

- `JointCandidateRanker` 从已完成程序的目标命中、新增 bin 和消耗 cycles 形成候选级统计；使用 UCB 保留未尝试候选，不修改任何模型权重。
- 每个仍未命中的候选只有有界重试预算，耗尽后强制回退 M3 基础事务，防止联合程序挤占完整预算。默认预算根据候选空间取 2 或 3，也可显式覆盖。
- 排序器使用结构门控：联合候选数为 1～8 时启用；候选更多时沿用已验证的 M3b 轮换。门控不读取 DUT 名称。
- 实验 JSON 现在包含 `algorithm_variant`、`algorithm_parameters`、精简 `program_trace` 和 `joint_candidate_stats`，可定位每个程序的参数、成本及覆盖增量。
- M4 已通过 5k、30k 和 30k×3 种子配对门禁，`EDA_ADAPTIVE_JOINT_RANKING` 默认开启；设置为 `0` 可复现 M3b。

### M5：字段选择完整事务与并发 campaign

- 语义 IR 新增 `instance_select` 角色与 packed-data 位域解析，可从 `data[15:0]=len` 一类描述恢复打包字段，不依赖 DUT 名称。
- 对“资源选择 + 字段选择 + 写使能 + 数据 + 请求”接口，生成“逐字段配置→安全长度覆写→请求上升沿→观察窗口”的完整事务。零长度和超长边界先写入命中覆盖，再覆写为短可执行长度，防止资源被永久占用。
- 所有宏共享同一个字段事务游标，避免 control/temporal 宏绕过配置直接请求。并发 campaign 先配置多个资源，再连续请求，覆盖冲突、仲裁解析、资源切换和快速完成。
- mode×burst 使用二维系统搜索：普通事务扫描低 burst，并发 campaign 提前扫描高 burst；32 位高边界通过 4 个字节 lane 精确发送，避免单个 float32 标量的舍入问题。
- M5 已通过 5k、30k 和 30k×3 种子配对门禁，`EDA_FIELD_TRANSACTION_TEMPLATES` 默认开启；不满足结构条件的 SPI 接口自动保持 M4 行为。

### M6：黑盒配置接受窗口鲁棒性

- 新增 `tools/compare_dma_backends.py`：同一条固定动作流从复位开始同时重放到 local 与 Verilator，逐周期比较 28 个可见信号、覆盖新增 bin，并保存首次分歧前后窗口。
- DMA 首次信号和覆盖分歧出现在第 3 周期：RTL 的配置观测因非阻塞赋值晚一拍；长度 1 事务在 local 当拍完成，而 RTL 先进入 ACTIVE、下一拍完成。
- 第 555 周期出现决定性相位分歧：公开 RTL 的完成事件保持为高，hold 窗口会再次进入；M5 随后的安全长度覆写落在 hold 窗口内而被拒绝，start 仍使用旧的超长配置，使多个通道长期占用。
- M6 对每个 selector 字段写执行默认 16 周期的有界、幂等重试，使不知道内部 busy/hold 信号的策略仍能跨越拒绝窗口，并让 RTL 配置观测稳定。该机制由接口结构触发，不读取 DUT 名称。
- `EDA_ROBUST_FIELD_WRITES=0/1` 和 `--robust-field-writes off/on` 支持 M5/M6 消融；`--field-write-repeats` 可修改重试周期，默认 16。

### M7：状态历史与认证时序序列

- 将此前存在但未接入提交调度的 branch/cache 状态事务生成器接入统一 `_GenericPolicy`；入口按动作字段角色和 spec 语义结构门控，不读取 DUT 名称。
- Branch 序列覆盖 PHT 饱和与翻转、历史模式、BTB 命中/替换、调用返回栈、stall 和 flush 恢复。
- Cache 序列覆盖重复命中、同 set 多 tag 的干净/脏替换、写后读、失效后重填、backpressure 和完整 flush，并系统扫描隐藏异常 tag。
- Watchdog 使用覆盖反馈发现未知有序密钥：`key_phase.waiting_b` 首次命中确定 key A；每个 key B 候选均在合法窗口中与已知 A 配对，`action_result.service_accept` 首次命中确定 key B。发现后生成 early/open/pretimeout/timeout、故障升级、锁定写拒绝和外部复位恢复序列。
- `EDA_STATEFUL_SEQUENCE_TEMPLATES=0/1` 和 `--stateful-sequence-templates off/on` 支持 M6/M7 消融。实验 JSON 记录结构门控是否实际应用、序列族以及两阶段密钥是否发现。
- M7 依赖 branch/cache/watchdog 的已知结构，现已降级为默认关闭的离线教师和性能上界，不作为最终通用方案。

### M8：通用覆盖反馈程序搜索

- 新增 `app/inference/generic_sequence_search.py`。候选只由语义 IR 字段角色、合法范围与枚举、覆盖目标序列、覆盖增量和程序成本生成与选择，不读取 DUT 名称，也不判断 branch/cache/watchdog 家族。
- 初始候选覆盖枚举重复与转移、地址关系与边界、多尺度地址扫描、stall/flush/ready 恢复、寄存器边界写、0～255 token 探测和多种 pulse 时长。产生覆盖的 token 会派生有序 token 对，成功 token 对会派生时距变体，成功程序会派生重复变体。
- 覆盖元数据没有 sequence 标签，或动作结构无法生成上述原生候选时，M8 仍会依据寄存器读写、字段选择和请求能力构造有界兜底候选。兜底候选在覆盖深度停滞后逐个投放，每次投放后冷却 1024～4096 周期，避免抢占 M3～M6 的正常事务预算。
- 搜索器先尝试未执行的原生候选，再按新增 bin/周期与 UCB 探索项排序；每个候选最多尝试两次。候选耗尽后自动回退 M6 调度。实验 JSON 记录候选总数、原生候选数、是否立即激活、完成程序数和新增 bin 数。
- `EDA_GENERIC_SEQUENCE_SEARCH=0/1` 和 `--generic-sequence-search off/on` 支持 M6/M8 消融，默认开启。M7 默认关闭；显式开启 M7 时保留专家教师优先级。

### M8.1：成功事务轨迹学习

- 从调度历史中提取实际产生新 bin 的 M3～M6 完整动作程序，派生重复 2 次、1/4/16 周期间隔、请求保持 2/4 周期、至多两个字段邻域变体，以及相邻成功事务的正反顺序组合。
- 最多使用 32 条成功源轨迹，派生程序限制为 2048 周期；学习候选具有较高的首次探索优先级，但仍受剩余预算 10% 上限、1024～4096 周期冷却以及最后六分之一预算保护。
- 学习候选使用收益准入。原生或能力 M8 探针必须先在当前 DUT 上直接新增 bin，学习候选才能从影子模式进入执行；默认两次无收益后熔断，每个直接学习收益增加两次尝试额度。
- `EDA_GENERIC_TRACE_LEARNING=0/1` 和 `--generic-trace-learning off/on` 支持 M8/M8.1 消融，默认开启。实验 JSON 记录 seed 数、学习候选数、准入收益、成功/失败计数以及学习程序直接新增 bin。

## 当前验证

- `tools/test_inference_contract.py`：通过。
- `tools/test_generic_temporal_planner.py`：通过。
- DMA 本地 100-cycle smoke：23/92，归一化 AUC 0.191576，非法动作 0，predict P99 约 0.88 ms。该短跑只验证评测管线。
- 5k 单种子公开回归：DMA 38/92、SPI master 61/120、SPI xfer 38/86；SPI master random 为 5/120、greedy 为 7/120。
- 30k 单种子公开回归：DMA 34/92（36.96%，AUC 0.3634）、SPI master 65/120（54.17%，AUC 0.5062）、SPI xfer 47/86（54.65%，AUC 0.4444）；三者非法动作均为 0，predict P99 均小于 1.2 ms。
- 旧规划记录的同为 30k、但不同代码快照的通用门禁结果是 32/92、5/120、2/86。新结果表明完整事务解决了主要 SPI 退化，但尚未达到旧专家 100% 上界；严格结论仍需同快照多种子对照。
- M3b 的 5k 同种子开关消融：DMA 保持 38/92；SPI master 从 61/120、AUC 0.4300 提升到 64/120、AUC 0.4700；SPI xfer 从 38/86、AUC 0.3700 提升到 47/86、AUC 0.4616。
- M3b 的 30k、3 种子消融（种子 260923～260925）：DMA 开关两侧均为 29.00±4.55/92、AUC 0.2889±0.0453；SPI master 从 65.00±0.00/120、AUC 0.4771 提升到 73.00±0.00/120、AUC 0.5299；SPI xfer 从 47.00±0.00/86、AUC 0.4196 提升到 55.00±0.00/86、AUC 0.5436。所有运行非法动作数为 0。
- 上述结果支持在当前公开本地 DUT 上默认开启联合候选，但不证明对赛题实际 AES、SPI master、温控镜像同样有效；进入真实镜像后必须复跑同样的开关消融。
- M4 对 M3b 的 30k、3 种子配对结果：DMA 每个种子完全持平；SPI master 每个种子均从 73/120 提升到 75/120，归一化 AUC 从 0.5299 提升到 0.5507；SPI xfer 因候选数为 12 自动回退 M3b，每个种子的终值和 AUC 完全持平。全部非法动作数为 0。
- M5 对 M4 的 30k、3 种子配对结果：DMA 基线分别为 34、30、23/92，M5 三个种子均为 85/92；平均终值从 29/92 提升到 85/92，归一化 AUC 从约 0.29 提升到约 0.85。SPI master 三个种子均保持 75/120，SPI xfer 均保持 55/86。全部非法动作数为 0，M5 P99 `predict()` 延迟为 0.25～1.01 ms。
- DMA 本地剩余 7 个 bin 全部依赖占位 `done_ok` 类别或其交叉；占位 secret 的 mode 20/21 超出本地四位解码，类别 0/2 不可达，类别 3 的方向定义也与 mode 7 不一致。因此 85/92 是当前本地占位参数下更合理的开发上界，不代表真实评测上界。
- Verilator 30k、3 种子配对：DMA 的 M4 结果为 32、37、23/92，M5 三个种子均为 68/92；平均终值从 30.67/92 提升到 68/92，归一化 AUC 从约 0.31 提升到约 0.68。SPI master 每个种子均保持 75/120，SPI xfer 均保持 55/86，非法动作均为 0。
- local 与 Verilator 的 DMA 终值分别为 85/92 和 68/92，说明 Python 模型与 RTL 在 hold/配置接受时序上存在差异；M5 相对 M4 的增益在两个后端都稳定，真实 RTL 结果作为更强证据。
- cache、branch predictor、watchdog 三个留出结构的 M4/M5 30k 配对分别保持 55/75、25/66、35/59；三者 `field_programs=0`，证明 M5 结构门控未误触发。
- 修复多个 Verilator DUT、多 repeat 同进程运行时的 `run_verilator` 模块搜索路径污染；双 repeat、三 DUT smoke 共 6 条记录全部完成。
- `tools/run_experiments.py --trace-actions` 可选输出压缩动作序列和安全空闲动作；默认关闭，不增加提交结果体积或在线开销。
- 新增 `tools/replay_program_trace.py`，按原周期精确重放轨迹，并用空闲动作替换整段程序或单个动作块进行删除试验。DMA 5k 的 49 个已完成宏精确重放出原结果 84/92；17 个有收益程序中抽查最高收益的 12 个，12 个整段删除均丢失覆盖，共识别 27 个不可删除动作块。
- 删除试验验证了主要因果关系：完整单通道事务产生完成/长度边界，正确 mode 事务产生 hold/done_ok/时序序列，并发 campaign 产生四通道 ACTIVE、冲突、连续仲裁和通道切换。报告同时保存 bin 索引与名称。
- DMA 同动作 5k 差分复现 M5 的 local 84/92、Verilator 68/92，并确认 16 个最终 bin 仅由 local 命中。M6 的 5k 配对结果为 local 80/92、Verilator 85/92；短预算 local AUC 因重复写成本下降，但 RTL 不再在第 573 周期停滞。
- M6 的 8/12/16/20 重试消融中，30k RTL 结果分别验证了 12 为 89/92、16 和 20 均为 92/92；16 的归一化 AUC 为 0.9486，高于 20 的 0.9417，因此默认选择 16。默认配置的 DMA 30k×3 种子回归中，Verilator 三个种子均从 M5 的 68/92 提升到 92/92、AUC 0.9486，local 三个种子均为 85/92、AUC 0.8821。使用 20 重试的完整公开 DUT 预回归中，SPI master 保持 75/120、SPI xfer 保持 55/86，两个后端完全一致，所有运行非法动作数为 0。
- M6 留出结构 30k 回归保持 cache 55/75、branch predictor 25/66、watchdog 35/59，证明新增重试没有越过字段选择结构门控。
- M7 的 5k 同种子开关消融：branch predictor 从 25/66 提升到 66/66，最后命中周期 238；cache 从 49/75 提升到 74/75，只剩隐藏 poison tag；watchdog 从 35/59 提升到 59/59，最后命中周期 1678。
- M7 的 30k 门禁：branch predictor 66/66、AUC 0.9833；cache 75/75、AUC 0.9580；watchdog 59/59、AUC 0.9658。公开 DUT 两个后端与 M6 完全一致，DMA local 85/92、DMA Verilator 92/92、SPI master 75/120、SPI xfer 55/86；公开 DUT 的 M7 结构门控均未触发，所有运行非法动作数为 0。
- M7 隐藏参数外推：两组 branch 的 history bits/salt/replacement/RAS depth 变体均为 66/66；两组 cache 的 replacement/latency/poison tag 变体均为 75/75，其中 tag 80 在第 28892 周期命中；watchdog 的 key `(0,1)` 变体为 59/59，最坏扫描 key `(255,254)` 成功发现两把密钥并达到 57/59。后者缺失的 `fault_count.two` 与 `two_normal` 在 `escalation_limit=2` 时不可达，因为第二次故障直接进入锁定类。
- 在明确关闭 M7 后，M8 的 30k 留出验证结果为 branch predictor 66/66（AUC 0.9833）、cache 75/75（AUC 0.8696）、watchdog 59/59（AUC 0.8963）。这说明三类满覆盖不再依赖 M7 的 DUT 家族模板。
- 放宽能力门控后的 M8 公开 DUT 30k 回归保持原终值：DMA local 85/92、DMA Verilator 92/92、SPI master 两后端均为 75/120、SPI xfer 两后端均为 55/86，所有运行非法动作数为 0。DMA、SPI master、SPI xfer 分别生成 16、40、40 个兜底候选；两个后端分别完成约 6～7 个停滞探针。当前这些探针没有直接新增 bin，说明它们已能参与未知结构探索，但公开 DUT 的主要覆盖仍由 M3～M6 完成。
- M8.1 在公开 DUT 30k 中分别提取约 14～22 条成功源轨迹并生成 138～218 个影子学习候选。三个公开 DUT 的能力探针没有直接新增 bin，因此收益准入没有放行学习重放；local 与 Verilator 的终值和 AUC 均与 M8 能力门控版一致，非法动作数为 0。早期无准入实验曾使 SPI xfer 从 55/86 降至 54/86，这一反例是默认启用收益准入的依据。
- 新增第四类留出 DUT `tlb_mmu_validation` 后，初始 M8.1 因未将 VPN 识别为地址通道而与 M6 同为 66/78，DUT 自带 greedy 为 78/78。语义 IR 增加 VPN、context/ASID、privilege、scope/global 和 fence/flush/root-change 角色后，M8 可生成地址重复、上下文别名以及 local/global 恢复序列。最终默认参数在 5k 达到 78/78、AUC 0.8872、最后命中周期 3021；两组隐藏参数变体也均在 5k 达到 78/78，最后命中周期分别为 3179 和 943。三组运行非法动作数均为 0。
- M8 隐藏参数外推中，两组 branch 均为 66/66，两组 cache 均为 75/75，watchdog key `(0,1)` 为 59/59。key `(255,254)` 且 `escalation_limit=2` 的变体为 57/59；缺失的两个 bin 在该参数下不可达，因为第二次故障直接锁定。
- `tools/run_experiments.py --secrets-json` 可接受开发用 JSON 对象或文件，只在结果中记录是否覆盖 secret，不回显参数内容；用于上述参数外推，不改变提交入口。
- 当前工作区未发现正式赛题 AES、温控或新的官方镜像，因此尚不能执行真实镜像接口审计。现有结果验证了 M8 对三类留出结构和参数变化的外推，但不能证明对任意新 DUT 都能满覆盖。

## 下一步

1. 在真实赛题镜像可用后，优先完成接口审计和 M4/M5/M6/M7/M8 开关复验，重点比较默认 M8 与默认关闭的 M7 教师，再决定是否进入 LLM 初始化与学习排序器实验。
2. 对真实镜像中仍无法由语义 IR 映射的 hardest sequence，评估 LLM 初始化和停滞重规划；只有真实未见结构消融稳定增益才默认启用。
