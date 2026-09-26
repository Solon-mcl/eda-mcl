# 算法入口与版本关系

## 提交入口

评测只导入 `app/inference/InferenceInterface`。它始终实例化
`UniversalPolicy`；`UniversalPolicy` 继承 `_GenericPolicy`。当前提交算法是
累积实现，不是四套互相竞争的入口：

1. M1 提供评测记录，不改变策略入口；
2. M2 在通用策略中加入覆盖目标和依赖图；
3. M3 在同一通用策略中加入合法事务模板；
4. M3b 在 M3 上加入可开关的 cross bin 联合候选。
5. M4 根据候选结果/成本进行结构门控排序。
6. M5 为“资源选择 + 字段选择 + 数据 + 请求”接口编译完整事务和并发 campaign。
7. M6 为可能在内部 busy/hold 窗口拒绝配置的黑盒接口加入有界字段写重试。
8. M8 从语义 IR 编译候选程序，并按实际覆盖增量与周期成本在线搜索。
9. M8.1 从产生覆盖的 M3～M6 事务中提取程序并生成反馈约束的序列变体。

`coverage_dependency.py`、`semantic_ir.py`、`generic_planner.py`、
`joint_candidates.py` 和 `generic_sequence_search.py` 分别负责依赖图、语义
IR、宏调度、联合候选编译和通用程序搜索，最终由 `_GenericPolicy` 组合执行。

## 旧算法

2026-09-26 移除了三块与泛化目标冲突的遗留实现：`_SpiPolicy` / `_DmaPolicy`
旧专家教师、`neural_router.py` 家族路由 MLP（含训练/测试脚本与模型权重），
以及 M7 的 branch/cache/watchdog 状态序列模板。`InferenceInterface` 从不
选择它们，公开 DUT 目录中的 `inference_interface.py` 也只在实验工具选择
`random` 或 `greedy` 基线时加载，均不属于提交路径。移除前状态见 git 提交
`3a53a95`。

## 可复现实验

实验工具在每条 JSON 记录中写入 `algorithm_variant`：

- `m3_legal_transactions`：M3，关闭联合候选；
- `m3b_joint_candidates`：M3b，开启联合候选；
- `m4_adaptive_joint_ranking`：M3b 加候选结果/成本自适应排序；
- `m5_field_selected_transactions`：M4 加字段选择完整事务与并发 campaign；
- `m6_backend_robust_transactions`：M5 加字段写接受窗口鲁棒性；
- `m8_generic_sequence_search`：默认提交路径，M6 加通用覆盖反馈程序搜索；
- `m8_1_trace_learning`：当前默认提交路径，M8 加成功事务轨迹学习；
- `baseline_random` / `baseline_greedy`：公开基线。

推荐显式使用命令行开关，避免环境变量造成歧义：

```bash
python3 tools/run_experiments.py --agent full --joint-candidates off \
  --output results/m3.json
python3 tools/run_experiments.py --agent full --joint-candidates on \
  --output results/m3b.json
python3 tools/run_experiments.py --agent full --joint-candidates on \
  --adaptive-joint-ranking on --output results/m4.json
python3 tools/run_experiments.py --agent full --field-transaction-templates on \
  --robust-field-writes on --field-write-repeats 16 --output results/m6.json
python3 tools/run_experiments.py --agent full \
  --generic-sequence-search on --output results/m8.json
python3 tools/run_experiments.py --agent full \
  --generic-sequence-search on --generic-trace-learning on \
  --output results/m8_1.json
```

`--joint-candidates default` 保留环境变量 `EDA_JOINT_CANDIDATES` 的值；若
环境变量也未设置，则采用提交默认值 `on`。

M4 排序层在 5k、30k 和 3 种子配对消融通过最终覆盖率与 AUC 门槛后已默认
开启，环境变量为 `EDA_ADAPTIVE_JOINT_RANKING`。设置为 `0` 可复现 M3b。
失败重试预算默认按联合候选空间调整：候选数不超过 8 时为 3，否则为 2；
可用 `--joint-max-failed-attempts` 显式覆盖，实际值写入实验 JSON。
当前结构门控只在联合候选数为 1～8 时应用排序器；更大的候选空间自动回退
M3b 轮换，防止排序开销延迟早期覆盖。门控是否实际应用同样写入 JSON。

M5 已通过 5k、30k 和 30k×3 种子配对门禁，当前默认开启。设置
`EDA_FIELD_TRANSACTION_TEMPLATES=0` 或使用
`--field-transaction-templates off` 可复现 M4。模板只在语义 IR 同时识别到
资源选择、字段选择、写使能、数据和请求字段时生效，其余接口保持 M4 行为。

M6 当前默认开启。设置 `EDA_ROBUST_FIELD_WRITES=0` 或使用
`--robust-field-writes off` 可精确回退 M5。默认对每个字段写保持 16 个周期，
可用 `--field-write-repeats` 调整；开关和实际重试数均写入实验 JSON。重复写
只在 M5 的字段选择接口结构门控通过后执行，不影响普通寄存器接口。

M7（branch/cache/watchdog 专用状态序列）已于 2026-09-26 整体移除。它依赖
DUT 家族专用知识，与“不读 DUT 名、不做家族路由”的泛化目标冲突，因此不再
作为可开启选项保留。移除后 7 个 DUT 的 5k 回归与移除前逐位一致，见
`results/m7_removal_regression_5k.json`。

M8 当前默认开启。设置 `EDA_GENERIC_SEQUENCE_SEARCH=0` 或使用
`--generic-sequence-search off` 可精确回退到 M6。
M8 不读取 DUT 名称，也不判断 branch/cache/watchdog 类型。它只使用语义 IR 中
的字段角色、合法范围与枚举、可选的覆盖目标序列信息，以及执行后新增 bin 和周期
成本。候选包括枚举重复与转移、地址关系和边界、多尺度地址扫描、控制恢复、寄存器
读写序列、字段选择事务、token 探测及 pulse 时长。产生覆盖的 token、token 对和
程序会派生时距或重复变体。

地址角色同时支持普通地址、PC 和 VPN 字节通道。上下文接口可识别 context/ASID、
privilege、scope/global、fence/flush/root-change 等角色，并生成地址重复、上下文切换、
别名以及 local/global 与恢复操作的组合序列。这些规则适用于带标签缓存、IOMMU 和
虚拟内存类接口，不读取 DUT 名称。

带显式时序目标且能生成原生候选的接口立即进入搜索。没有时序元数据、或现有原生
构造器无法覆盖其动作结构的接口，也会从寄存器、字段选择和请求能力生成候选；这类
兜底候选只在深度停滞后投放一个，随后至少冷却 1024～4096 周期。搜索器对原生候选
先保证每个候选至少尝试一次，再按“新增 bin/周期 + UCB 探索项”选择；每个候选最多
执行两次。候选耗尽后回退 M6 基础调度。M8 的通用性指候选生成与反馈规则不依赖已知
DUT 家族；对真正未见接口的效果仍需用正式镜像或新的第四类 DUT 复验。

M8.1 默认开启，可用 `EDA_GENERIC_TRACE_LEARNING=0` 或
`--generic-trace-learning off` 精确回退 M8。它从新增覆盖的非 M8 完整事务中提取
动作程序，生成重复、事务间隔、请求保持、字段邻域以及成功事务顺序组合变体。最多
保留 32 条源轨迹，单个派生程序不超过 2048 周期。派生程序按剩余预算过滤，最后
六分之一预算停止探索。

轨迹变体采用收益准入：当前 DUT 的原生或能力 M8 探针必须先直接产生新 bin，学习
变体才允许执行。默认允许两次无收益尝试，每个学习收益增加两次后续尝试额度。未被
准入的变体保留为影子候选，不改变动作流。这避免仅凭相关性重放 M3～M6 事务而扰动
已有策略。
