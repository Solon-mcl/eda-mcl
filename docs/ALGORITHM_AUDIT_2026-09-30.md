# 当前算法复核（2026-09-30）

本记录使用工作区当前代码、默认推理开关、seed 260923、30,000 cycles、每 1,000 cycles 采样。`full` 不启用云端 LLM。所有新运行记录在 `results/audit_20260930_*.json`。没有重新打包。

## 可复现的本地结果

| DUT | full 终值 | full AUC | random 终值 | random AUC | 包内 greedy 终值 | greedy AUC |
|---|---:|---:|---:|---:|---:|---:|
| dma_xfer_public | 85/92 | 0.882 | 46/92 | 0.396 | 43/92 | 0.436 |
| spi_master_public | 102/120 | 0.796 | 5/120 | 0.041 | 7/120 | 0.057 |
| spi_xfer_public | 77/86 | 0.843 | 2/86 | 0.023 | 2/86 | 0.023 |
| branch_predictor_validation | 66/66 | 0.983 | 55/66 | 0.808 | 66/66 | 0.983 |
| cache_ctrl_validation | 75/75 | 0.866 | 64/75 | 0.828 | 75/75 | 0.983 |
| tlb_mmu_validation | 78/78 | 0.982 | 55/78 | 0.693 | 78/78 | 0.983 |
| watchdog_safety_validation | 59/59 | 0.896 | 27/59 | 0.450 | 59/59 | 0.983 |
| dma_desc_engine_validation | 85/87 | 0.938 | 27/87 | 0.305 | 87/87 | 0.983 |

`full` 合计 627/663。八项 `invalid_actions` 均为 0。这里的 greedy 是各包自带的参考实现；验证包的 greedy 能迅速满覆盖，但使用包内定制知识，因此只作为可达性参照。公开 DUT 的历史 Verilator 结果不能与本次本地结果混列：当前 Windows 主机无法运行包内 Linux 格式的 Verilator 可执行文件（WinError 193）。

## 修正旧结论

先前报告给描述符 DMA 标出的 AUC 0.443 不代表当前代码。本次同一 30k 本地口径是 **85/87，AUC 0.938**；5,000 cycles 已达到 85/87。因此当前无需将它列为首要的早期收敛优化对象。余下 `step_result.err_clear`、`seq_error_recover.hit` 两个 bin 与规格未说明的控制信号相关。

## SPI 缺口与一次预算消融

SPI master 缺 18/120，SPI xfer 缺 9/86。缺口包括串行接收数据、特定中间态、多帧序列、波特率与部分交叉覆盖。SPI master 的联合候选执行 54 次、新增 89 bin；SPI xfer 执行 33 次、新增 68 bin。两包的通用序列搜索分别执行 4、5 次，直接新增均为 0。下一步应定位缺失目标的具体前置时序和候选程序，而不是扩大通用候选池。

联合候选失败重试上限的 30k 本地对照：

| 上限 | SPI master | AUC | SPI xfer | AUC |
|---:|---:|---:|---:|---:|
| 2（默认） | 102/120 | 0.796 | 77/86 | 0.843 |
| 3 | 100/120 | 0.784 | 78/86 | 0.855 |
| 4 | 102/120 | 0.789 | 77/86 | 0.851 |

两枚额外种子（260924、260925）在这组试验中逐项给出相同结果；当前路径对种子基本确定，不能把这些重复当成独立随机性证据。重试 3 的收益只在 xfer 出现且使 master 退化，默认值保持 2。

## 泛化与验证状态

- `test_inference_contract.py` 通过。
- `check_parsing_generalization.py` 通过：字段角色 87/87，候选产出 26/26。
- `check_name_invariance.py` 的最差规格改写仍比原规格少 2 bin；该检查报告非零差距，不能称作完全命名不变。
- 仓库没有独立来源、从未用于开发或调参的 DUT 包。现有 validation 目录已参与算法迭代，不能作为真正盲测；外部盲测仍是未完成的验证项。

## 本次代码修正

`tools/run_experiments.py` 的结果元数据原将 `adaptive_joint_max_candidates` 默认值写成 8，而实际推理器默认值是 160。已对齐为 160，避免后续按错误配置复盘。这一处元数据修正本身不改变算法决策。

## 目标驱动多周期程序（本次后续改动）

上表是改动前基线。随后新增两个可关闭的通用候选族：

1. **整字寄存器目标**：覆盖 signal 与规格中的可写、无位域寄存器精确同名，且目标值是可执行数值时，构造“关闭使能 → 支持配置 → 写目标值 → 使能 → 推送数据 → 观察”的程序。已覆盖的目标不再试探；只读寄存器不生成写候选。
2. **单比特输入波形**：规格明确描述单比特串行输入，覆盖目标含对应数据值时，从联合候选复用合法配置，按目标值生成多周期输入波形。只有第一类程序已直接新增 bin，才允许试这一类。

两类候选均在前 1/4 预算后才投放；连续无收益时停止试探。`EDA_TARGET_CONDITION_PROBES=0` 与 `EDA_TARGET_INPUT_WAVEFORMS=0` 可分别关闭，默认均为 1。它们没有 DUT 名称分流。

30k 本地、相同 seed 与采样口径的 A/B：

| DUT | 改动前 | 改动后 | AUC 前 → 后 |
|---|---:|---:|---:|
| spi_xfer_public | 77/86 | **79/86** | 0.843 → **0.861** |
| spi_master_public | 102/120 | 102/120 | 0.796 → 0.796 |
| dma_xfer_public | 85/92 | 85/92 | 0.882 → 0.882 |
| branch/cache/tlb/watchdog 验证包 | 均不变 | 均不变 | 均不变 |
| dma_desc_engine_validation | 85/87 | 85/87 | 0.938 → 0.938 |

SPI xfer 新增 `baudr_1`、`baudr_3` 和 `rx_0x55`，但少了较晚命中的 `seq_b_0`，终值净增 2。两类候选单独启用时曾使 SPI master 或 xfer 退化；当前准入门控是本次 A/B 的必要部分。新结果保存在 `results/target_gated_20260930_*_local_30k.json`，默认开关的复核在 `results/target_default_20260930_spi_xfer_public_local_30k.json`。

契约、通用时序规划与解析泛化检查通过；命名不变性仍有最多 2 bin 的旧差距。由于没有独立来源的盲测 DUT，以上只能证明当前八包的局部收益和零回归，尚不能证明新机制对未知来源也有收益。公开 RTL 的本次复核仍受 Windows/Linux 可执行文件不兼容限制。

## 新增 ECC DUT：先盲测，后开发

`ecc_memory_validation` 加入工作区后，首次读取其 RTL 和包内参考策略之前，用当时代码、30k cycles、seed 260923、本地后端测得：默认策略 **51/72，AUC 0.6891**；关闭目标候选仍为 51/72，AUC 0.6891；random 为 70/72，AUC 0.9384；包内 greedy 为 72/72，AUC 0.9833。原始记录为 `results/blind_20260930_ecc_memory_*_30k.json`。这组数据是该 DUT 的首次盲测，下面的改进已使用 ECC 规格和行为模型，不能再称盲测。

原因是故障位 `bit_a`、`bit_b` 被当作布尔量，且通用候选缺少完整的“写入、注入、读取或 scrub”时序。现按规格中的 32 位字宽推断位位置范围，新增可用 `EDA_FAULT_SITE_PROGRAMS=0` 关闭的通用故障位程序。程序遍历地址和多组故障位，分别探索双位错误、停顿后的单比特读取修正，以及不经读取、直接由 scrub 扫描修正的路径；没有按 DUT 名称分流。

同一代码下仅关闭或开启新程序的中间对照是 **51/72、AUC 0.6812 → 68/72、AUC 0.9241**。随后补齐 scrub 路径和故障位组合，最终默认参数达到 **72/72、AUC 0.9764**；两组改变 syndrome、scrub 顺序、修正延迟、DBE 返回值和 poison 地址的参数分别达到 **71/72、AUC 0.9627**。结果见 `results/ecc_fault_final2_20260930_*_local_30k.json`。最终默认运行中，该程序族完成 24 次，直接新增 58 个 bin，非法动作 0。

复核发现，上述两个 71/72 并非调度退化：各自唯一缺失的 `addr_x_error.a0_double`、`addr_x_error.a7_double` 恰好对应隐藏的 `poison_word=0`、`poison_word=7`。本地模型在 poison 地址发生双位错误时输出 `error_class=3`，而这两个交叉 bin 要求 `error_class=2`；因此在该参数下不可达。把 poison 地址固定在 6，仅改变其余隐藏参数的两组运行均为 **72/72、AUC 0.9764**。

曾实现一次基于未覆盖地址的最多三次定向重试，并用 `EDA_FAULT_TARGET_RETRY` 在 WSL 内显式设置开关，做了 5 组参数 × 开关两侧、每次 30k 的 A/B。默认及两组可达参数两侧均为 72/72、AUC 0.9764；两组 poison 参数两侧均为 71/72、AUC 0.9627。开启重试确实对缺口地址多执行一次程序，但新增 bin 为 0。该重试逻辑已撤回，未把不可达目标的重复尝试并入默认算法。有效对照记录在 `results/ecc_target_retry_valid_20260930_*_local_30k.json`；未正确传递 WSL 环境变量的先前同名无 `valid` 文件不用于结论。

撤回后再跑旧 8 个 DUT 的 30k 本地回归，覆盖率和 AUC 均与上节一致，记录在 `results/ecc_retry_audit_20260930_*_local_30k.json`。下一步泛化评估应把隐藏参数下的可达 bin 集合作为诊断指标；仅凭覆盖率向量无法从有限尝试证明某个 bin 不可达，因此不要据此给默认策略加入无限或过多重试。

## SPI master 缺口机制实验（同日后续）

沿用本地、seed 260923、30k cycles 的 102/120、AUC 0.7961 基线，逐项检验 18 个缺失 bin。缺口集中在 Microwire 握手状态、DFS 边界、接收数据、FIFO 溢出和若干交叉／序列。联合候选对 `rx_0xAA`、`rx_all_ones8/32` 各尝试两次，目标命中均为 0；通用序列搜索只执行 4 次、直接新增 0 bin。`rx_all_ones32` 候选把 32 位 payload 送往数据端口，但仍使用 8 位帧配置，这是一个可解释的候选参数缺口；单独修正该参数仍未产生净收益。

| 机制消融 | SPI master 终值 / AUC | SPI xfer 终值 / AUC | 判断 |
|---|---:|---:|---|
| 基线 | 102/120 / 0.7961 | 79/86 / 0.8610 | — |
| 放开串行波形候选准入 | 100/120 / 0.7953 | 79/86 / 0.8610 | master 退化 |
| 按 payload 位宽配置帧长 | 101/120 / 0.8026 | 79/86 / 0.8607 | 终值退化 |
| 寄存器 FIFO 填满后读空 | 102/120 / 0.7572 | 76/86 / 0.7919 | 预算竞争明显 |
| 延后 FIFO 候选 | 97/120 / 0.7296 | 80/86 / 0.8337 | master 大幅退化 |
| 在原联合候选中补 TX 溢出写入 | 101/120 / 0.8004 | 79/86 / 0.8610 | 新增 TX overflow，但失去更多 bin |
| RX 目标后追加读回 | 99/120 / 0.7924 | 78/86 / 0.8581 | 两包退化 |

曾有一个末段 FIFO 候选试验得到 103/120、AUC 0.7963；追查 `program_trace` 后确认新候选根本没有执行，新增 bin 是候选池变化引起的排序波动。另两组改变 SPI master 隐藏参数的开关对照分别保持 93/120、AUC 0.7390 和 99/120、AUC 0.7857，均未获益。这些数据不支持将末段候选纳入默认算法。所有试验开关代码已撤回，相关 JSON 保存在 `results/spi_*_20260930_*.json` 和 `results/spi_late_fifo_secret_20260930_*.json`。

本轮结论：这些缺口不能靠扩大候选池或简单延长程序解决。下一轮应在不改变在线预算分配的离线回放中，先验证候选程序的前置条件、有效写入和首个失败状态，再将证实有效的通用程序加入在线搜索。

随后完成了这项离线回放。单独从复位状态执行 `rx_0xAA`、`rx_all_ones8` 的联合程序，观测信号确实出现 170、255；同一程序在 30k 在线轨迹中首轮执行时，`rx_0xAA` 全程没有 `rx_push`。首轮在线程序先在禁用期间把 170 写到 DR 地址 24，启用后却因轮转选择了另一 DR 别名地址 41，并写入探测值 85/170/255；候选目标值没有被一致地送进传输。更关键的是，当次配置将 TX 启动阈值写成 7，却只发送 3 至 4 帧，无法达到规格规定的 `TX level > TXFTLR` 启动条件。另一个候选在配置分频 16 后只有 64 周期观察窗，也不足以保证收帧。

按上述因果链，仅在联合候选第一次失败后的重试中固定 DR 别名、分频 2、阈值 0，`rx_0xAA` 和 `rx_all_ones8` 均被命中；但同预算结果为 **SPI master 101/120、AUC 0.8051**（基线 102/120、0.7961），SPI xfer **79/86、AUC 0.8672**（基线 79/86、0.8610）。master 的 TX/RX FIFO 满相关 bin 丢失，终值净回退 1。其他组合试验见 `results/spi_payload_*_20260930_*_local_30k.json`。在线改动已撤回，保留现有默认策略；这说明候选“可执行”与“加入后净提升”是两个独立验收条件。

### 同前缀反事实回放

新增 `tools/compare_candidate_programs.py`，并让 `tools/run_experiments.py --trace-actions` 额外保存完整逐周期动作流。此前 `replay_trace` 只包含已结束的宏，SPI master 的 30k 轨迹缺少开头和末尾合计 10 个周期，不足以核对同预算 AUC。新工具要求替换程序与原程序等长，核验替换前的完整动作前缀、目标宏动作、基线终值、AUC 和采样曲线；可分别固定原后续动作，或使用 `--adaptive` 让策略从相同前缀按反馈继续调度。

以第一次 `rx_data_boundary.rx_0xAA` 候选为例：第 2992 周期开始，原程序和替换程序均为 97 周期，前缀覆盖 89 bin，前缀哈希已记录在 `results/spi_rx_aa_counterfactual_adaptive_20260930_30k.json`。替换程序使用原候选目标值，修正 DR 别名、分频和 TX 阈值。固定原后续动作到 30k，基线 **102/120、AUC 0.7961**，替换后 **103/120、AUC 0.8035**，仅新增 `rx_0xAA`。从同一前缀让在线策略自行续跑，则为 **101/120、AUC 0.8021**：新增 `rx_0xAA`，失去 `rx_fifo_level_boundary.rx_full` 和 `risr_boundary.rx_full`。这直接证明净回退来自反馈后的调度变化，不是修正程序本身未触发目标。

原始逐周期轨迹为 `results/spi_master_counterfactual_source_20260930_30k.json`，等长替换程序为 `results/spi_rx_aa_ready_replacement_20260930.json`。调用方式：`python3 tools/compare_candidate_programs.py --input <trace.json> --target-name rx_data_boundary.rx_0xAA --occurrence 1 --replacement <program.json> --adaptive --output <report.json>`。该工具仅供本地开发评估；在线推理无法复制评测 DUT 的隐藏状态，因此不能直接使用反事实分支作运行时 oracle。下一步应收集多个 DUT、多个候选的净收益标签，再训练或设计不依赖 DUT 名称的准入条件。

旧 8 个 DUT 用 30k、本地后端回归，终值和 AUC 均与上节目标驱动改动后的记录一致；见 `results/ecc_fault_regression_20260930_*_local_30k.json`。通用时序测试、推理契约测试、解析泛化检查均通过。ECC 包只提供本地 harness，因此这些数据不是 Verilator 仿真结果。用户本轮只要求先跑这一个新增 DUT，另外两个新增包尚未进入此轮测试；本轮未打包。

## 2026-10-01：声明式候选变换与批量反事实标签

新增 `tools/batch_candidate_counterfactuals.py`。它从外部接口声明读取写使能、地址、数据维度及使能、分频、启动阈值、数据端口寄存器，不依据 DUT 名称选择程序。当前提供四种等长变换：对齐数据端口别名和目标 payload、把启动阈值降到待发送帧数以下、使用规格允许的最小有效分频、把设置阶段的多余空闲周期移到末尾观察窗。变换若不适用或没有实际改动会跳过；无法由 float32 动作精确表示的目标 payload 也会拒绝。两个 SPI 接口声明见 `tools/interfaces/`，数值来自各自公开规格。本工具是离线开发工具，声明目前由人工核对规格填写，并非在线自动解析结果。

对每个检查点，工具仍调用 `compare_candidate_programs.py` 的完整基线复现校验，分别记录固定原后续动作与在线自适应续跑到 30k 的 bin、AUC、目标首命中，并保存替换程序。新增 `features_before` 与 `features_after` 包括预算位置、覆盖数、目标是否已覆盖、写入数、DR 别名数、payload 种数、阈值余量、分频及末尾观察周期。这些特征只使用检查点前已知的程序与覆盖，不使用反事实结果。早期批次的报告还没有特征字段，带特征的复核见 `results/spi_master_hidden_rx_aa_featured_20261001.json`。

使用 SPI master 默认参数的完整动作轨迹，分别评估 `rx_0xAA`、`rx_all_ones8` 各两个检查点；使用 SPI xfer 默认参数轨迹评估 `rx_0x55` 两个检查点；另用 SPI master `spec_cfs_min=10, spec_hold_ss=6, spec_txftlr_dflt=3` 生成独立轨迹并评估 `rx_0xAA` 两个检查点。后者基线 **99/120，AUC 0.7874**；SPI xfer 基线 **79/86，AUC 0.8610**。原始轨迹、隐藏参数和 4 个批次报告分别为 `results/spi_xfer_counterfactual_source_20261001_30k.json`、`results/spi_master_hidden_source_20261001_30k.json`、`results/spi_master_hidden_case_20261001.json`、`results/spi_rx_aa_batch_counterfactual_20261001.json`、`results/spi_rx_ff8_batch_counterfactual_20261001.json`、`results/spi_xfer_rx_55_batch_counterfactual_20261001.json`、`results/spi_master_hidden_rx_aa_batch_20261001.json`。

| 轨迹 / 目标 | 检查点 | 三项组合：固定续跑 bin 差 | 三项组合：自适应 bin 差 |
|---|---:|---:|---:|
| SPI master 默认 / `rx_0xAA` | 1、2 | +1、+1 | -1、0 |
| SPI master 默认 / `rx_all_ones8` | 1、2 | +1、+1 | -1、-2 |
| SPI xfer 默认 / `rx_0x55` | 1、2 | 0、0 | -2、+1 |
| SPI master 改隐藏参数 / `rx_0xAA` | 1、2 | +1、+1 | -1、+1 |

四批次共 31 个实际改变动作的候选：自适应终值净增 2、持平 22、净减 7。这些候选来自相同的少数轨迹，不能作为 31 个独立样本。SPI xfer 的 `rx_0x55` 在基线最终已覆盖，其第二检查点的 +1 是 `seq_b.seq_b_0`，不是接收目标新增。组合修正在固定动作下经常有效，但在线调度有时会失去 FIFO 或序列 bin；不能仅依据目标命中决定准入，也不能凭这组数据训练一个可信的泛化模型。默认在线算法保持不变。下一轮应在更多独立 DUT 或参数配置上采集轨迹，按 DUT 留一验证准入规则；训练时按轨迹分组，避免同轨迹检查点泄漏。

复现示例：`python3 tools/batch_candidate_counterfactuals.py --input results/spi_master_counterfactual_source_20260930_30k.json --interface tools/interfaces/spi_master_register_bus.json --target-name rx_data_boundary.rx_0xAA --occurrences 1,2 --adaptive --output results/<name>.json`。隐藏参数运行需同时传 `--secrets-json`，且源轨迹必须用同一配置生成。WSL 中 `python3 -m py_compile tools/batch_candidate_counterfactuals.py` 通过；带特征的隐藏参数检查点复核仍为固定 +1、自适应 +1。未打包。
