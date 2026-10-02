# 目标任务调度试验（2026-10-02）

## 改动与选择

原策略先选择 `configure/control/temporal/recovery` 宏，再在宏内选择覆盖目标。新增 `TargetTaskScheduler`，在程序边界读取已完成程序的覆盖收益，从所有仍缺失、可由联合寄存器候选执行的目标中选择任务；四类宏仍用于程序承载和其他接口的后备搜索。程序、目标和反馈均来自规格与覆盖元数据，不依据 DUT 名称分流。

最初不限目标种类的每 4 次程序插入一次任务的试验，SPI master 从 102/120 降到 97/120，SPI xfer 从 79/86 降到 76/86。每 12 次插入一次并按“多个数据端口别名”的接口能力门控后，SPI master 恢复 102/120、AUC 从 0.7961 到 0.8111，但被调度的 4 个目标程序直接新增 0 bin；这说明整体 AUC 改善仍可能只是顺序扰动，不能作为新结构有效的证据。

最终方案进一步把任务选择限定在**候选写入规格声明的数据端口地址**的目标。只有解析出的数据端口有至少两个别名时启用；其余接口走原路径。对这类数据目标，程序使用候选写入的同一个端口别名和 payload，并将支持配置中的分频置为 2、TX 启动阈值置为 0。它每 12 个程序边界最多占用一个任务槽；其余时间仍由原调度器运行。两个开关 `EDA_TARGET_TASK_SCHEDULER=0`、`EDA_TARGET_TASK_PROGRAM_FIX=0` 可分别关闭调度和程序修正，`EDA_TARGET_TASK_PERIOD` 可改变任务槽间隔。当前三者默认值为 1、1、12，且都受接口能力门控。

代码入口：`app/inference/target_tasks.py` 记录目标尝试、命中、周期成本和覆盖点组；`app/inference/generic_planner.py` 支持在程序完成后由目标选择器指定宏；`app/inference/__init__.py` 实施接口门控、目标选择和数据端口程序修正。`tools/run_experiments.py` 记录生效状态、直接新增 bin 和逐程序上下文。旧反事实轨迹生成时没有新开关，`tools/compare_candidate_programs.py` 现在按源轨迹的参数恢复策略设置，旧轨迹复核仍为固定续跑 +1、自适应 -1。

## 30k 本地对照

所有结果使用 seed 260923、local harness、相同 1000 周期采样、非法动作 0。SPI master 的两组新增隐藏参数并未用于宽泛调度的频率选择；其余参数仍属于同一 DUT 家族，不能算独立盲测。

| SPI master 参数 | 原默认覆盖 / AUC | 仅修程序覆盖 / AUC | 最终方案覆盖 / AUC | 任务程序直接新增 |
|---|---:|---:|---:|---:|
| 默认 | 102/120 / 0.7961 | 101/120 / 0.8063 | **102/120 / 0.8158** | 4 |
| 低：CFS 4、HOLD 2、TX 阈值复位 0 | 94/120 / 0.7475 | 95/120 / 0.7554 | **96/120 / 0.7642** | 3 |
| 中：CFS 10、HOLD 6、TX 阈值复位 3 | 99/120 / 0.7874 | 100/120 / 0.7972 | **100/120 / 0.7994** | 4 |
| 高：CFS 12、HOLD 8、TX 阈值复位 5 | 99/120 / 0.7857 | 100/120 / 0.7983 | **100/120 / 0.8000** | 4 |

最终默认方案的 9 个可用 DUT 本地回归合计 **701/735**；旧 8 个合计 **629/663**，ECC **72/72**。只有 SPI master 的多别名数据端口触发新调度；其他 8 个 DUT 的覆盖与 AUC 保持原值。SPI master 终值没有增加，AUC 提高 0.0197；另外三组隐藏参数终值均未退化。记录见 `results/task_focus_public_final_20261002_30k.json`、`results/task_focus_*_validation_final_20261002_30k.json`、`results/task_focus_{default,low,mid,high}_20261002_30k.json`。低、高参数文件见 `results/spi_master_hidden_{low,high}_20261002.json`。

`test_inference_contract.py`、`test_generic_temporal_planner.py`、`check_parsing_generalization.py` 均通过；旧反事实轨迹自适应复现也通过。本地验证包已参与开发，没有独立来源的新 DUT 可用于真正留一或盲测。因此这里证明的是已知接口族的局部泛化和旧包零回归，不能推断任意未知 DUT 都会获益。当前未打包。
