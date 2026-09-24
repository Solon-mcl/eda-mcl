# 消融实验报告

## 配置

在相同 Verilator、30,000-cycle 预算下比较四种配置：

1. `random`：组委会接口骨架的逐拍随机动作，用于移除全部事务语义和反馈规划。
2. `greedy`：组委会接口骨架的固定候选池与增量保留策略，用于保留简单反馈、移除完整协议规划。
3. `no learned controller`：保留全部协议策略，但移除离线拟合的覆盖控制器，使用固定启发式宏顺序。
4. `full`：本文方法，加入 DUT 语义路由、完整事务模板、边界/序列定向覆盖、隐藏参数探测和学习式宏排序。

## 终值覆盖率

| 配置 | DMA | SPI Master | SPI Xfer | 三 DUT 平均 |
|---|---:|---:|---:|---:|
| random | 50.000% | 4.167% | 2.326% | 18.831% |
| greedy | 52.174% | 5.833% | 2.326% | 20.111% |
| no learned controller | 100.000% | 100.000% | 100.000% | 100.000% |
| full | 100.000% | 100.000% | 100.000% | 100.000% |

## AUC

| 配置 | DMA AUC | SPI Master AUC | SPI Xfer AUC |
|---|---:|---:|---:|
| random | 11,966.89 | 1,245.79 | 691.84 |
| greedy | 14,706.00 | 1,737.44 | 691.84 |
| no learned controller | 27,368.57 | 22,857.34 | 25,853.65 |
| full | 27,368.57 | 24,715.67 | 26,993.19 |

结论：简单反馈保留对 DMA 有小幅帮助，但无法稳定完成寄存器配置、FIFO 装载和串行传输。事务语义是终值覆盖的主因；学习式宏排序不改变三项终值，但相对固定启发式顺序将 SPI Master AUC 提高 8.13%，SPI Xfer AUC 提高 4.41%。DMA 不使用该宏控制器，因此结果不变。

原始记录为 `experiments/ablation_random.json`、`experiments/ablation_greedy.json`、
`experiments/ablation_no_learned_controller.json` 和三次 full 运行 JSON。
