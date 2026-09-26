# 神经网络泛化方案与真实 RTL 离线 Q 学习

> **状态（2026-09-26）**：本文记录 M8 通用策略之前的混合专家架构，保留作为历史记录与
> 实验数据出处。其中的 **DUT 路由 MLP（第 3.1 节）已随家族路由组件一并移除**，
> 推理路径不再做任何 DUT 家族判定。当前算法的完整说明见
> [`CURRENT_ALGORITHM_OVERVIEW.md`](CURRENT_ALGORITHM_OVERVIEW.md)。

## 1. 总体结构

评测期 `predict()` 只进行前向推理，不更新权重。当前方案分为四层：

1. 可选 DeepSeek-V4 在 DUT 初始化时解析 spec/covergroup，为未知 DUT 生成结构化事务计划；API 失败时自动跳过。
2. ~~文本 MLP 根据 `dut_spec.md` 和 `covergroup.svh` 选择 DMA、SPI Master 或 SPI Xfer 策略族。~~（该家族路由 MLP 已移除，推理不再做 DUT 家族选择）
3. Deep Sets 将数量可变、顺序不固定的覆盖 bins 编码成状态，并输出四个宏动作的 Q 值。
4. 确定性事务解码器将 `basic / boundary / cross / temporal` 宏展开成合法的逐周期总线动作。

这种分层保留协议合法性和已有覆盖下限，同时允许 LLM 处理未见过的规格语义，并由神经网络根据 coverage schema 与实时反馈改变宏执行顺序。LLM 不在逐周期路径中运行，详细约束见 [`CURRENT_ALGORITHM_OVERVIEW.md`](CURRENT_ALGORITHM_OVERVIEW.md)。

## 2. 相关工作

- [Efficient Stimuli Generation using Reinforcement Learning in Design Verification](https://arxiv.org/abs/2405.19815) 将仿真器、刺激和覆盖反馈建模为环境、动作和奖励。
- [Design2Vec](https://research.google/pubs/learning-semantic-representations-to-verify-hardware-designs/) 使用学习得到的 RTL 语义表示预测覆盖并生成测试。
- [Deep Sets](https://arxiv.org/abs/1703.06114) 给出了可变长度、排列不变集合的共享编码与池化形式。
- [VerilogReader](https://arxiv.org/abs/2406.04373) 展示了语义理解、覆盖反馈和专用生成器结合的分层结构。

## 3. 两个神经网络

### 3.1 DUT 路由 MLP（已移除）

规格文本经稳定 signed hashing 形成 256 维 unigram/bigram 特征：

```text
Linear(256, 48) -> ReLU -> Linear(48, 3)
-> Softmax {DMA, SPI Master/Microwire, SPI Xfer}
```

训练集包含 1500 个经过删除、遮蔽、重命名和重排的规格样本。训练/验证准确率均为 100%。对 300 个额外扰动样本的路由准确率为 100%，最低置信度 99.98%。这验证的是同协议规格扰动鲁棒性，并不等价于对任意新协议的泛化。

### 3.2 Deep Sets 覆盖状态编码器

每个 bin 编码为 13 维元素：命中状态、7 类 coverage type、bin/coverpoint 位置、预算进度和总覆盖率。

```text
N x 13 variable-length bin set
  -> shared phi: Linear(13, 24) + ReLU
  -> mean-pool(all) || mean-pool(uncovered) || global state
  -> rho: Linear(51, 32) + ReLU
  -> Q(basic, boundary, cross, temporal)
```

同一组参数可以处理 86、92、120 bins 或其他长度。集合网络每 32 cycles 刷新一次，宏边界只从尚未执行的队列中选择最高 Q 值。

## 4. 真实轨迹与 fitted-Q

先用合成 coverage 状态训练集合编码器，再从真实 Verilator RTL 采集宏级转移：

```text
(coverage state, macro action, new bins, duration, next state)
```

对 SPI Master 和 SPI Xfer 各执行 8 种位置均衡的宏排列，共采集 64 条真实转移。训练采用半马尔可夫 Bellman 目标：宏耗时越长，未来收益折扣越大；新增 bin 奖励近似在宏执行中点到达。对当前状态未观测到的动作加入低权重零值伪样本，形成保守的 CQL-style 惩罚，避免小数据集上的 Q 值外推过高。

训练结果：

| 指标 | 结果 |
|---|---:|
| 真实 RTL 转移 | 64 |
| fitted-Q 迭代 | 30 |
| 每 1000 cycles 折扣 | 0.97 |
| 保守惩罚权重 | 0.04 |
| Bellman RMSE | 0.0520 |

两个 SPI DUT 的初始贪心动作都从 `basic` 变为实测单位时间收益更高的 `temporal`。

## 5. 最终真实 RTL 回归

每个 DUT 运行 30,000 cycles：

| DUT | 覆盖率 | bins | predict 均值 | predict P99 |
|---|---:|---:|---:|---:|
| dma_xfer_public | 100.000% | 92/92 | 0.0460 ms | 0.3547 ms |
| spi_master_public | 100.000% | 120/120 | 0.0188 ms | 0.3254 ms |
| spi_xfer_public | 100.000% | 86/86 | 0.0184 ms | 0.3068 ms |

移除学习式控制器、改用固定启发式宏顺序时，最终覆盖率仍为 100%，但 AUC 明显下降：

| DUT | 固定启发式 AUC | 离线 Q AUC | 相对提升 |
|---|---:|---:|---:|
| SPI Master | 22,857.34 | 24,715.67 | +8.13% |
| SPI Xfer | 25,853.65 | 26,993.19 | +4.41% |

最终宏顺序分别为：

- SPI Master：`temporal -> cross -> boundary -> basic`
- SPI Xfer：`temporal -> basic -> cross -> boundary`

SPI Xfer 在约 17k cycles、SPI Master 在约 25k cycles 达到 100%；DMA 的最终覆盖和 AUC 不受宏 Q 控制器影响。

## 6. 文件与复现

- 推理：`app/inference/coverage_controller.py`
- 合成预训练模型：`app/inference/model/coverage_controller.npz`
- 真实轨迹 Q 模型：`app/inference/model/coverage_q_controller.npz`
- 轨迹采集：`tools/collect_macro_trajectories.py`
- 离线 Q 训练：`tools/train_offline_q_controller.py`
- 真实轨迹：`results/macro_trajectories_verilator.json`
- Q 训练指标：`results/coverage_q_training.json`
- 最终 RTL 回归：`results/offline_q_final_verilator_30k.json`

```bash
python3 tools/collect_macro_trajectories.py --backend verilator --steps 30000 --orders 8
python3 tools/train_offline_q_controller.py
python3 tools/run_experiments.py --dut all --steps 30000 --backend verilator \
  --output results/offline_q_final_verilator_30k.json
```

## 7. 当前边界

离线 Q 已使用真实覆盖增量和耗时，而不是合成启发式标签；但数据仍只来自两个公开 SPI DUT，主要证明同类 coverage schema 和宏调度上的泛化。确定性事务生成器仍是安全下限。若继续提升最终覆盖，下一步应分别针对 DMA 的隐藏模式/仲裁探测吞吐和 SPI Master 的精确中断、abort-recovery 时序做黑盒序列搜索。
