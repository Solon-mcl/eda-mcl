# 泛化与隐藏参数评测

## 方法

最终推理器采用两层策略：

1. 规格中存在强结构签名时，使用 DMA/SPI 专家策略。
2. 对未识别 DUT，拒绝闭集神经路由器的普通置信预测，解析规格中的
   `action = [...]` 声明并启用通用覆盖引导事务生成器。

通用策略自动识别 reset、valid/start、ready、opcode、地址和数据字节，
生成边界值、重复地址、同组冲突、背压恢复、低频 flush 和 tag 扫描。
推理期间不修改模型权重。以下实验均关闭 DeepSeek。

## Held-out validation DUT

`cache_ctrl_validation` 仅提供 local cycle model，因此按仓库说明使用 local
后端。没有将验证集提供的 greedy/oracle 策略导入候选算法。

| 配置 | 步数 | 覆盖率/AUC |
|---|---:|---:|
| 优化前候选算法，默认参数 | 50,000 | 11/75，14.67% |
| 优化后候选算法，默认参数 | 50,000 | 75/75，100% |
| 8 组隐藏参数，优化后均值 | 10,000 | 99.00%，AUC 0.9069 |
| 8 组隐藏参数，优化后最差 | 10,000 | 98.67% |
| 8 组隐藏参数，随机均值 | 10,000 | 84.00%，AUC 0.8124 |

隐藏参数组合覆盖 `replacement_xor=0/1`、`memory_latency=1..8`，以及多个
低位、中位和 26 位高值 `poison_tag`。1 万步未满分组合仅缺
`result.poison`；对低位 poison 参数可达到 100%。默认 poison 参数在约
16,000 步被系统 tag 扫描命中，最终达到 100%。

## Public DUT regression

### Branch predictor validation

新增的 `branch_predictor_validation` 改变为 PC/target 分支事务和长期预测器
状态。通用策略从 action 字段识别 PC、target、kind、taken、stall 与 flush，
组合 PHT 饱和训练、BTB 重放/替换、call-return、RAS overflow 和恢复序列。

| 配置 | 步数 | 覆盖率/AUC |
|---|---:|---:|
| 优化前候选算法，默认参数 | 50,000 | 53/66，80.30% |
| 随机基线，默认参数 | 50,000 | 55/66，83.33% |
| 8 组隐藏参数，优化后均值 | 2,000 | 66/66，100%，AUC 0.9598 |
| 8 组隐藏参数，随机均值 | 2,000 | 78.60%，AUC 0.7347 |

隐藏参数组合覆盖 `history_bits=4..6`、`index_salt=0..15`、两种
`replacement_xor` 和 `ras_depth=4..8`。8 组最终覆盖率均为 100%。

### Public regression

最终版本在真实 Verilator 后端、30,000 steps 下无退化：

| DUT | 覆盖率 |
|---|---:|
| dma_xfer_public | 92/92，100% |
| spi_master_public | 120/120，100% |
| spi_xfer_public | 86/86，100% |

## 结果文件

- `results/cache_hidden_parameter_sweep_final.json`
- `results/cache_validation_full_50k_after_branch.json`
- `results/branch_hidden_parameter_sweep_v1.json`
- `results/no_llm_verilator_30k_after_branch.json`
