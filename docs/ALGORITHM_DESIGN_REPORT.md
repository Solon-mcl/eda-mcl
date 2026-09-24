# 覆盖率驱动通用测试激励生成算法设计报告

## 摘要

本方案面向赛题的统一推理接口：算法只接收 DUT 规格文件、覆盖组定义、当前累计覆盖向量、当前步数和总预算，逐周期输出一维动作向量。核心目标不是记忆某个公开 DUT 的测试序列，而是在以下三类不确定性下持续获得覆盖增量：

1. DUT 协议、动作维度和状态结构未知；
2. 评测 DUT 与训练/公开 DUT 分布不同；
3. 正确模式、索引扰动、窗口长度、替换策略等参数被隐藏。

最终算法采用“规格解析 + OOD 路由 + 协议专家 + 通用语义生成 + 覆盖反馈控制”的分层结构。对已知协议族使用高效率专家策略；对未知 DUT 拒绝闭集神经分类器的强制三选一，根据规格自动解析动作字段，并选择通用的事务语义生成器。所有在线适应均通过状态变量、队列和覆盖反馈完成，推理期间不修改模型权重。

在当前仓库的三个公开 DUT、两个开发验证 DUT 及多组隐藏参数上，最终覆盖结果如下。两个 validation DUT 曾用于失败分析和策略迭代，因此其结果用于回归与鲁棒性验证，不作为严格未见分布的泛化证明：

| DUT/测试 | 后端与预算 | 最终覆盖率 |
|---|---:|---:|
| dma_xfer_public | Verilator，30k | 92/92，100% |
| spi_master_public | Verilator，30k | 120/120，100% |
| spi_xfer_public | Verilator，30k | 86/86，100% |
| cache_ctrl_validation 默认参数 | local cycle model，50k | 75/75，100% |
| cache 隐藏参数 8 组 | 每组 10k | 平均 99.00%，最差 98.67% |
| branch_predictor_validation 隐藏参数 8 组 | 每组 2k | 全部 66/66，100% |

DeepSeek 在上述最终实验中保持关闭，不参与逐周期推理。

---

## 1. 赛题约束与设计目标

### 1.1 接口约束

算法实现标准 `InferenceInterface`：

```python
InferenceInterface(dut_spec_path, covergroup_path)
predict(coverage_state, step, max_steps) -> np.ndarray
```

其中 `coverage_state` 是按 `coverage_meta.json` 顺序展开的累计二值 bin 向量。算法无法直接读取隐藏参数，也不能在推理时训练或修改模型参数。

### 1.2 优化目标

评分同时关注最终覆盖率和覆盖曲线 AUC。因此算法需要兼顾：

- 早期快速命中 basic、boundary 和普通 cross；
- 中期完成状态积累、冲突、背压和恢复序列；
- 后期探测隐藏正确性条件和 hardest bins；
- 在未知 DUT 上至少稳定超过随机基线，而不是输出错误协议的动作。

### 1.3 设计原则

- **合法性优先**：先生成可被 DUT 接受的事务，再优化参数。
- **闭集模型不得强制路由未知输入**：高 softmax 置信度不等于分布内。
- **隐藏参数通过行为和覆盖反馈探测**：不假设公开默认值等于评测值。
- **状态型 DUT 使用结构化序列**：随机独立动作难以覆盖缓存、预测器、FIFO 等长期状态。
- **推理状态可变，模型权重不可变**：符合赛题在线推理约束。

---

## 2. 总体架构

算法由五层组成：

```text
DUT spec + covergroup
        │
        ▼
动作模式解析器 ──────► 动作字段、维度、常见语义
        │
        ▼
结构签名 + 神经文本路由 + OOD 拒识
        │
        ├── DMA 专家策略
        ├── SPI Master 专家策略
        ├── SPI Xfer 专家策略
        └── 未知 DUT 通用语义策略
                    │
coverage_state ─────┤
                    ▼
         覆盖反馈控制与动作队列
                    │
                    ▼
              action vector
```

### 2.1 动作模式解析

解析器优先从规格中的声明提取动作字段：

```text
action = [field0, field1, ..., fieldN]
```

它自动获得：

- 动作维度；
- 字段顺序；
- active-low reset；
- valid/start、ready、opcode/kind；
- 地址、数据、PC 和 target 的 little-endian byte lanes；
- stall、flush 和保留字段。

如果规格没有显式列表，再从 “action space (N dims)” 或中文维度描述中推断维数。

### 2.2 神经文本路由器

神经路由器是一个离线训练的一层隐藏层 MLP：

1. 对 spec 和 covergroup 文本做 token 化；
2. 使用 signed hashing 构造固定 256 维 unigram/bigram 特征；
3. MLP 输出 DMA、SPI Master、SPI Xfer 三类概率。

它的作用是识别与公开协议相近的输入，不负责理解任意新 DUT。

### 2.3 OOD 拒识

早期版本直接采用最高 softmax 类别，导致 cache 和 branch predictor 都被高置信误判成 DMA。最终版本采用以下路由规则：

1. 规格存在 DMA/SPI 强结构签名时，进入对应专家；
2. 否则，神经分类结果必须同时满足：
   - 置信度不低于 0.995；
   - 动作维度与该协议族一致；
3. 不满足条件时进入 generic，而不是强制三选一。

这一步是跨 DUT 泛化的关键。它把“分类错误后完全无有效事务”的灾难性失败，转化为“使用通用策略逐步探索”。

### 2.4 Deep Sets 覆盖控制器

覆盖状态不是固定长度：不同 DUT 的 bin 数和顺序不同。方案使用 permutation-invariant 的集合编码思想，每个 bin 构造成包含以下信息的描述向量：

- 当前是否命中；
- coverage type；
- coverpoint/bin 的相对位置；
- 当前预算进度；
- 总覆盖率。

共享 MLP 编码各 bin 后做集合聚合，输出宏策略分数。离线 fitted-Q 模型用于 SPI 宏任务排序。该模型只改变不同测试宏的执行顺序，不直接生成非法底层总线动作，因此即使模型泛化不佳，确定性协议策略仍提供安全下限。

---

## 3. 通用未知 DUT 策略

### 3.1 基础覆盖引导生成

对于没有专用专家的 DUT，generic 策略执行以下步骤：

1. 解析动作字段及有效维数；
2. 若存在 active-low reset，先复位两个周期，再持续保持 deasserted；
3. pad/reserved 固定为零；
4. ready 默认拉高，并周期性制造有限背压；
5. 生成零、一、全一、`0xAA`、`0x55`、高低边界等参数；
6. 保存当前累计覆盖数及最近覆盖增长时刻；
7. 采用运行长度编码队列保持协议动作和必要等待周期。

与纯随机相比，最重要的差异是不会随机反复复位 DUT，并且会重复事务以形成持久状态。

### 3.2 事务语义适配

generic 不是单一随机公式，而是根据规格关键词与字段结构选择语义程序。

#### 存储/缓存类

当规格同时给出 read、write、invalidate、flush 和地址字段时，生成：

- 冷 miss 后同地址重放；
- write 后 read；
- 同 set 不同 tag 的冲突访问；
- clean/dirty eviction；
- invalidate hit、invalidate miss 和重新 refill；
- ready 拉低后的恢复；
- 长状态积累后低频 flush；
- 单调 tag 扫描，用于隐藏异常 tag。

flush 不能过早执行，否则 valid/dirty half/full 永远无法积累。最终策略每 128 个事务才执行一次 flush，并为隐藏最大延迟保留保守等待窗口。

#### 分支预测类

当规格包含 branch predictor、PC、target、kind 和 actual_taken 时，生成：

- 连续 taken 和连续 not-taken，推动 PHT 饱和；
- 交替结果，制造 mixed history 和 PHT alias；
- 相同 PC/target 重放，将 BTB miss 转换为 hit；
- 多个同类 PC，触发 BTB replacement；
- call 后 return，命中 RAS top；
- 超过规格最大深度的连续 call，触发 RAS overflow；
- cold return，覆盖 empty-RAS/BTB-miss；
- stall 后重放；
- flush 后恢复。

这些序列不依赖固定索引 salt 或 history 位数，而是使用足够长的训练窗口和系统性 PC stride。

---

## 4. 已知协议专家策略

### 4.1 DMA

DMA 专家包含边界配置、隐藏 mode 探测、通道回放、冲突仲裁和时序覆盖。

关键机制：

- RTL 实际保存 4 位 mode，因此穷举 0..15；
- 每个候选 mode 通过短事务触发完成路径；
- 从 `xfer_done_ok` 新增 bin 反推出 mode→class 映射；
- 对可能最长 512 周期的 hidden hold 使用 520 周期安全窗口；
- 发现正确 mode 后立即跳过剩余等待，提高 AUC；
- 在目标通道保持 ACTIVE 时触发正确 class，使 active channel、方向、仲裁 winner 和 done class 正确交叉；
- 单通道事务消除隐藏仲裁优先级对 winner 覆盖的影响。

公开 RTL 中完成事件具有 sticky 行为。策略通过在 hold 释放窗口重复写单一配置字段，确保配置最终被接受，并在完成后恢复到非匹配 mode，防止无限重复 hold。

### 4.2 SPI Master / SPI Xfer

SPI 专家把测试分为四类宏：

- 基础协议与高收益传输；
- abort/recovery/refill 时序；
- protocol × mode 交叉；
- 数值、FIFO、DFS/CFS/NDF 边界。

主要序列包括：

- SPI0、SPI1、SSP、Microwire；
- 全部 TMOD；
- DFS、CFS、BAUD 和 NDF 边界；
- TX/RX FIFO empty/full/threshold；
- 多个 slave-select；
- 2/4/8 周期的早期片选中止及恢复；
- 保持 TX 非空时读取空 RX，隔离纯 RX-underflow；
- multi-frame、last-frame、hold 和 refill 时序。

---

## 5. 隐藏参数处理方法

方案没有统一地“猜一个隐藏值”，而是按隐藏参数的性质采用四种方法。

### 5.1 小离散空间：完全枚举

适用于 DMA 4 位 mode：直接枚举所有 16 个候选值，使用覆盖增量识别正确类。

优点是无先验偏差，隐藏值变化不会失效。

### 5.2 有界时延：保守窗口与恢复握手

适用于 DMA hold、cache memory latency、SPI hold：

- 从规格获得最大范围；
- 等待时间覆盖最大值并留少量裕量；
- 用 ready/stall 的低→高转换显式覆盖恢复；
- 对窄释放窗口重复幂等写入，避免错过唯一可接受周期。

### 5.3 隐藏索引/替换：构造不变量序列

适用于 cache replacement、branch index salt/history：

- 单通道或单请求避免仲裁优先级影响；
- 多个同 set/tag stride，使两种 victim 方向都产生 eviction；
- 相同 PC/target 重放保证无论 salt 如何，第二次都访问相同表项；
- 训练长度超过最大 history bits，使历史最终稳定到全零或全一；
- call 数超过最大 RAS depth，保证所有允许深度都会 overflow。

这类方法不需要恢复具体隐藏参数，只需要生成对参数变化不敏感的覆盖序列。

### 5.4 大空间稀有值：预算内系统扫描

适用于 26 位 poison tag。完整穷举不可行，因此采用单调、确定性的 tag 扫描：

- 早期优先低值；
- 保证同一预算和种子下结果可复现；
- 一旦命中覆盖 bin，不需要知道隐藏值本身；
- 与正常冲突事务交织，扫描动作本身仍能贡献其他覆盖。

限制是任意高位 26 位隐藏值无法在 10k 周期内保证命中。实验中高位 poison 组合的唯一缺口就是该 bin；默认低位参数在约 16k 周期命中并最终达到 100%。

---

## 6. 覆盖反馈闭环

### 6.1 累计反馈

算法维护上一步覆盖状态：

```text
new_bins = current_coverage AND NOT previous_coverage
```

新增 bin 用于：

- DMA mode→done class 映射；
- 提前结束已经成功的保守等待；
- 更新最近有效探索时间；
- 调整宏任务顺序。

### 6.2 为什么不在线训练

赛题禁止在 `predict()` 中修改模型权重。方案的在线适应只修改：

- 当前事务队列；
- 已发现 mode/class 字典；
- 已执行宏集合；
- 最近覆盖增量和阶段计数器。

因此属于规划状态更新，而不是模型训练。

### 6.3 AUC 优化

专家策略把高收益 basic/boundary 放在前部，把长时序和隐藏搜索放在后部。通用策略则先复位、覆盖常规事务和边界，再做状态积累、替换、flush 和大空间扫描。这样既提高最终覆盖，也避免大量预算消耗在早期低概率事件上。

---

## 7. 实验设计与结果

### 7.1 公共 RTL

三个公开 DUT 使用真实 Verilator backend，30,000 steps：

| DUT | 覆盖 bins | 收敛情况 |
|---|---:|---:|
| DMA | 92/92 | 约 10k 达到 100% |
| SPI Master | 120/120 | 约 25k 达到 100% |
| SPI Xfer | 86/86 | 约 17k 达到 100% |

### 7.2 Cache 开发验证

优化前闭集路由器把 cache 高置信误判为 DMA，50k steps 只有 11/75。加入 OOD 拒识和存储事务语义后：

- 默认参数 50k：75/75；
- 8 组隐藏参数 10k：平均 99.00%，最差 98.67%；
- 随机基线 10k：平均 84.00%；
- 候选平均 AUC 0.9069，随机为 0.8124。

### 7.3 Branch Predictor 开发验证

初始 generic 没有识别 PC/target，50k 为 53/66，低于随机的 55/66。扩展规格语义后：

- 8 组隐藏参数、每组 2k：全部 66/66；
- 候选平均 AUC：0.9598；
- 随机平均覆盖率：78.60%；
- 随机平均 AUC：0.7347。

隐藏组合覆盖 history bits 4..6、salt 0..15、两种 replacement 和 RAS depth 4..8。

### 7.4 回归结论

加入两类 generic 语义适配后：

- Cache 仍为 75/75；
- DMA、SPI Master、SPI Xfer 均保持 100%；
- 未出现专家策略被 OOD 机制错误切换的问题。

### 7.5 评测完整性说明

- 两个 validation DUT 的 greedy/oracle 策略没有被导入候选算法；它们只用于仓库自身的可达性说明。
- 两个 validation DUT 的规格和失败结果参与过 generic 语义程序开发，故不将其表述为冻结 held-out 测试。
- 隐藏参数实验通过 harness 的 secrets 接口注入，候选算法看不到参数值。
- SPI Master 原 Verilator 文本适配器遗漏了 RTL 已存在的 `cov_s0` 输出，导致三个 `seq_a` bin 在模拟器侧不可见；实验前仅补齐该信号输出列，没有修改 DUT 行为或覆盖定义。
- 所有报告结果都保存了原始 JSON，避免只报告单一百分比。

---

## 8. 性能与工程实现

### 8.1 推理开销

动作生成主要是 NumPy 小向量运算、正则解析后的固定索引写入和队列弹出。公开回归中平均 `predict()` 时间约 0.02–0.04 ms，P99 约 0.25–0.30 ms，远低于赛题建议的 30 ms。

### 8.2 可复现性

- 固定随机种子；
- 系统枚举和边界序列确定性执行；
- 每次实验输出完整 curve、missing bins、耗时和模型路由信息；
- DeepSeek 默认关闭，避免网络状态影响结果。

### 8.3 主要代码

- `app/inference/__init__.py`：路由、专家和 generic 策略；
- `app/inference/neural_router.py`：文本 MLP；
- `app/inference/coverage_controller.py`：集合覆盖编码和宏控制；
- `tools/run_experiments.py`：统一实验入口；
- `tools/evaluate_hidden_parameters.py`：cache 隐藏参数评测；
- `tools/evaluate_branch_hidden_parameters.py`：branch 隐藏参数评测。

---

## 9. 局限性与后续工作

### 9.1 当前局限

1. generic 语义仍依赖规格使用可识别的字段名和操作描述；极度模糊的 spec 会退化到边界/随机探索。
2. 任意大位宽稀有 magic value 无法在有限周期内保证命中。
3. 当前神经路由器只训练了三个公开协议族，OOD 判断主要依赖结构签名、维度和高阈值，而非专门训练的能量模型或 one-class 模型。
4. 两个开发验证 DUT 只提供 local cycle model，且参与过策略迭代，无法替代冻结的外部泛化集或独立 Verilator/VCS 交叉验证。
5. 专家策略的最终覆盖很高，但部分 DUT 的早期 AUC 仍可继续优化。

### 9.2 后续方向

- 为 OOD 路由增加距离、energy score 或深度集成，而不只依赖 softmax 阈值；
- 将 spec 解析升级为字段类型、取值范围和操作语义的结构化 IR；
- 从 coverage metadata 自动合成 sequence/cross 的逆向目标；
- 建立通用事务语法库，如 FIFO、仲裁器、缓存、预测器、协议控制器；
- 对大空间隐藏值使用覆盖引导的分层搜索、位级扰动和约束求解；
- 在更多未参与开发的 DUT 上冻结评测，防止 validation 驱动的过拟合。

---

## 10. 结论

本方案的泛化不是由单一神经网络端到端产生，而是由多种机制共同实现：

- 神经模型负责已知协议族的快速识别和宏排序；
- OOD 拒识防止未知 DUT 被错误专家接管；
- 规格解析把不同动作空间转换为统一字段语义；
- 通用事务生成器构造状态型、边界型和恢复型序列；
- 覆盖反馈用于在线映射隐藏条件和推进探索阶段；
- 完全枚举、有界等待、不变量序列和系统扫描分别处理不同类型的隐藏参数。

实验表明，该结构既保持了公开 DUT 的满覆盖，也修复了两个分布外验证 DUT 上低于随机基线的问题。更重要的是，它明确暴露并处理了闭集分类器在未知 DUT 上的失败模式，使系统从“公开 DUT 专用策略集合”演进为“专家策略与通用语义策略结合的覆盖驱动生成框架”。
