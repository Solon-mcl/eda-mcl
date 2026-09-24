# 第一阶段优化：通用规格语义 IR

## 目标与边界

本阶段只实现 DUT 无关的规格解析，不加入任何验证集 DUT 名称、关键词组合或专用动作序列。目标是将 Markdown 规格转换为统一结构，使 generic 策略至少能够生成形状正确、范围合法、符合常见寄存器接口结构的动作。

## IR 内容

`app/inference/semantic_ir.py` 当前提取：

- action 字段顺序和维度；
- `d0..d3`、`pad[0:6]` 等压缩字段范围；
- 字段位宽、最小值、最大值和枚举描述；
- active-high / active-low reset；
- request、write/read enable、register address、data/address lane、ready、stall、recovery、advance、event、fault 等通用角色；
- Markdown 表格和 `Register N (NAME)` 形式的寄存器地址；
- 规格中显式出现的数值关系约束，供后续约束规划阶段使用。

generic 策略使用 IR 完成：

- LLM/本地生成动作的统一范围钳制；
- padding 强制清零；
- 已声明寄存器地址的系统扫描；
- 合法数据边界写入；
- 通用 advance/event/fault/read 控制的分离脉冲探索。

## 测试

- 新增合成寄存器接口契约测试，不引用任何验证 DUT；
- neural router、coverage controller、DeepSeek planner 离线测试通过；
- 三个公开 DUT 的 30k Verilator 回归仍为 100%；
- Cache 开发回归 75/75；
- Branch 开发回归 66/66。

## 冻结新 DUT 观察

在不增加 Watchdog 专用逻辑的情况下，三次 50k 结果均为 34/59（57.627%），平均 AUC 28,601.12。优化前平均覆盖率为 51.412%、平均 AUC 25,524.91，因此本阶段带来：

- 终值覆盖率提高 6.215 个百分点；
- AUC 提高 12.05%；
- 三个 seed 结果一致，消除了未知标量随机取值造成的波动。

剩余 25 个 bins 集中在合法窗口计时、ordered key pair、early/late service、timeout reset 和 external-reset recovery。结构化 IR 已能发现“有哪些字段和寄存器”，但尚不能把规格约束自动编译为多周期事务；这属于下一阶段的通用时序规划问题。

## 原始结果

- `results/semantic_ir_public_30k.json`
- `results/semantic_ir_cache_50k.json`
- `results/semantic_ir_branch_2k.json`
- `results/semantic_ir_watchdog_50k_3runs.json`
