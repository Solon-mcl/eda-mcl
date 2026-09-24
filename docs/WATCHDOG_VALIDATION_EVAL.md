# Watchdog 新验证 DUT 冻结评测

## 评测设置

- DUT：`watchdog_safety_validation`
- 后端：验证集提供的 local cycle model（该 DUT 无 Verilator backend）
- 算法：当前冻结的 `InferenceInterface`，未针对本 DUT 修改
- 预算：50,000 cycles，1,000 cycles 采样一次
- 重复：seed 260923、260924、260925
- 对照：验证集随机基线，同预算、同 seeds

## 结果

| 方法 | seed | bins | 终值覆盖率 | AUC |
|---|---:|---:|---:|---:|
| 当前算法 | 260923 | 32/59 | 54.237% | 26,923.19 |
| 当前算法 | 260924 | 28/59 | 47.458% | 23,567.32 |
| 当前算法 | 260925 | 31/59 | 52.542% | 26,084.22 |
| 随机 | 260923 | 27/59 | 45.763% | 22,728.36 |
| 随机 | 260924 | 25/59 | 42.373% | 21,050.42 |
| 随机 | 260925 | 25/59 | 42.373% | 21,050.42 |

汇总：

- 当前算法：覆盖率 `51.412% ± 2.881%`，平均 AUC `25,524.91`；
- 随机基线：覆盖率 `43.503% ± 1.598%`，平均 AUC `21,609.73`；
- 当前算法高出随机基线 7.91 个覆盖率百分点，AUC 高 18.12%。

## 冻结评测观察

三个 seed 均在约 1,000 cycles 后停止获得新 bin；增加到 50,000 cycles 没有继续改善。神经分类器的原始最高类别为 DMA、置信度 0.99216，但未达到 0.995 且动作维度不匹配，因此 OOD 门控正确拒绝专家路由，实际执行 generic 策略。

稳定缺失集中在：

- `OPEN_WINDOW / PRETIMEOUT / RESET_PENDING` 状态；
- early/late/accepted service 结果；
- window、near-timeout 计数边界；
- key A→B 的有序认证过程；
- timeout reset 和 external reset recovery 序列。

原因是 generic 字段解析能识别 16 维动作及低有效复位，但没有从规格建立 watchdog 事务语义：`reg_addr` 被当作未知标量，只探索 0/1，无法稳定配置地址 2 的 TIMEOUT；`tick`、`service` 和 `fault_inject` 也只作为独立布尔量探索，无法形成“配置窗口→使能→计时→双 key→超时/恢复”的有序程序。

## 原始记录

- `results/watchdog_safety_validation_full_50k_3runs.json`
- `results/watchdog_safety_validation_random_50k_3runs.json`

仓库中 watchdog 自带的 `greedy` 策略包含该 DUT 的定向寄存器和 key 序列知识，因此不作为通用算法的公平基线。
