# 当前算法与复现

更新日期：2026-10-02。本文描述工作区当前默认推理路径；历史实验与旧版本分别见 [算法版本](ALGORITHM_VARIANTS.md)和 [9 月 30 日复核](ALGORITHM_AUDIT_2026-09-30.md)。

## 推理路径

评测器导入 `app/inference/__init__.py` 中的 `InferenceInterface`。初始化读取 DUT 规格与覆盖定义，`semantic_ir.py` 解析动作字段、角色、范围及接口能力；`coverage_targets.py` 和 `coverage_dependency.py` 建立覆盖目标与依赖。所有 DUT 使用同一 `UniversalPolicy`，没有按 DUT 名称选择策略。

`predict()` 在程序边界由 `generic_planner.py` 调度候选，再逐周期输出动作。`joint_candidates.py` 编译联合字段写入；`generic_sequence_search.py` 用实际覆盖增量与周期成本更新候选优先级。`target_tasks.py` 对具备多数据端口别名的接口调度可执行覆盖目标；规格含多端口、flit 阶段、信用掩码和网格坐标时，还会生成信用与包传输程序。所有动作经过字段范围和合法性约束，输出 `float32` 向量。在线只更新调度统计，不更新模型权重。

`DeepSeekPlanner` 和 `LLMEnricher` 都是可选的初始化阶段提示源。默认关闭；接口不可用时使用本地策略。逐周期 `predict()` 不调用网络。配置和实验开关以代码及 `tools/run_experiments.py --help` 为准。

## 运行和测试

```bash
python3 tools/test_inference_contract.py
python3 tools/test_generic_temporal_planner.py
python3 tools/test_credit_packet_capability.py
python3 tools/check_parsing_generalization.py
python3 tools/run_experiments.py --dut noc_router_validation --backend local --agent full --steps 30000 --output results/noc_recheck.json
```

`--dut all` 只运行三个公开 DUT；验证 DUT 需逐个指定。实验结果写入 `results/`，不会进入提交镜像。`build.sh` 会构建并打包镜像。

## 最近验证结果

- NoC 验证包默认参数：75/75，AUC 0.9762；隐藏参数 A：74/75，B：75/75。详见 [首次盲测](NOC_ROUTER_BLIND_2026-10-02.md)与 [机制改进](NOC_ROUTER_GENERALIZATION_2026-10-02.md)。
- 原有 9 个 DUT 的 30k 本地回归：701/735；目标调度对照见 [任务调度实验](TARGET_TASK_SCHEDULER_2026-10-02.md)。
- 这些验证包已参与开发，结果只能说明已测接口的表现；不能推断任意未知 DUT 的覆盖率。

正式提交证据和历史 RTL 实验保存在 [提交证据](submission_evidence/README.md)。
