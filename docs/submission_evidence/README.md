# EDA 覆盖率驱动激励生成系统：提交说明

本目录归档设计报告、覆盖率报告、消融结果和原始实验记录。作品不包含学校或指导教师信息。平台实际上传文件位于工作区 `deliverables/submission.zip`，ZIP 根目录只含 `submission.tar`。

## 提交文件

- `DESIGN.md`：算法架构、状态/动作空间和策略说明
- `COVERAGE_REPORT.md`：公开 DUT 三次真实 RTL 重复实验及覆盖率收敛结果
- `ABLATION_REPORT.md`：四种配置的消融对比
- `LLM_REPORT.md`：LLM 使用声明
- `coverage_curves.svg` / `coverage_curves.csv`：曲线图和原始聚合数据
- `experiment_summary.json`：机器可读实验摘要
- `experiments/`：三次正式运行与三组对照/消融的逐采样点原始 JSON

## 镜像加载与接口验证

```bash
docker load -i ../../deliverables/submission.tar
docker run --rm eda-cup-coverage-agent:v1
```

预期输出：

```text
inference image ready
```

评测端按标准方式导入：

```python
import sys
sys.path.insert(0, "/app")
from inference import InferenceInterface

agent = InferenceInterface("/dut/dut_spec.md", "/dut/covergroup.svh")
action = agent.predict(coverage_state, step, max_steps)
```

返回值始终为一维 `numpy.ndarray`，`dtype=float32`；维度由 DUT 规格自动识别。镜像基于组委会 `eda-coverage-base:1.0` 构建，无新增依赖、无需网络或 GPU。

## 复现实验

源码工作区中执行：

```bash
python3 tools/run_experiments.py --dut all --steps 30000 \
  --backend verilator --agent full --output results/reproduce.json
```

正式数据使用基础镜像内的预编译/按参数重编译 Verilator RTL 后端，而非仅使用 Python 快速模型。
