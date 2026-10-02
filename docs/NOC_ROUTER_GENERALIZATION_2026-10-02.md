# NoC 新验证包：首次盲测后的通用机制改进

首次盲测记录见 `docs/NOC_ROUTER_BLIND_2026-10-02.md`。以下改动已阅读该 DUT 的规格和本地模型，后续结果不再称盲测。全部实验采用 local harness、seed 260923、30,000 cycles、1000 周期采样，非法动作均为 0；该包没有 Verilator 后端。

## 改动

1. `semantic_ir.py` 从规格中的字段枚举（包括句末最后一项）和公共端口枚举提取上界。原解析把 `in_port`、`flit_type` 都限制到 0..1；现分别解析为 0..4、0..3。若规格声明 mesh/grid 节点坐标，则对目的地 X/Y 允许一跳邻域探测；它是候选探索范围，不是假定全网格边界。
2. `__init__.py` 新增能力门控的 `credit_packet` 程序族。只有规格同时给出多端口输入、head/body/tail/single 枚举、信用掩码和 mesh 坐标时才生成。程序覆盖正信用端口/目的地/VC 扫描、完整 flit 阶段、无信用缓冲及释放、拥塞切换、同步多输入争用、尾拍信用恢复和单独停顿。填充长度来自规格公开的 FIFO 深度范围；没有 DUT 名称分流。`EDA_CREDIT_PACKET_PROGRAMS=0` 可关闭，默认开启。
3. 旧反事实回放工具读取源轨迹中的新开关值；旧轨迹无该字段时按关闭处理，避免以后默认值改变使历史自适应回放发散。

## 同预算消融

| 策略 | 默认参数覆盖 / AUC | 隐藏参数 A 覆盖 / AUC | 隐藏参数 B 覆盖 / AUC |
|---|---:|---:|---:|
| 首次盲测默认策略 | 32/75 / 0.4196 | 31/75 / 0.4056 | 32/75 / 0.4187 |
| 仅显式端口和 flit 枚举范围 | 37/75 / 0.4851 | 36/75 / 0.4720 | 37/75 / 0.4851 |
| 加入首版信用/包程序 | 64/75 / 0.8347 | 63/75 / 0.8216 | 63/75 / 0.8211 |
| 最终默认方案 | **75/75 / 0.9762** | **74/75 / 0.9631** | **75/75 / 0.9762** |
| 包内 greedy 参考 | 75/75 / 0.9833 | 72/75 / 0.9440 | 未运行 |

最终默认运行中，七个 `credit_packet` 程序首次执行分别直接新增 40、7、5、7、5、2、1 个 bin；后续重复未带来新增。完整记录见 `results/noc_default_final_20261002_30k.json` 和 `results/noc_credit_program_v3_{default,a,b}_20261002_30k.json`。

隐藏参数 A 唯一缺失 `queue_level.almost_full`。该配置 `buffer_depth=2`；本地模型对水位 1 返回 single 类，对水位 ≥2 返回 full 类，故中间的 almost-full 类不可达。包内 greedy 同样缺此 bin，且另缺两个序列 bin。没有为这一不可达目标加入额外重试。

对已有 9 个 DUT 开启最终程序族，30k 本地回归合计仍为 **701/735**，各 DUT 覆盖和 AUC 均与前一轮默认策略一致；结果见 `results/noc_generalization_public_20261002_30k.json` 和 `results/noc_generalization_*_validation_20261002_30k.json`。加上 NoC 默认参数，10 个包合计 **776/810**。`test_credit_packet_capability.py`、`test_inference_contract.py`、`test_generic_temporal_planner.py`、`check_parsing_generalization.py` 均通过；最终默认开关无环境变量复核仍为 75/75、AUC 0.9762。

该新 DUT 已用于开发，因此最终结果证明的是针对公开规格结构的机制迁移和已知包零回归，不能作为另一个独立未知 DUT 的盲测证据。当前未打包。
