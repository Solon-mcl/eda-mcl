# EDA 覆盖率驱动激励生成 · 基础镜像使用说明

## 一、镜像是什么

`eda-coverage-base-1.0.tar.gz` 是离线可复现的开发环境，已内置三题公开包、预编译的 Verilator 真实 RTL 模型和完整工具链（Verilator 5.020 / Python 3.10 / numpy / g++ / make），开箱即用，无需联网安装。

## 二、加载与进入

```bash
docker load < eda-coverage-base-1.0.tar.gz
docker run -it --rm eda-coverage-base:1.0
```

## 三、目录结构

```
/workspace
├── dma_xfer_public/dma_xfer_public/      # 题一：可编程 DMA 传输控制器
├── spi_master_public/spi_master_public/  # 题二：全功能 SPI/SSP/Microwire 主机控制器
└── spi_xfer_public/spi_xfer_public/      # 题三：SPI/SSP 主机串行传输控制器
```

每题内部结构相同：

```
<题名>_public/
├── local_sim.py            # 周期精确 Python DUT（快速推理/调试）
├── coverage_simulator.py   # 覆盖率统计
├── harness.py              # 评测循环（内置公平随机基线示例）
├── inference_interface.py  # 推理接口
├── dut/                    # 规格书 dut_spec.md、RTL 源码、coverage_meta.json
└── verilator_harness/      # Verilator 真实 RTL 仿真（预编译模型在 obj_dir/）
```

## 四、在哪里开发

进入对应题目目录：

```bash
cd /workspace/spi_xfer_public/spi_xfer_public
```

在此目录下编写你的激励生成 agent：实现 `predict(coverage_state, step, max_steps)` 返回动作向量，再通过 `harness.run_episode(agent)` 跑覆盖率闭环。可先运行 `python3 harness.py` 观察自带的公平随机基线结果。

## 五、备注

- 默认使用真实 RTL 仿真（backend=verilator），走 `obj_dir/` 预编译模型；个别题目首次运行若参数不匹配会自动重编，等待数秒即可。

