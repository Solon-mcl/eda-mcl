# EDA Cup 覆盖率驱动验证

评测入口为 [`app/inference/__init__.py`](app/inference/__init__.py) 的 `InferenceInterface`。当前算法、测试命令和最近验证结果见 [算法说明](docs/CURRENT_ALGORITHM_OVERVIEW.md)；公开与验证 DUT 的运行方法见 [验证包说明](validation_duts/README.md)。历史提交证据保存在 [提交证据目录](docs/submission_evidence/README.md)。

`tools/run_experiments.py` 用于本地评估。`build.sh` 用于构建 Docker 镜像，并在 `deliverables/` 生成 `submission.tar`、校验和与 `submission.zip`。当前实验结果不会复制进镜像。

## 可选云端提示

默认使用本地策略，不访问网络。设置 `DEEPSEEK_ENABLED=1` 可在初始化时启用 DeepSeek 程序提示；设置 `EDA_LLM_ENRICH=1` 可启用语义富化。二者独立，均只在初始化阶段调用；逐周期 `predict()` 不联网。端点、密钥和超时由相应环境变量配置，调用失败时回退到本地策略。实际可用开关以代码为准。
