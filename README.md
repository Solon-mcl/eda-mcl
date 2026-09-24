# EDA Cup submission workspace

最终上传文件为 [`deliverables/submission.zip`](deliverables/submission.zip)，压缩包内只包含 `submission.tar`，与组委会提交示例一致。SHA256 记录位于 [`deliverables/submission.tar.sha256`](deliverables/submission.tar.sha256)。

推理入口位于 `app/inference/__init__.py`，实验脚本位于 `tools/`，设计报告、覆盖率报告和实验记录统一保存在 [`docs`](docs/)；提交证据归档位于 [`docs/submission_evidence`](docs/submission_evidence/README.md)。运行 `./build.sh` 可重新构建、测试、导出并压缩提交镜像。

## 可选 DeepSeek-V4 配置

推理模块支持赛题指定的 DeepSeek-V4 云端 API。调用只发生在每个 DUT 的 `InferenceInterface.__init__()`，逐周期 `predict()` 不访问网络。未提供密钥、请求超时、Token 被拒或响应不合法时，会自动使用本地 MLP、Deep Sets 和确定性策略。

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"   # 组委会端点可覆盖
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_ENABLED="1"
export DEEPSEEK_TIMEOUT_S="45"
export DEEPSEEK_MAX_TOKENS="4096"
```

也兼容 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 以及 OpenAI 风格的环境变量。默认不调用；仅当 `DEEPSEEK_ENABLED=1` 时启用。详细说明见 [`docs/LLM_USAGE.md`](docs/LLM_USAGE.md)。
