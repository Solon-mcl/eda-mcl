# DeepSeek-V4 使用报告

## 模型与调用方式

- 模型：赛题指定的 DeepSeek-V4，API 模型标识默认 `deepseek-v4-pro`。
- 方式：云端 API，OpenAI-compatible `POST /chat/completions`。
- 默认官方端点：`https://api.deepseek.com`；评测时可通过环境变量切换到组委会提供的端点。
- 输出模式：JSON Object，关闭 thinking，最大输出 4096 tokens。
- 调用频率：每个 DUT 初始化时最多一次，`predict()` 内不调用 API。

## 在算法中的角色

DeepSeek 读取评测时传入的 `dut_spec.md` 和 `covergroup.svh`，输出经过约束的结构化计划：

```json
{
  "family_hint": "dma|spi_master|spi_xfer|generic",
  "action_dim": 12,
  "macro_order": [3, 0, 2, 1],
  "program": [
    {"action": [0, 0, 0], "cycles": 2, "purpose": "reset"}
  ],
  "summary": "strategy"
}
```

对已知 DMA/SPI 语义族，LLM 只辅助纠正闭集文本路由，不替换已通过 RTL 回归的事务策略和离线 Q 排序。对未知 DUT，LLM 给出动作维度与合法事务序列，替代原本语义较弱的低差异随机初始探索；LLM 程序耗尽后仍回到确定性探索。

## Token 与时间预算

- 输入最多截取 60,000 字符 spec 和 20,000 字符 covergroup。
- 输出上限 4096 tokens，显著低于赛题每 DUT 500,000 tokens 上限。
- 实测小规格调用 9.41 秒，公开 SPI Xfer 完整规格调用 28.39 秒；默认硬超时据此设为 45 秒，只在 `__init__()` 调用。
- 返回的 `prompt_tokens`、`completion_tokens`、`total_tokens` 和调用耗时保存在 `llm_status`，实验脚本会写入结果 JSON。

真实 API 已完成两次最终验证：小型 ALU 规格使用 369 输入 Token、824 输出 Token，共 1,193 Token；公开 SPI Xfer 使用 7,225 输入 Token、3,073 输出 Token，共 10,298 Token。两次均为 `finish_reason=stop`，结构化计划通过全部本地校验。此前还验证了 60 秒超时和 JSON 截断两条降级路径，均能继续使用本地策略。

## 安全校验与降级

以下情况均不会阻断仿真：

- API Key 缺失或通过 `DEEPSEEK_ENABLED=0` 禁用；
- DNS、连接、超时、HTTP 错误或 Token 超限；
- 响应不是 JSON、字段缺失或动作维度不匹配；
- 动作包含非有限数值、绝对值超过 32-bit 范围、非法 cycles。

程序限制为最多 32 条动作，每条持续 1–4096 cycles，动作维度必须严格一致。API Key、完整 prompt 和模型原始回复不会写入日志。

## 配置变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 无 | 缺失即关闭调用 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 可改为组委会端点 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | 模型标识 |
| `DEEPSEEK_TIMEOUT_S` | `45` | 单次请求硬超时 |
| `DEEPSEEK_MAX_TOKENS` | `4096` | 输出上限，代码限制为 256–8192 |
| `DEEPSEEK_ENABLED` | `0` | 仅设为 `1/true/yes` 时启用 |

兼容别名：`DEEPSEEK_API_TOKEN`、`LLM_API_KEY`、`LLM_API_TOKEN`、`OPENAI_API_KEY`、`LLM_BASE_URL`、`OPENAI_BASE_URL`、`LLM_MODEL`。

## 测试

```bash
python3 tools/test_deepseek_planner.py
```

该测试使用内存中的模拟 API 响应，不访问外网，覆盖请求格式、模型名、JSON 校验、程序执行和无凭据降级。
