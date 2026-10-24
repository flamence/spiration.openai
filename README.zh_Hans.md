# Spiration OpenAI API 格式支持

[简体中文](#) · [English](README.md)

为 Spiration 内置智能体提供 OpenAI Chat Completions 格式的 provider。本拓展
以 Python 编写，向宿主注册名为 `openai` 的 provider，因此任何兼容 OpenAI
Chat Completions 的端点都可以作为对话后端。

## 快速开始

1. 确认已安装并启用 Python 拓展加载器（`com.flamence.spiration.python`）。
2. 把本拓展安装到 Spiration 的数据目录，并保持目录名与拓展 ID 一致：

   ```
   <数据目录>/com.flamence.spiration.openai/
       extension.json
       dist/extension.py
       dist/spiration_openai/
   ```

   Windows 下数据目录通常为 `%APPDATA%\spiration`。
3. 重启 Spiration，确认拓展已启用。
4. 在 `models.json` 中配置模型：

```json
{
    "deepseek-flash": {
        "name": "DeepSeek V4.1 Flash",
        "endpoint": "https://api.deepseek.com",
        "provider": "openai",
        "api_key": "sk-...",
        "model": "deepseek-flash"
    }
}
```

`endpoint` 是接口基地址；若未以 `/chat/completions` 结尾，拓展会自动追加该路径。

在 [Spiration 文档](https://flamence.github.io/spiration.docs/) 中了解更多。

## 功能

- SSE 流式响应：正文与思考过程的增量在收到时立即转发给界面。
- 工具调用：工具定义转换为 OpenAI function 格式；流式返回的工具调用分片会被
  累积为完整调用。
- 思考过程：解析 `reasoning_content`（兼容部分端点的 `reasoning` 别名），并
  按宿主的思考挡位映射 `reasoning_effort`。
- Token 用量：读取 `usage`（流式请求会携带 `stream_options.include_usage`）。
- 中止：用户请求中止后立即停止读取，并正常返回已收到的内容。
- 仅使用标准库（`urllib`），无任何第三方依赖。

## 兼容性

- 兼容任何 OpenAI Chat Completions 格式端点。
- 声明 `none/low/medium/high` 四个思考挡位：`none` 不携带
  `reasoning_effort`；`xhigh`/`max`（若传入）降级为 `high`。
- 对话历史中的 `reasoning_content` 不会发回服务端，因为各兼容实现对其处理
  方式并不一致。
- 同时支持非流式与 SSE 流式（`stream: true`）。网络、超时与 HTTP 错误会以
  可读消息报告；请求不会重试，避免重复计费。

## 贡献支持

详见 [为 Spiration 贡献](https://flamence.github.io/spiration.docs/contributing)。

## 协议许可

该项目仓库使用 [MIT License](LICENSE) 协议。
