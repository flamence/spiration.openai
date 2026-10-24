# Spiration OpenAI API 格式支持

[简体中文](#) · [English](README.md)

为 Spiration 内置智能体提供 OpenAI Chat Completions 格式的 provider。本拓展
以 Python 编写、基于官方 [`openai`](https://pypi.org/project/openai/) Python SDK，
向宿主注册名为 `openai` 的 provider，因此任何兼容 OpenAI Chat Completions 的
端点都可以作为对话后端。

## 快速开始

1. 确认已安装并启用 Python 拓展加载器（`com.flamence.spiration.python`）。
2. 把本拓展安装到 Spiration 的数据目录，并保持目录名与拓展 ID 一致：

   ```
   <数据目录>/com.flamence.spiration.openai/
       extension.json
       dist/extension.py
       dist/spiration_openai/
   ```

   Windows 下数据目录通常为 `%APPDATA%\spiration`。SDK 依赖也会安装到该目录下
   （见下文），无需手动 `pip install`。
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

`endpoint` 是接口**基地址**：OpenAI SDK 会自行拼接 `/chat/completions`（以及
其它 API 路径），因此不要填写完整的 `/chat/completions` 地址。

在 [Spiration 文档](https://flamence.github.io/spiration.docs/) 中了解更多。

## 环境要求

- Python 拓展加载器（`com.flamence.spiration.python`）。
- Python >= 3.10（当前加载器使用 Python 3.11）。
- 官方 `openai` SDK，版本 `>=3.37,<4`。依赖在 `extension.json` 中声明：

  ```json
  {
      "python": {
          "requirements": ["openai>=3.37,<4"]
      }
  }
  ```

  在导入拓展入口脚本之前，加载器会把它安装到拓展私有的数据目录并加入
  `sys.path`。因此首次加载需要能访问所配置的 PyPI 源；离线或打包部署可设置
  `SPIRATION_PYTHON_DISABLE_PIP=1` 跳过自动安装（此时需已预置依赖）。

## 功能

- 基于官方 `openai` SDK：客户端按 `(端点, API key, 超时)` 缓存，设置
  `max_retries=0`，请求不会在背后自动重试。
- 流式响应：正文与思考过程的增量在收到时立即转发给界面。
- 工具调用：工具定义转换为 OpenAI function 格式；流式返回的工具调用分片会按
  index 累积为完整调用。
- 思考过程：从 SDK 的扩展字段读取 `reasoning_content`（兼容部分端点的
  `reasoning` 别名），并据宿主的思考挡位映射 `reasoning_effort`。
- Token 用量：读取 `usage`（流式请求会携带 `stream_options.include_usage`，并
  处理最后一个只含 usage 的 chunk）。
- 输出上限：宿主给出的正数 `max_tokens` 会以现代字段 `max_completion_tokens`
  发送，DeepSeek 等 OpenAI 兼容端点同样接受。
- 中止：用户请求中止后立即停止迭代、关闭 SDK 流，并正常返回已收到的内容。
- 错误：SDK 的状态码/连接/超时异常统一转换为可读的 `RuntimeError`（尽量附上
  状态码与响应体摘要）。

## 兼容性

- 兼容任何 OpenAI Chat Completions 格式端点。
- 声明 `none/low/medium/high` 四个思考挡位：`none` 不携带
  `reasoning_effort`；`xhigh`/`max`（若传入）降级为 `high`。
- 对话历史中的 `reasoning_content` 不会发回服务端，因为各兼容实现对其处理
  方式并不一致。
- 同时支持非流式与流式（`stream: true`）。思考模型可能只返回
  `reasoning_content` 而 `content` 为 null，本拓展会原样保留 null，不会伪造
  空串。

## 开发

测试只依赖清单声明的 `openai` 包（SDK 调用均以桩对象替代，不需要联网）：

```sh
py -3.11 -m pip install "openai>=3.37,<4"
py -3.11 -m unittest discover -s tests -v
```

## 贡献支持

详见 [为 Spiration 贡献](https://flamence.github.io/spiration.docs/contributing)。

## 协议许可

该项目仓库使用 [MIT License](LICENSE) 协议。
