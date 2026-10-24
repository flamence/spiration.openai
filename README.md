# Spiration OpenAI API Format Supports

[简体中文](README.zh_Hans.md) · [English](#)

An OpenAI Chat Completions compatible provider for Spiration's built-in agent.
The extension is written in Python and registers an agent provider named
`openai`, so any OpenAI-compatible endpoint can be used as the conversation
backend.

## Quick Start

1. Make sure the Python extension loader (`com.flamence.spiration.python`) is
   installed and enabled.
2. Install this extension into Spiration's data directory, keeping the directory
   name equal to the extension ID:

   ```
   <data directory>/com.flamence.spiration.openai/
       extension.json
       dist/extension.py
       dist/spiration_openai/
   ```

   On Windows the data directory is usually `%APPDATA%\spiration`.
3. Restart Spiration and make sure the extension is enabled.
4. Configure the model in `models.json`:

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

`endpoint` is the API base URL; `/chat/completions` is appended automatically
unless it is already present.

Learning more on [Spiration Documents](https://flamence.github.io/spiration.docs/en/).

## Features

- Streaming responses over SSE: text and reasoning deltas are forwarded to the
  UI as soon as they arrive.
- Tool calling: tool definitions are converted to the OpenAI function format,
  and streamed tool-call fragments are accumulated into complete calls.
- Reasoning content: `reasoning_content` (and the `reasoning` alias used by some
  compatible endpoints) is parsed, and `reasoning_effort` is derived from the
  host's reasoning level.
- Token usage from `usage` (`stream_options.include_usage` is sent for streaming
  responses).
- Cancellation: reading stops as soon as the host requests it, and the partial
  content is returned normally.
- Standard library only (`urllib`): no third-party dependency.

## Compatibility

- Works with any OpenAI Chat Completions compatible endpoint.
- Reasoning levels `none/low/medium/high` are declared. `none` sends no
  `reasoning_effort`; `xhigh`/`max` (if ever passed) are downgraded to `high`.
- `reasoning_content` from the conversation history is never sent back to the
  server, because compatible implementations handle it differently.
- Both non-streaming and SSE streaming (`stream: true`) are supported. Network,
  timeout and HTTP errors are reported with a readable message; requests are
  never retried to avoid duplicated billing.

## Contributing

For details, see [Contributing to Spiration](https://flamence.github.io/spiration.docs/en/contributing/).

## License

This repository is licensed under the [MIT License](LICENSE).
