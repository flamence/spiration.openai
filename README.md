# Spiration OpenAI API Format Supports

[简体中文](README.zh_Hans.md) · [English](#)

An OpenAI Chat Completions compatible provider for Spiration's built-in agent.
The extension is written in Python and built on the official
[`openai`](https://pypi.org/project/openai/) Python SDK; it registers an agent
provider named `openai`, so any OpenAI-compatible endpoint can be used as the
conversation backend.

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

   On Windows the data directory is usually `%APPDATA%\spiration`. The SDK
   dependency is installed into that directory as well (see below), so no
   manual `pip install` is needed.
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

`endpoint` is the API **base URL**: the OpenAI SDK appends `/chat/completions`
(and any other API path) itself, so it must not include the full
`/chat/completions` path.

Learning more on [Spiration Documents](https://flamence.github.io/spiration.docs/en/).

## Requirements

- The Python extension loader (`com.flamence.spiration.python`).
- Python >= 3.10 (the current loader runs Python 3.11).
- The official `openai` SDK, version `>=3.37,<4`. The dependency is declared in
  `extension.json`:

  ```json
  {
      "python": {
          "requirements": ["openai>=3.37,<4"]
      }
  }
  ```

  Before importing the extension entry, the loader installs it into the
  extension's private data directory and adds it to `sys.path`. The first load
  therefore needs network access to the configured PyPI index; set
  `SPIRATION_PYTHON_DISABLE_PIP=1` to skip automatic installation when the
  dependency is already provided (offline or packaged deployments).

## Features

- Built on the official `openai` SDK: clients are created per
  `(endpoint, api_key, timeout)` and cached, with `max_retries=0` so requests
  are never retried behind your back.
- Streaming responses: text and reasoning deltas are forwarded to the UI as
  soon as they arrive.
- Tool calling: tool definitions are converted to the OpenAI function format,
  and streamed tool-call fragments are accumulated into complete calls.
- Reasoning content: `reasoning_content` (and the `reasoning` alias used by some
  compatible endpoints) is read from the SDK's extra fields, and
  `reasoning_effort` is derived from the host's reasoning level.
- Token usage from `usage` (`stream_options.include_usage` is sent for streaming
  responses; the final usage-only chunk is handled).
- Output limit: a positive host `max_tokens` is sent as the modern
  `max_completion_tokens` field, which OpenAI-compatible endpoints such as
  DeepSeek accept.
- Cancellation: iteration stops as soon as the host requests it, the SDK stream
  is closed, and the partial content is returned normally.
- Errors: SDK status/connection/timeout errors are reported as readable
  `RuntimeError`s (with status code and response body preview when available).

## Compatibility

- Works with any OpenAI Chat Completions compatible endpoint.
- Reasoning levels `none/low/medium/high` are declared. `none` sends no
  `reasoning_effort`; `xhigh`/`max` (if ever passed) are downgraded to `high`.
- `reasoning_content` from the conversation history is never sent back to the
  server, because compatible implementations handle it differently.
- Both non-streaming and streaming (`stream: true`) are supported. A reasoning
  model may return only `reasoning_content` with `content: null`; the null is
  preserved instead of being replaced by an empty string.

## Development

The test-suite only needs the declared dependency; SDK calls are stubbed, so no
network access is required:

```sh
py -3.11 -m pip install "openai>=3.37,<4"
py -3.11 -m unittest discover -s tests -v
```

## Contributing

For details, see [Contributing to Spiration](https://flamence.github.io/spiration.docs/en/contributing/).

## License

This repository is licensed under the [MIT License](LICENSE).
