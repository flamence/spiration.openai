"""Spiration 的 OpenAI Chat Completions 格式 provider 实现。

模块划分：

- ``wire``：宿主 JSON 与 OpenAI 线上格式之间的转换；
- ``stream``：SSE 解析与流式增量累积；
- ``transport``：基于标准库 ``urllib`` 的 HTTP 传输；
- ``provider``：一次补全请求的完整执行流程。
"""

from .provider import OpenAIProvider

__all__ = ["OpenAIProvider"]
