"""Spiration 的 OpenAI Chat Completions 格式 provider 实现。

实现基于官方 ``openai`` Python SDK（``extension.json`` 的
``python.requirements`` 声明依赖，由加载器在导入前安装）。模块划分：

- ``wire``：宿主 JSON 与 SDK 请求/响应对象之间的转换；
- ``stream``：SDK 流式 chunk 的累积；
- ``provider``：客户端缓存与一次补全请求的完整执行流程。
"""

from .provider import OpenAIProvider

__all__ = ["OpenAIProvider"]
