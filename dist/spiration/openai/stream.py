"""官方 ``openai`` SDK 流式 chunk 的累积器。

- :class:`StreamAssembler`：把 ``ChatCompletionChunk`` 序列累积为宿主所需的
  完整响应，同时给出每个 chunk 的正文/思考增量；
- ``chunk.choices`` 可能为空（纯 usage chunk）、``chunk.usage`` 可能为 None，
  这些都由 SDK 的模型对象表达；
- 服务端扩展字段（``reasoning_content``）的读取统一走 :func:`wire.reasoning_text`。
"""

from typing import NamedTuple

from . import wire


class Delta(NamedTuple):
    """单个 chunk 解析出的增量；空串表示该通道没有增量。"""

    content: str
    reasoning: str


class _ToolCallAccumulator:
    """单个工具调用的累积状态（id 只在首个片段出现，名称与参数可能分片）。"""

    def __init__(self):
        self.id = ""
        self.name = ""
        self.argument_parts = []


class StreamAssembler:
    """累积 SDK 流式 chunk，产出完整的补全响应。"""

    def __init__(self):
        self._content_parts = []
        self._reasoning_parts = []
        self._tool_calls = {}
        self.finish_reason = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def feed(self, chunk):
        """吸收一个 ``ChatCompletionChunk``，返回需要上报给宿主的增量。

        对 ``choices`` 为空的用量 chunk（通常是最后一个 chunk）同样会读取
        ``usage``。
        """
        if chunk is None:
            return Delta("", "")

        self._feed_usage(getattr(chunk, "usage", None))

        choices = getattr(chunk, "choices", None)
        if not isinstance(choices, (list, tuple)) or not choices:
            return Delta("", "")
        choice = choices[0]
        if choice is None:
            return Delta("", "")

        finish_reason = getattr(choice, "finish_reason", None)
        if isinstance(finish_reason, str) and finish_reason:
            self.finish_reason = finish_reason

        delta = getattr(choice, "delta", None)
        if delta is None:
            return Delta("", "")

        content = ""
        text = getattr(delta, "content", None)
        if isinstance(text, str) and text:
            self._content_parts.append(text)
            content = text

        reasoning = ""
        reasoning_text = wire.reasoning_text(delta)
        if reasoning_text:
            self._reasoning_parts.append(reasoning_text)
            reasoning = reasoning_text

        self._accumulate_tool_calls(getattr(delta, "tool_calls", None))
        return Delta(content, reasoning)

    def result(self):
        """产出宿主响应结构（完整累积的正文/思考过程/工具调用与用量）。"""
        content = "".join(self._content_parts) or None
        reasoning_content = "".join(self._reasoning_parts) or None
        tool_calls = []
        for index in sorted(self._tool_calls):
            call = self._tool_calls[index]
            tool_calls.append(
                {
                    "id": call.id,
                    "function_name": call.name,
                    "arguments": "".join(call.argument_parts),
                }
            )
        return {
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "finish_reason": self.finish_reason or "",
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def _feed_usage(self, usage):
        """吸收 token 用量；字段缺省时保留已有值。"""
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is not None:
            self.prompt_tokens = wire.token_count(prompt_tokens)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is not None:
            self.completion_tokens = wire.token_count(completion_tokens)

    def _accumulate_tool_calls(self, raw_calls):
        """按 SDK 元素自带的 ``index`` 累积工具调用片段。"""
        if not isinstance(raw_calls, (list, tuple)):
            return
        for item in raw_calls:
            if item is None:
                continue
            index = getattr(item, "index", None)
            if isinstance(index, bool) or not isinstance(index, int):
                index = 0
            call = self._tool_calls.get(index)
            if call is None:
                call = _ToolCallAccumulator()
                self._tool_calls[index] = call

            call_id = getattr(item, "id", None)
            if isinstance(call_id, str) and call_id and not call.id:
                call.id = call_id

            function = getattr(item, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                call.name += name
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str) and arguments:
                call.argument_parts.append(arguments)
