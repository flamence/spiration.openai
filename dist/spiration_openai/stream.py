"""SSE（Server-Sent Events）解析与增量累积。

- :func:`iter_sse_payloads`：把响应行序列还原为 ``data:`` 载荷；
- :class:`StreamAssembler`：把流式 chunk 累积为宿主所需的完整响应，
  同时给出每个 chunk 的正文/思考增量。
"""

from typing import NamedTuple

from . import wire

#: SSE 流结束标记。
SSE_DONE = "[DONE]"


class Delta(NamedTuple):
    """单个 chunk 解析出的增量；空串表示该通道没有增量。"""

    content: str
    reasoning: str


def iter_sse_payloads(lines):
    """从 SSE 行序列中逐个产出 ``data:`` 字段的载荷文本。

    - 跳过空行、以 ``:`` 开头的注释行以及 ``event:`` / ``id:`` / ``retry:``
      等非数据字段；
    - 遇到 ``data: [DONE]`` 时结束迭代（其后的行不再读取）。
    """
    for raw in lines:
        if isinstance(raw, (bytes, bytearray)):
            line = bytes(raw).decode("utf-8", errors="replace")
        else:
            line = str(raw)
        line = line.rstrip("\r\n")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:]
        if payload.startswith(" "):
            payload = payload[1:]
        payload = payload.strip()
        if not payload:
            continue
        if payload == SSE_DONE:
            return
        yield payload


class _ToolCallAccumulator:
    """单个工具调用的累积状态（id 只在首个片段出现，名称与参数可能分片）。"""

    def __init__(self):
        self.id = ""
        self.name = ""
        self.argument_parts = []


class StreamAssembler:
    """累积流式 chunk，产出完整的补全响应。"""

    def __init__(self):
        self._content_parts = []
        self._reasoning_parts = []
        self._tool_calls = {}
        self.finish_reason = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def feed(self, chunk):
        """吸收一个 chunk，返回需要上报给宿主的增量。

        对 ``choices`` 为空的用量 chunk（最后一个 chunk）同样会读取 ``usage``。
        """
        if not isinstance(chunk, dict):
            return Delta("", "")

        self._feed_usage(chunk.get("usage"))

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return Delta("", "")
        choice = choices[0] if isinstance(choices[0], dict) else None
        if choice is None:
            return Delta("", "")

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.finish_reason = finish_reason

        delta = choice.get("delta")
        delta = delta if isinstance(delta, dict) else {}

        content = ""
        text = delta.get("content")
        if isinstance(text, str) and text:
            self._content_parts.append(text)
            content = text

        reasoning = ""
        reasoning_text = delta.get("reasoning_content")
        if not isinstance(reasoning_text, str) or not reasoning_text:
            # 部分兼容端点把该字段命名为 reasoning。
            reasoning_text = delta.get("reasoning")
        if isinstance(reasoning_text, str) and reasoning_text:
            self._reasoning_parts.append(reasoning_text)
            reasoning = reasoning_text

        self._accumulate_tool_calls(delta.get("tool_calls"))
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
        if not isinstance(usage, dict):
            return
        if "prompt_tokens" in usage:
            self.prompt_tokens = wire.token_count(usage.get("prompt_tokens"))
        if "completion_tokens" in usage:
            self.completion_tokens = wire.token_count(usage.get("completion_tokens"))

    def _accumulate_tool_calls(self, raw_calls):
        """按 ``index`` 累积工具调用片段。"""
        if not isinstance(raw_calls, list):
            return
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                index = 0
            call = self._tool_calls.get(index)
            if call is None:
                call = _ToolCallAccumulator()
                self._tool_calls[index] = call

            call_id = item.get("id")
            if isinstance(call_id, str) and call_id and not call.id:
                call.id = call_id

            function = item.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                call.name += name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                call.argument_parts.append(arguments)
