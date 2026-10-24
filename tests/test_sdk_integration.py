"""离线端到端测试：真实 openai 客户端 + ``httpx2.MockTransport``（不联网）。

其他测试文件用假 client 把整个 SDK 桩掉；本文件只把网络层换成本地
``MockTransport``，从而额外验证真实 SDK 侧的行为：

- ``base_url`` 拼接成 ``<endpoint>/chat/completions``；
- 我们传入的 ``max_completion_tokens`` / ``stream_options`` / ``tools`` /
  ``reasoning_effort`` 等参数被 SDK 接受并序列化；
- 真实 SSE chunk 被 SDK 解析后由 :class:`StreamAssembler` 正确累积；
- 真实 ``APIStatusError`` / ``APIConnectionError`` 的映射结果。
"""

import json
import unittest

import httpx2
import openai

import support
from support import FakeStream, make_request
from spiration_openai.provider import OpenAIProvider


def _sse_body(chunks):
    """把 chunk 列表编码成 OpenAI 流式响应的 SSE 字节。"""
    body = b""
    for chunk in chunks:
        body += b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
    return body + b"data: [DONE]\n\n"


def _chunk(delta, finish_reason=None):
    """构造流式 chunk 的 JSON 文档。"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _usage_chunk(prompt_tokens, completion_tokens):
    """构造只有 usage 的最后一个 chunk。"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class RealSdkClientTests(unittest.TestCase):
    """真实 SDK 客户端 + 假传输层的端到端行为。"""

    def _provider(self, handler):
        """构造 provider，其真实 SDK 客户端的网络层由 ``handler`` 模拟。"""
        transport = httpx2.MockTransport(handler)
        http_client = httpx2.Client(transport=transport)

        def factory(**kwargs):
            return openai.OpenAI(
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
                timeout=kwargs["timeout"],
                max_retries=kwargs["max_retries"],
                http_client=http_client,
            )

        return OpenAIProvider(client_factory=factory)

    def test_non_stream_request_and_response(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            captured["authorization"] = request.headers.get("authorization")
            return httpx2.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "最终答案",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    },
                },
            )

        provider = self._provider(handler)
        request = make_request(
            max_tokens=64,
            temperature=0.3,
            reasoning="medium",
            tools=[
                {
                    "name": "read_file",
                    "description": "读取文件",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        result = provider.execute(request, FakeStream())

        self.assertEqual(
            captured["url"],
            support.DEFAULT_ENDPOINT + "/chat/completions",
        )
        self.assertEqual(captured["authorization"], "Bearer sk-test")
        self.assertEqual(captured["body"]["model"], "test-model")
        self.assertFalse(captured["body"]["stream"])
        self.assertEqual(captured["body"]["max_completion_tokens"], 64)
        self.assertEqual(captured["body"]["reasoning_effort"], "medium")
        self.assertEqual(captured["body"]["temperature"], 0.3)
        self.assertEqual(
            captured["body"]["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "读取文件",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        self.assertEqual(result["content"], "最终答案")
        self.assertEqual(result["prompt_tokens"], 7)
        self.assertEqual(result["completion_tokens"], 3)

    def test_stream_request_and_response(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            chunks = [
                _chunk({"role": "assistant", "reasoning_content": "想"}),
                _chunk({"content": "你好"}),
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_",
                                    "arguments": '{"pa',
                                },
                            }
                        ]
                    }
                ),
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "file",
                                    "arguments": 'th":"a"}',
                                },
                            }
                        ]
                    }
                ),
                _chunk({}, finish_reason="tool_calls"),
                _usage_chunk(12, 7),
            ]
            return httpx2.Response(
                200,
                content=_sse_body(chunks),
                headers={"content-type": "text/event-stream"},
            )

        provider = self._provider(handler)
        host_stream = FakeStream()
        result = provider.execute(make_request(stream=True), host_stream)

        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(
            captured["body"]["stream_options"], {"include_usage": True}
        )
        self.assertEqual(result["content"], "你好")
        self.assertEqual(result["reasoning_content"], "想")
        self.assertEqual(
            result["tool_calls"],
            [
                {
                    "id": "call_1",
                    "function_name": "read_file",
                    "arguments": '{"path":"a"}',
                }
            ],
        )
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["prompt_tokens"], 12)
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(host_stream.content_chunks, ["你好"])
        self.assertEqual(host_stream.reasoning_chunks, ["想"])

    def test_abort_stops_reading(self):
        def handler(request):
            chunks = [
                _chunk({"content": "A"}),
                _chunk({"content": "B"}),
                _chunk({"content": "C"}, finish_reason="stop"),
            ]
            return httpx2.Response(
                200,
                content=_sse_body(chunks),
                headers={"content-type": "text/event-stream"},
            )

        provider = self._provider(handler)
        host_stream = FakeStream(stop_after_deltas=1)
        result = provider.execute(make_request(stream=True), host_stream)

        self.assertEqual(result["content"], "A")
        self.assertEqual(result["finish_reason"], "")
        self.assertEqual(host_stream.content_chunks, ["A"])

    def test_status_error_mapped(self):
        def handler(request):
            return httpx2.Response(
                401,
                json={"error": {"message": "invalid api key", "type": "auth"}},
            )

        provider = self._provider(handler)
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(), FakeStream())
        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("invalid api key", message)

    def test_connection_error_mapped(self):
        def handler(request):
            raise httpx2.ConnectError("connection refused")

        provider = self._provider(handler)
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(), FakeStream())
        message = str(caught.exception)
        self.assertIn("OpenAI 连接失败", message)
        self.assertIn("Connection error", message)


if __name__ == "__main__":
    unittest.main()
