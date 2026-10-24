"""与真实 openai SDK 模型对象的兼容性测试（离线构造，不发起任何网络请求）。

``test_wire`` / ``test_stream`` 使用 ``SimpleNamespace`` 模拟 SDK 对象以保证
测试快速、可读；本文件再用真实的 ``ChatCompletion`` / ``ChatCompletionChunk``
模型对象（从 JSON 构造）验证同一套解析逻辑，防止假对象与 SDK 实际行为脱节。
"""

import unittest

import support  # noqa: F401 - 导入即把 dist/ 加入 sys.path

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from support import FakeClient, FakeClientFactory, FakeSdkStream, FakeStream
from spiration_openai.provider import OpenAIProvider
from spiration_openai.stream import Delta, StreamAssembler
from spiration_openai import wire


def _chunk(**overrides):
    """从 JSON 构造真实的 ``ChatCompletionChunk`` 模型对象。"""
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [],
    }
    payload.update(overrides)
    return ChatCompletionChunk.model_validate(payload)


def _completion(**overrides):
    """从 JSON 构造真实的 ``ChatCompletion`` 模型对象。"""
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [],
    }
    payload.update(overrides)
    return ChatCompletion.model_validate(payload)


class RealCompletionTests(unittest.TestCase):
    """真实非流式响应对象。"""

    def test_reasoning_extra_field_and_tool_calls(self):
        completion = _completion(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "先思考",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"a"}',
                                },
                            }
                        ],
                    },
                }
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        # SDK 把服务端扩展字段放进 model_extra，同时可直接 getattr。
        message = completion.choices[0].message
        self.assertEqual(message.model_extra.get("reasoning_content"), "先思考")
        self.assertEqual(getattr(message, "reasoning_content", None), "先思考")

        self.assertEqual(
            wire.parse_response(completion),
            {
                "content": None,
                "reasoning_content": "先思考",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function_name": "read_file",
                        "arguments": '{"path":"a"}',
                    }
                ],
                "finish_reason": "tool_calls",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        )

    def test_reasoning_only_content_stays_none(self):
        completion = _completion(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "只思考",
                    },
                }
            ],
        )
        response = wire.parse_response(completion)
        self.assertIsNone(response["content"])
        self.assertEqual(response["reasoning_content"], "只思考")
        self.assertEqual(response["prompt_tokens"], 0)


class RealChunkTests(unittest.TestCase):
    """真实流式 chunk 对象。"""

    def test_stream_accumulation(self):
        chunks = [
            _chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": "想"},
                        "finish_reason": None,
                    }
                ]
            ),
            _chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read_", "arguments": '{"p'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            ),
            _chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "file",
                                        "arguments": ':1}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            ),
            _chunk(
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            ),
        ]
        assembler = StreamAssembler()
        deltas = [assembler.feed(chunk) for chunk in chunks]
        self.assertEqual(deltas[0], Delta(content="", reasoning="想"))
        self.assertEqual(
            assembler.result(),
            {
                "content": None,
                "reasoning_content": "想",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function_name": "read_file",
                        "arguments": '{"p:1}',
                    }
                ],
                "finish_reason": "tool_calls",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        )


class ProviderWithRealObjectsTests(unittest.TestCase):
    """provider 全流程使用真实 SDK 响应对象（client 仍是假对象，不联网）。"""

    def test_non_stream(self):
        completion = _completion(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "答案"},
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )
        client = FakeClient(result=completion)
        provider = OpenAIProvider(client_factory=FakeClientFactory([client]))
        result = provider.execute(
            support.make_request(), FakeStream()
        )
        self.assertEqual(result["content"], "答案")
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["completion_tokens"], 2)

    def test_stream(self):
        chunks = [
            _chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": "你"},
                        "finish_reason": None,
                    }
                ]
            ),
            _chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": "好"},
                        "finish_reason": "stop",
                    }
                ]
            ),
        ]
        sdk_stream = FakeSdkStream(chunks)
        client = FakeClient(result=sdk_stream)
        provider = OpenAIProvider(client_factory=FakeClientFactory([client]))
        host_stream = FakeStream()
        result = provider.execute(
            support.make_request(stream=True), host_stream
        )
        self.assertEqual(result["content"], "你好")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(host_stream.content_chunks, ["你", "好"])
        self.assertTrue(sdk_stream.closed)


if __name__ == "__main__":
    unittest.main()
