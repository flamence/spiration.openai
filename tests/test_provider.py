"""``OpenAIProvider`` 单元测试：注入假 client / 假 SDK 对象，完全不联网。"""

import threading
import unittest

import openai

import support  # noqa: F401 - 导入即把 dist/ 加入 sys.path

import spiration_openai.provider as provider_module
from support import (
    FakeClient,
    FakeClientFactory,
    FakeSdkStream,
    FakeStream,
    make_chunk,
    make_completion,
    make_connection_error,
    make_delta,
    make_message,
    make_request,
    make_status_error,
    make_timeout_error,
    make_tool_call,
    make_usage,
    make_usage_chunk,
)
from spiration_openai.provider import OpenAIProvider


def _provider(result=None, error=None, clients=None):
    """构造注入了假 client 的 provider，返回 ``(provider, client, factory)``。"""
    client = FakeClient(result=result, error=error)
    factory = FakeClientFactory(clients if clients is not None else [client])
    return OpenAIProvider(client_factory=factory), client, factory


class _BrokenFactory:
    """构造客户端时抛异常的假工厂。"""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        raise self.error


class ProviderMetadataTests(unittest.TestCase):
    """provider 的注册元信息。"""

    def test_name_and_reasoning_levels(self):
        provider = OpenAIProvider()
        self.assertEqual(provider.name, "openai")
        self.assertEqual(
            provider.reasoning_levels(), ["none", "low", "medium", "high"]
        )

    def test_reasoning_levels_is_copy(self):
        provider = OpenAIProvider()
        levels = provider.reasoning_levels()
        levels.append("xhigh")
        self.assertEqual(
            provider.reasoning_levels(), ["none", "low", "medium", "high"]
        )


class NormalizeTests(unittest.TestCase):
    """超时/端点/密钥归一化。"""

    def test_timeout(self):
        self.assertEqual(
            provider_module.normalize_timeout(30), 30.0
        )
        self.assertEqual(provider_module.normalize_timeout(0.5), 0.5)
        for value in (None, 0, -1, "bad", True):
            self.assertEqual(
                provider_module.normalize_timeout(value),
                provider_module.DEFAULT_TIMEOUT_SECONDS,
                value,
            )

    def test_endpoint(self):
        self.assertEqual(
            provider_module.normalize_endpoint("  https://api.test/v1  "),
            "https://api.test/v1",
        )
        for value in (None, "", "   "):
            with self.assertRaises(RuntimeError):
                provider_module.normalize_endpoint(value)

    def test_api_key(self):
        self.assertEqual(provider_module.normalize_api_key("  sk-abc  "), "sk-abc")
        for value in (None, "", "   "):
            self.assertEqual(
                provider_module.normalize_api_key(value),
                provider_module.ANONYMOUS_API_KEY,
            )


class ClientCacheTests(unittest.TestCase):
    """client 缓存：按 (endpoint, api_key, timeout) 复用。"""

    def test_reuses_client_for_same_config(self):
        completion = make_completion(message=make_message(content="ok"))
        provider, client, factory = _provider(result=completion)
        for _ in range(2):
            provider.execute(make_request(), FakeStream())
        self.assertEqual(len(factory.init_kwargs), 1)
        self.assertEqual(len(client.calls), 2)

    def test_new_client_for_different_api_key(self):
        completion = make_completion(message=make_message(content="ok"))
        clients = [
            FakeClient(result=completion),
            FakeClient(result=completion),
        ]
        provider, _, factory = _provider(clients=clients)
        provider.execute(make_request(api_key="sk-a"), FakeStream())
        provider.execute(make_request(api_key="sk-b"), FakeStream())
        self.assertEqual(len(factory.init_kwargs), 2)
        self.assertEqual(len(clients[0].calls), 1)
        self.assertEqual(len(clients[1].calls), 1)

    def test_new_client_for_different_timeout(self):
        completion = make_completion(message=make_message(content="ok"))
        clients = [
            FakeClient(result=completion),
            FakeClient(result=completion),
        ]
        provider, _, factory = _provider(clients=clients)
        provider.execute(make_request(timeout_seconds=1), FakeStream())
        provider.execute(make_request(timeout_seconds=2), FakeStream())
        self.assertEqual(len(factory.init_kwargs), 2)

    def test_client_construction_options(self):
        completion = make_completion(message=make_message(content="ok"))
        provider, _, factory = _provider(result=completion)
        provider.execute(make_request(), FakeStream())
        self.assertEqual(
            factory.init_kwargs[0],
            {
                "api_key": "sk-test",
                "base_url": support.DEFAULT_ENDPOINT,
                "timeout": 5.0,
                "max_retries": 0,
            },
        )

    def test_empty_api_key_uses_placeholder(self):
        completion = make_completion(message=make_message(content="ok"))
        provider, _, factory = _provider(result=completion)
        provider.execute(make_request(api_key=""), FakeStream())
        self.assertEqual(
            factory.init_kwargs[0]["api_key"], provider_module.ANONYMOUS_API_KEY
        )

    def test_concurrent_calls_share_cached_client(self):
        completion = make_completion(message=make_message(content="ok"))
        provider, client, factory = _provider(result=completion)
        errors = []

        def run():
            try:
                provider.execute(make_request(), FakeStream())
            except Exception as err:  # noqa: BLE001 - 交给主线程断言
                errors.append(err)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(len(factory.init_kwargs), 1)
        self.assertEqual(len(client.calls), 4)

    def test_client_construction_error_mapped(self):
        factory = _BrokenFactory(openai.OpenAIError("Missing credentials."))
        provider = OpenAIProvider(client_factory=factory)
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(api_key=""), FakeStream())
        self.assertIn("OpenAI 客户端错误", str(caught.exception))
        self.assertIn("Missing credentials", str(caught.exception))


class NonStreamTests(unittest.TestCase):
    """非流式执行。"""

    def test_create_kwargs_and_result(self):
        completion = make_completion(
            message=make_message(content="最终答案"),
            finish_reason="stop",
            usage=make_usage(100, 20),
        )
        provider, client, _ = _provider(result=completion)

        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "读文件"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "我需要调用工具",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function_name": "read_file",
                        "arguments": '{"path":"a"}',
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "内容"},
        ]
        tools = [
            {
                "name": "read_file",
                "description": "读取文件",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        request = make_request(
            messages=messages,
            tools=tools,
            max_tokens=64,
            temperature=0.3,
            reasoning="medium",
        )
        stream = FakeStream()
        result = provider.execute(request, stream)

        self.assertEqual(
            result,
            {
                "content": "最终答案",
                "reasoning_content": None,
                "tool_calls": [],
                "finish_reason": "stop",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            },
        )
        # 非流式不调用流式回调。
        self.assertEqual(stream.content_chunks, [])
        self.assertEqual(stream.reasoning_chunks, [])

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            client.calls[0],
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "你是助手"},
                    {"role": "user", "content": "读文件"},
                    {
                        "role": "assistant",
                        "content": None,
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
                    {"role": "tool", "tool_call_id": "call_1", "content": "内容"},
                ],
                "stream": False,
                "temperature": 0.3,
                "max_completion_tokens": 64,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "读取文件",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "reasoning_effort": "medium",
            },
        )

    def test_response_with_reasoning_and_tool_calls(self):
        completion = make_completion(
            message=make_message(
                content="最终答案",
                reasoning_content="答案思考",
                tool_calls=[
                    make_tool_call(0, "call_9", "sleep", '{"seconds":1}')
                ],
            ),
            finish_reason="tool_calls",
            usage=make_usage(100, 20),
        )
        provider, _, _ = _provider(result=completion)
        result = provider.execute(make_request(), FakeStream())
        self.assertEqual(
            result,
            {
                "content": "最终答案",
                "reasoning_content": "答案思考",
                "tool_calls": [
                    {
                        "id": "call_9",
                        "function_name": "sleep",
                        "arguments": '{"seconds":1}',
                    }
                ],
                "finish_reason": "tool_calls",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            },
        )

    def test_reasoning_only_response_keeps_content_none(self):
        completion = make_completion(
            message=make_message(content=None, reasoning_content="只思考"),
            finish_reason="length",
            usage=make_usage(41, 32),
        )
        provider, _, _ = _provider(result=completion)
        result = provider.execute(make_request(), FakeStream())
        self.assertIsNone(result["content"])
        self.assertEqual(result["reasoning_content"], "只思考")

    def test_invalid_request_type(self):
        provider, client, factory = _provider(result=make_completion())
        for request in (None, "x", [], 42):
            with self.assertRaises(RuntimeError):
                provider.execute(request, FakeStream())
        self.assertEqual(factory.init_kwargs, [])
        self.assertEqual(client.calls, [])

    def test_missing_endpoint_raises_without_client(self):
        provider, client, factory = _provider(result=make_completion())
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(endpoint="  "), FakeStream())
        self.assertIn("endpoint", str(caught.exception))
        self.assertEqual(factory.init_kwargs, [])
        self.assertEqual(client.calls, [])


class StreamTests(unittest.TestCase):
    """流式执行。"""

    def test_stream_result_deltas_and_kwargs(self):
        chunks = [
            make_chunk(make_delta(reasoning_content="想")),
            make_chunk(make_delta(reasoning_content="一想")),
            make_chunk(make_delta(content="你好")),
            make_chunk(make_delta(content="！")),
            make_chunk(
                make_delta(
                    tool_calls=[
                        make_tool_call(0, "call_1", "read_file", '{"path":"a"}')
                    ]
                )
            ),
            make_chunk(finish_reason="tool_calls"),
            make_usage_chunk(12, 7),
        ]
        sdk_stream = FakeSdkStream(chunks)
        provider, client, _ = _provider(result=sdk_stream)
        host_stream = FakeStream()
        result = provider.execute(make_request(stream=True), host_stream)

        self.assertEqual(
            result,
            {
                "content": "你好！",
                "reasoning_content": "想一想",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function_name": "read_file",
                        "arguments": '{"path":"a"}',
                    }
                ],
                "finish_reason": "tool_calls",
                "prompt_tokens": 12,
                "completion_tokens": 7,
            },
        )
        self.assertEqual(host_stream.content_chunks, ["你好", "！"])
        self.assertEqual(host_stream.reasoning_chunks, ["想", "一想"])
        self.assertEqual(sdk_stream.consumed, len(chunks))
        self.assertTrue(sdk_stream.closed)

        kwargs = client.calls[0]
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})
        self.assertNotIn("reasoning_effort", kwargs)

    def test_stream_abort_returns_partial_content(self):
        chunks = [
            make_chunk(make_delta(content="A")),
            make_chunk(make_delta(content="B"), finish_reason="stop"),
            make_chunk(make_delta(content="C")),
        ]
        sdk_stream = FakeSdkStream(chunks)
        provider, _, _ = _provider(result=sdk_stream)
        host_stream = FakeStream(stop_after_deltas=1)
        result = provider.execute(make_request(stream=True), host_stream)

        self.assertEqual(result["content"], "A")
        self.assertEqual(result["finish_reason"], "")
        self.assertEqual(host_stream.content_chunks, ["A"])
        self.assertEqual(sdk_stream.consumed, 1)
        self.assertTrue(sdk_stream.closed)

    def test_stream_reasoning_only(self):
        chunks = [
            make_chunk(make_delta(reasoning_content="思考")),
            make_chunk(make_delta(reasoning_content="过程")),
        ]
        sdk_stream = FakeSdkStream(chunks)
        provider, _, _ = _provider(result=sdk_stream)
        host_stream = FakeStream()
        result = provider.execute(make_request(stream=True), host_stream)
        self.assertIsNone(result["content"])
        self.assertEqual(result["reasoning_content"], "思考过程")
        self.assertEqual(host_stream.reasoning_chunks, ["思考", "过程"])
        self.assertEqual(result["prompt_tokens"], 0)
        self.assertEqual(result["completion_tokens"], 0)

    def test_stream_create_error_mapped(self):
        provider, client, _ = _provider(
            error=make_status_error(400, text='{"error": {"message": "bad request"}}')
        )
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(stream=True), FakeStream())
        self.assertIn("400", str(caught.exception))
        self.assertIn("bad request", str(caught.exception))
        # 不重试：只发起一次请求。
        self.assertEqual(len(client.calls), 1)

    def test_stream_iteration_error_mapped_and_closed(self):
        sdk_stream = FakeSdkStream(
            [make_chunk(make_delta(content="开始"))],
            error=make_connection_error(),
        )
        provider, _, _ = _provider(result=sdk_stream)
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(stream=True), FakeStream())
        self.assertIn("连接", str(caught.exception))
        self.assertTrue(sdk_stream.closed)

    def test_stream_close_error_swallowed(self):
        class _BadCloseStream(FakeSdkStream):
            def close(self):
                super().close()
                raise OSError("close failed")

        sdk_stream = _BadCloseStream([make_chunk(make_delta(content="ok"))])
        provider, _, _ = _provider(result=sdk_stream)
        result = provider.execute(make_request(stream=True), FakeStream())
        self.assertEqual(result["content"], "ok")
        self.assertTrue(sdk_stream.closed)


class ErrorMappingTests(unittest.TestCase):
    """SDK 异常 → RuntimeError（不重试）。"""

    def test_status_error_with_body_preview(self):
        provider, client, _ = _provider(
            error=make_status_error(
                401, text='{"error": {"message": "invalid api key"}}'
            )
        )
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(), FakeStream())
        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("invalid api key", message)
        self.assertEqual(len(client.calls), 1)

    def test_status_error_without_response_text(self):
        provider, _, _ = _provider(
            error=make_status_error(
                500,
                text="",
                body={"error": {"message": "server overloaded"}},
            )
        )
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(), FakeStream())
        message = str(caught.exception)
        self.assertIn("500", message)
        self.assertIn("server overloaded", message)

    def test_timeout_error(self):
        provider, _, _ = _provider(error=make_timeout_error())
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(), FakeStream())
        message = str(caught.exception)
        self.assertIn("超时", message)
        self.assertIn("timed out", message)

    def test_connection_error(self):
        provider, _, _ = _provider(
            error=make_connection_error("Connection error.")
        )
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(), FakeStream())
        self.assertIn("连接", str(caught.exception))

    def test_generic_api_error(self):
        provider, _, _ = _provider(
            error=openai.APIError("服务器内部错误", request=None, body=None)
        )
        with self.assertRaises(RuntimeError) as caught:
            provider.execute(make_request(), FakeStream())
        self.assertIn("服务器内部错误", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
