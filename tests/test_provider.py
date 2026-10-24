"""OpenAIProvider 端到端测试：本地 http.server 模拟 Chat Completions 端点。

所有测试只访问 127.0.0.1 上临时启动的 HTTP 服务，不依赖外网与宿主。
"""

import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from support import FakeStream, make_request

from spiration_openai.provider import OpenAIProvider


class _Handler(BaseHTTPRequestHandler):
    """把请求记录到服务器上，再交给当前 responder 生成响应。"""

    # 默认 HTTP/1.0：响应结束即关闭连接，无需 Content-Length，便于测试流式。
    protocol_version = "HTTP/1.0"

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler 的约定命名
        """处理一次 POST 请求。"""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            body = {}
        with self.server.lock:
            self.server.requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                }
            )
        try:
            self.server.responder(self, body)
        except OSError:
            # 客户端提前断开（中止路径）时写入失败属预期，忽略。
            pass
        except Exception as err:  # noqa: BLE001 - 交给测试线程断言
            self.server.errors.append(err)

    def log_message(self, fmt, *args):  # noqa: A003 - 覆盖基类以静默测试输出
        """关闭默认的访问日志输出。"""


class _MockServer:
    """在 127.0.0.1 的随机端口上运行一个 OpenAI 兼容端点。"""

    def __init__(self, responder):
        self.httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.responder = responder
        self.httpd.requests = []
        self.httpd.errors = []
        self.httpd.lock = threading.Lock()
        self._thread = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )

    @property
    def endpoint(self):
        """端点地址（不含 /chat/completions）。"""
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self):
        """收到的请求记录列表。"""
        with self.httpd.lock:
            return list(self.httpd.requests)

    @property
    def errors(self):
        """responder 中未被预期的异常列表。"""
        return list(self.httpd.errors)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)
        return False


def _write_json(handler, payload, status=200):
    """写出 JSON 响应。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_sse(handler, chunks, delay=0.0):
    """把 chunk 列表写成 SSE 流（末尾追加 [DONE]）。"""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()
    for chunk in chunks:
        if delay:
            time.sleep(delay)
        data = json.dumps(chunk, ensure_ascii=False).encode("utf-8")
        handler.wfile.write(b"data: " + data + b"\n\n")
        handler.wfile.flush()
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


def _header(headers, name):
    """大小写不敏感地取请求头。"""
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


class ProviderMetadataTests(unittest.TestCase):
    """provider 的注册元信息。"""

    def test_name_and_reasoning_levels(self):
        provider = OpenAIProvider()
        self.assertEqual(provider.name, "openai")
        self.assertEqual(
            provider.reasoning_levels(), ["none", "low", "medium", "high"]
        )


class NonStreamTests(unittest.TestCase):
    """非流式端到端。"""

    def test_request_and_response(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": "最终答案",
                        "reasoning_content": "答案思考",
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "type": "function",
                                "function": {
                                    "name": "sleep",
                                    "arguments": '{"seconds":1}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
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

        with _MockServer(
            lambda handler, body: _write_json(handler, response_payload)
        ) as server:
            request = make_request(
                server.endpoint,
                messages=messages,
                tools=tools,
                max_tokens=64,
                temperature=0.3,
                reasoning="medium",
                stream=False,
            )
            stream = FakeStream()
            result = OpenAIProvider().execute(request, stream)
            records = server.requests
            self.assertEqual(server.errors, [])

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
                "finish_reason": "stop",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            },
        )
        # 非流式不触发流式回调。
        self.assertEqual(stream.content_chunks, [])
        self.assertEqual(stream.reasoning_chunks, [])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["path"], "/chat/completions")
        self.assertEqual(
            _header(records[0]["headers"], "Authorization"), "Bearer sk-test"
        )
        self.assertEqual(
            _header(records[0]["headers"], "Content-Type"), "application/json"
        )
        self.assertIsNone(_header(records[0]["headers"], "Accept"))

        body = records[0]["body"]
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["max_tokens"], 64)
        self.assertNotIn("stream_options", body)
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(
            body["tools"],
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
        self.assertEqual(
            body["messages"],
            [
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
        )
        # reasoning_content 不能被发回服务端。
        self.assertNotIn("reasoning_content", body["messages"][2])

    def test_endpoint_with_full_path(self):
        with _MockServer(
            lambda handler, body: _write_json(
                handler, {"choices": [{"message": {"content": "ok"}}]}
            )
        ) as server:
            request = make_request(server.endpoint + "/chat/completions")
            result = OpenAIProvider().execute(request, FakeStream())
            records = server.requests
        self.assertEqual(result["content"], "ok")
        self.assertEqual(records[0]["path"], "/chat/completions")

    def test_http_error_raises_with_body_preview(self):
        error_payload = {"error": {"message": "invalid api key"}}
        with _MockServer(
            lambda handler, body: _write_json(handler, error_payload, status=401)
        ) as server:
            request = make_request(server.endpoint)
            with self.assertRaises(RuntimeError) as caught:
                OpenAIProvider().execute(request, FakeStream())
            self.assertEqual(len(server.requests), 1)  # 不重试

        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("invalid api key", message)

    def test_error_payload_with_http_200_raises(self):
        with _MockServer(
            lambda handler, body: _write_json(
                handler, {"error": {"message": "quota exceeded"}}
            )
        ) as server:
            request = make_request(server.endpoint)
            with self.assertRaises(RuntimeError) as caught:
                OpenAIProvider().execute(request, FakeStream())
        self.assertIn("quota exceeded", str(caught.exception))

    def test_invalid_json_response_raises(self):
        def responder(handler, body):
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(b"not json at all")

        with _MockServer(responder) as server:
            with self.assertRaises(RuntimeError) as caught:
                OpenAIProvider().execute(
                    make_request(server.endpoint), FakeStream()
                )
        self.assertIn("解析", str(caught.exception))

    def test_connection_error_raises(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        request = make_request(f"http://127.0.0.1:{port}", timeout_seconds=1)
        with self.assertRaises(RuntimeError):
            OpenAIProvider().execute(request, FakeStream())

    def test_timeout_raises(self):
        def responder(handler, body):
            time.sleep(0.6)
            _write_json(handler, {"choices": [{"message": {"content": "late"}}]})

        with _MockServer(responder) as server:
            request = make_request(server.endpoint, timeout_seconds=0.1)
            started = time.monotonic()
            with self.assertRaises(RuntimeError):
                OpenAIProvider().execute(request, FakeStream())
            self.assertLess(time.monotonic() - started, 3.0)

    def test_missing_endpoint_raises(self):
        with self.assertRaises(RuntimeError):
            OpenAIProvider().execute({"model": "m"}, FakeStream())


class StreamTests(unittest.TestCase):
    """流式端到端。"""

    STREAM_CHUNKS = [
        {"choices": [{"delta": {"role": "assistant", "reasoning_content": "想"}}]},
        {"choices": [{"delta": {"reasoning_content": "一想"}}]},
        {"choices": [{"delta": {"content": "你好"}}]},
        {"choices": [{"delta": {"content": "！"}}]},
        {
            "choices": [
                {
                    "delta": {
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
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
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
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        },
    ]

    def test_request_and_response(self):
        with _MockServer(
            lambda handler, body: _write_sse(handler, self.STREAM_CHUNKS)
        ) as server:
            request = make_request(
                server.endpoint, stream=True, reasoning="none", timeout_seconds=5
            )
            stream = FakeStream()
            result = OpenAIProvider().execute(request, stream)
            records = server.requests
            self.assertEqual(server.errors, [])

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
        self.assertEqual(stream.content_chunks, ["你好", "！"])
        self.assertEqual(stream.reasoning_chunks, ["想", "一想"])

        self.assertEqual(len(records), 1)
        self.assertEqual(
            _header(records[0]["headers"], "Accept"), "text/event-stream"
        )
        body = records[0]["body"]
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertNotIn("reasoning_effort", body)

    def test_malformed_chunk_and_comments_skipped(self):
        def responder(handler, body):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(b": keep-alive\n\n")
            handler.wfile.write(b"event: message\n")
            handler.wfile.write(b"data: {this is not json}\n\n")
            handler.wfile.write(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            )
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.write(b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n\n')
            handler.wfile.flush()

        with _MockServer(responder) as server:
            result = OpenAIProvider().execute(
                make_request(server.endpoint, stream=True), FakeStream()
            )
        self.assertEqual(result["content"], "ok")
        self.assertEqual(server.errors, [])

    def test_stream_abort_returns_partial_content(self):
        chunks = [
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {"content": "B"}}]},
            {"choices": [{"delta": {"content": "C"}, "finish_reason": "stop"}]},
        ]
        with _MockServer(
            lambda handler, body: _write_sse(handler, chunks, delay=0.02)
        ) as server:
            stream = FakeStream(stop_after_deltas=1)
            result = OpenAIProvider().execute(
                make_request(server.endpoint, stream=True), stream
            )
        self.assertEqual(result["content"], "A")
        self.assertEqual(stream.content_chunks, ["A"])
        # 中止后不再读取 finish_reason。
        self.assertEqual(result["finish_reason"], "")

    def test_stream_error_chunk_raises(self):
        chunks = [
            {"choices": [{"delta": {"content": "开始"}}]},
            {"error": {"message": "server overloaded"}},
        ]
        with _MockServer(lambda handler, body: _write_sse(handler, chunks)) as server:
            with self.assertRaises(RuntimeError) as caught:
                OpenAIProvider().execute(
                    make_request(server.endpoint, stream=True), FakeStream()
                )
        self.assertIn("server overloaded", str(caught.exception))

    def test_stream_http_error_raises(self):
        with _MockServer(
            lambda handler, body: _write_json(
                handler, {"error": {"message": "bad request"}}, status=400
            )
        ) as server:
            with self.assertRaises(RuntimeError) as caught:
                OpenAIProvider().execute(
                    make_request(server.endpoint, stream=True), FakeStream()
                )
        self.assertIn("400", str(caught.exception))

    def test_stream_without_usage_keeps_zero_tokens(self):
        chunks = [{"choices": [{"delta": {"content": "hi"}}]}]
        with _MockServer(lambda handler, body: _write_sse(handler, chunks)) as server:
            result = OpenAIProvider().execute(
                make_request(server.endpoint, stream=True), FakeStream()
            )
        self.assertEqual(result["content"], "hi")
        self.assertEqual(result["prompt_tokens"], 0)
        self.assertEqual(result["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
