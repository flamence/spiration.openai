"""``wire`` 与 ``transport`` 纯函数单元测试（不联网）。"""

import unittest

import support  # noqa: F401 - 导入即把 dist/ 加入 sys.path

import spiration_openai.transport as transport
import spiration_openai.wire as wire


class TransportUrlTests(unittest.TestCase):
    """endpoint → Chat Completions URL 的归一化。"""

    def test_append_path(self):
        self.assertEqual(
            transport.chat_completions_url("https://api.deepseek.com"),
            "https://api.deepseek.com/chat/completions",
        )

    def test_strip_trailing_slash(self):
        self.assertEqual(
            transport.chat_completions_url("https://api.deepseek.com/"),
            "https://api.deepseek.com/chat/completions",
        )

    def test_keep_existing_suffix(self):
        self.assertEqual(
            transport.chat_completions_url("https://host/v1/chat/completions"),
            "https://host/v1/chat/completions",
        )
        self.assertEqual(
            transport.chat_completions_url("https://host/v1/chat/completions/"),
            "https://host/v1/chat/completions",
        )

    def test_empty_endpoint(self):
        with self.assertRaises(RuntimeError):
            transport.chat_completions_url("")
        with self.assertRaises(RuntimeError):
            transport.chat_completions_url(None)


class TransportHeaderTests(unittest.TestCase):
    """请求头构造。"""

    def test_json_and_authorization(self):
        headers = transport.request_headers("sk-abc", stream=False)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Authorization"], "Bearer sk-abc")
        self.assertNotIn("Accept", headers)

    def test_stream_accept(self):
        headers = transport.request_headers("sk-abc", stream=True)
        self.assertEqual(headers["Accept"], "text/event-stream")

    def test_empty_api_key_has_no_authorization(self):
        for key in (None, "", "   "):
            headers = transport.request_headers(key, stream=False)
            self.assertNotIn("Authorization", headers)


class TransportTimeoutTests(unittest.TestCase):
    """超时值归一化。"""

    def test_valid(self):
        self.assertEqual(transport.normalize_timeout(30), 30.0)
        self.assertEqual(transport.normalize_timeout(0.5), 0.5)

    def test_fallback(self):
        for value in (None, 0, -1, "bad", True):
            self.assertEqual(
                transport.normalize_timeout(value),
                transport.DEFAULT_TIMEOUT_SECONDS,
            )


class BuildRequestTests(unittest.TestCase):
    """宿主请求 → OpenAI 请求体。"""

    def test_minimal_request(self):
        body = wire.build_request(
            {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
            }
        )
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], 0.5)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("tools", body)
        self.assertNotIn("stream_options", body)
        self.assertNotIn("reasoning_effort", body)

    def test_max_tokens_only_when_positive(self):
        for value in (0, None, -5, "x"):
            body = wire.build_request({"model": "m", "max_tokens": value})
            self.assertNotIn("max_tokens", body)
        body = wire.build_request({"model": "m", "max_tokens": 128})
        self.assertEqual(body["max_tokens"], 128)

    def test_stream_options(self):
        body = wire.build_request({"model": "m", "stream": True})
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_tools_conversion(self):
        body = wire.build_request(
            {
                "model": "m",
                "tools": [
                    {
                        "name": "read_file",
                        "description": "读取文件",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    {"name": "no_params"},
                ],
            }
        )
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "no_params",
                        "description": "",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )

    def test_empty_tools_omitted(self):
        body = wire.build_request({"model": "m", "tools": []})
        self.assertNotIn("tools", body)

    def test_reasoning_effort_mapping(self):
        cases = {
            None: None,
            "none": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
            "unknown": "medium",
        }
        for reasoning, expected in cases.items():
            body = wire.build_request({"model": "m", "reasoning": reasoning})
            if expected is None:
                self.assertNotIn("reasoning_effort", body, reasoning)
            else:
                self.assertEqual(body["reasoning_effort"], expected, reasoning)


class ConvertMessageTests(unittest.TestCase):
    """宿主消息 → OpenAI 消息。"""

    def test_system_and_user(self):
        self.assertEqual(
            wire.convert_message({"role": "system", "content": "规则"}),
            {"role": "system", "content": "规则"},
        )
        self.assertEqual(
            wire.convert_message({"role": "user", "content": "问题"}),
            {"role": "user", "content": "问题"},
        )

    def test_assistant_drops_reasoning_content(self):
        converted = wire.convert_message(
            {
                "role": "assistant",
                "content": "答案",
                "reasoning_content": "思考",
                "name": "assistant_name",
            }
        )
        self.assertEqual(converted, {"role": "assistant", "content": "答案"})
        self.assertNotIn("reasoning_content", converted)
        self.assertNotIn("name", converted)

    def test_assistant_with_tool_calls(self):
        converted = wire.convert_message(
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "思考",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function_name": "read_file",
                        "arguments": '{"path":"a"}',
                    }
                ],
            }
        )
        self.assertEqual(
            converted,
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
        )

    def test_assistant_empty_tool_calls_omitted(self):
        converted = wire.convert_message(
            {"role": "assistant", "content": "答案", "tool_calls": []}
        )
        self.assertNotIn("tool_calls", converted)

    def test_tool_message(self):
        self.assertEqual(
            wire.convert_message(
                {"role": "tool", "tool_call_id": "call_1", "content": "结果"}
            ),
            {"role": "tool", "tool_call_id": "call_1", "content": "结果"},
        )


class ParseResponseTests(unittest.TestCase):
    """非流式响应解析。"""

    def test_full_response(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "你好",
                        "reasoning_content": "先想一想",
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
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
        self.assertEqual(
            wire.parse_response(payload),
            {
                "content": "你好",
                "reasoning_content": "先想一想",
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

    def test_reasoning_alias(self):
        payload = {
            "choices": [{"message": {"content": "答", "reasoning": "想一想"}}]
        }
        response = wire.parse_response(payload)
        self.assertEqual(response["reasoning_content"], "想一想")

    def test_missing_fields(self):
        response = wire.parse_response({"choices": [{"message": {}}]})
        self.assertEqual(
            response,
            {
                "content": None,
                "reasoning_content": None,
                "tool_calls": [],
                "finish_reason": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )

    def test_empty_choices(self):
        response = wire.parse_response({"choices": []})
        self.assertIsNone(response["content"])
        self.assertEqual(response["finish_reason"], "")

    def test_empty_content_normalized_to_none(self):
        payload = {"choices": [{"message": {"content": "", "reasoning_content": ""}}]}
        response = wire.parse_response(payload)
        self.assertIsNone(response["content"])
        self.assertIsNone(response["reasoning_content"])


class ExtractErrorTests(unittest.TestCase):
    """服务端错误对象提取。"""

    def test_no_error(self):
        self.assertIsNone(wire.extract_error({"choices": []}))
        self.assertIsNone(wire.extract_error({"error": None}))
        self.assertIsNone(wire.extract_error("not a dict"))

    def test_dict_error(self):
        message = wire.extract_error(
            {"error": {"message": "invalid api key", "code": "invalid_api_key"}}
        )
        self.assertIn("invalid api key", message)
        self.assertIn("invalid_api_key", message)

    def test_string_error(self):
        self.assertIn("quota", wire.extract_error({"error": "quota exceeded"}))


class TokenCountTests(unittest.TestCase):
    """token 计数归一化。"""

    def test_values(self):
        self.assertEqual(wire.token_count(12), 12)
        self.assertEqual(wire.token_count(3.9), 3)
        for value in (None, "12", True, -1.5, 0):
            self.assertEqual(wire.token_count(value), 0, value)


if __name__ == "__main__":
    unittest.main()
