"""``wire`` 纯函数单元测试（不联网）：宿主请求 → SDK 参数，SDK 对象 → 宿主响应。"""

import unittest

import support  # noqa: F401 - 导入即把 dist/ 加入 sys.path

import spiration_openai.wire as wire
from support import (
    fake_model,
    make_completion,
    make_message,
    make_tool_call,
    make_usage,
)


class BuildRequestTests(unittest.TestCase):
    """宿主请求 → ``chat.completions.create`` 关键字参数。"""

    def test_minimal_request(self):
        kwargs = wire.build_request(
            {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
            }
        )
        self.assertEqual(
            kwargs,
            {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "temperature": 0.5,
            },
        )

    def test_max_completion_tokens_only_when_positive(self):
        for value in (0, None, -5, "x", True):
            kwargs = wire.build_request({"model": "m", "max_tokens": value})
            self.assertNotIn("max_completion_tokens", kwargs, value)
            self.assertNotIn("max_tokens", kwargs, value)

        for value, expected in ((128, 128), (3.9, 3), (1, 1)):
            kwargs = wire.build_request({"model": "m", "max_tokens": value})
            self.assertEqual(kwargs["max_completion_tokens"], expected, value)

    def test_stream_options(self):
        kwargs = wire.build_request({"model": "m", "stream": True})
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})

    def test_tools_conversion(self):
        kwargs = wire.build_request(
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
            kwargs["tools"],
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
        kwargs = wire.build_request({"model": "m", "tools": []})
        self.assertNotIn("tools", kwargs)

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
            kwargs = wire.build_request({"model": "m", "reasoning": reasoning})
            if expected is None:
                self.assertNotIn("reasoning_effort", kwargs, reasoning)
            else:
                self.assertEqual(kwargs["reasoning_effort"], expected, reasoning)

    def test_temperature_fallback(self):
        for value in (None, True, "hot", [], {}):
            kwargs = wire.build_request({"model": "m", "temperature": value})
            self.assertEqual(kwargs["temperature"], 0, value)
        for value in (0.7, 2, 0):
            kwargs = wire.build_request({"model": "m", "temperature": value})
            self.assertEqual(kwargs["temperature"], value)

    def test_invalid_messages_skipped(self):
        kwargs = wire.build_request(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    "not a dict",
                    None,
                    42,
                ],
            }
        )
        self.assertEqual(
            kwargs["messages"], [{"role": "user", "content": "hi"}]
        )


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

    def test_tool_message_without_id(self):
        self.assertEqual(
            wire.convert_message({"role": "tool", "content": "结果"}),
            {"role": "tool", "tool_call_id": "", "content": "结果"},
        )

    def test_non_string_arguments_normalized(self):
        converted = wire.convert_assistant_tool_calls(
            [{"id": "call_1", "function_name": "f", "arguments": {"path": "a"}}]
        )
        self.assertEqual(converted[0]["function"]["arguments"], "")

    def test_non_dict_tool_calls_skipped(self):
        self.assertEqual(
            wire.convert_assistant_tool_calls([None, "bad", {"id": "call_1"}]),
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            ],
        )


class ReasoningTextTests(unittest.TestCase):
    """服务端扩展字段的读取（直接属性 + ``model_extra`` 两种形态）。"""

    def test_direct_attribute(self):
        self.assertEqual(
            wire.reasoning_text(fake_model(reasoning_content="直接属性")),
            "直接属性",
        )

    def test_model_extra_attribute(self):
        self.assertEqual(
            wire.reasoning_text(fake_model(_extra={"reasoning_content": "扩展字段"})),
            "扩展字段",
        )

    def test_reasoning_alias(self):
        self.assertEqual(
            wire.reasoning_text(fake_model(reasoning="别名")), "别名"
        )
        self.assertEqual(
            wire.reasoning_text(fake_model(_extra={"reasoning": "扩展别名"})),
            "扩展别名",
        )

    def test_reasoning_content_preferred_over_alias(self):
        self.assertEqual(
            wire.reasoning_text(fake_model(reasoning_content="A", reasoning="B")),
            "A",
        )
        self.assertEqual(
            wire.reasoning_text(
                fake_model(_extra={"reasoning_content": "A", "reasoning": "B"})
            ),
            "A",
        )

    def test_empty_values_normalized_to_none(self):
        for obj in (
            fake_model(reasoning_content=""),
            fake_model(_extra={"reasoning_content": ""}),
            fake_model(reasoning_content=None),
            fake_model(),
            None,
        ):
            self.assertIsNone(wire.reasoning_text(obj), obj)

    def test_non_dict_model_extra_ignored(self):
        self.assertIsNone(wire.reasoning_text(fake_model(model_extra="bad")))


class ParseResponseTests(unittest.TestCase):
    """非流式响应解析。"""

    def test_full_response(self):
        completion = make_completion(
            message=make_message(
                content="你好",
                reasoning_content="先想一想",
                tool_calls=[
                    make_tool_call(0, "call_1", "read_file", '{"path":"a"}')
                ],
            ),
            finish_reason="tool_calls",
            usage=make_usage(12, 7),
        )
        self.assertEqual(
            wire.parse_response(completion),
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

    def test_reasoning_only_keeps_content_none(self):
        completion = make_completion(
            message=make_message(content=None, reasoning_content="只思考"),
            finish_reason="length",
            usage=make_usage(41, 32),
        )
        response = wire.parse_response(completion)
        self.assertIsNone(response["content"])
        self.assertEqual(response["reasoning_content"], "只思考")
        self.assertEqual(response["finish_reason"], "length")

    def test_reasoning_alias(self):
        completion = make_completion(
            message=make_message(content="答", reasoning="想一想")
        )
        self.assertEqual(
            wire.parse_response(completion)["reasoning_content"], "想一想"
        )

    def test_empty_content_normalized_to_none(self):
        completion = make_completion(
            message=make_message(content="", reasoning_content="")
        )
        response = wire.parse_response(completion)
        self.assertIsNone(response["content"])
        self.assertIsNone(response["reasoning_content"])

    def test_missing_fields(self):
        completion = fake_model(
            choices=[fake_model(message=fake_model(), finish_reason=None)],
            usage=None,
        )
        self.assertEqual(
            wire.parse_response(completion),
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
        response = wire.parse_response(fake_model(choices=[], usage=None))
        self.assertIsNone(response["content"])
        self.assertEqual(response["finish_reason"], "")

    def test_none_completion(self):
        self.assertEqual(
            wire.parse_response(None),
            {
                "content": None,
                "reasoning_content": None,
                "tool_calls": [],
                "finish_reason": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )

    def test_tool_call_without_function(self):
        completion = make_completion(
            message=make_message(tool_calls=[fake_model(id="call_2", function=None)])
        )
        self.assertEqual(
            wire.parse_response(completion)["tool_calls"],
            [{"id": "call_2", "function_name": "", "arguments": ""}],
        )

    def test_usage_invalid_values(self):
        completion = make_completion(
            usage=fake_model(prompt_tokens="12", completion_tokens=True)
        )
        response = wire.parse_response(completion)
        self.assertEqual(response["prompt_tokens"], 0)
        self.assertEqual(response["completion_tokens"], 0)


class TokenCountTests(unittest.TestCase):
    """token 计数归一化。"""

    def test_values(self):
        self.assertEqual(wire.token_count(12), 12)
        self.assertEqual(wire.token_count(3.9), 3)
        for value in (None, "12", True, -1.5, 0):
            self.assertEqual(wire.token_count(value), 0, value)


if __name__ == "__main__":
    unittest.main()
