"""``stream`` 模块单元测试：SSE 解析与增量累积。"""

import unittest

import support  # noqa: F401 - 导入即把 dist/ 加入 sys.path

from spiration_openai.stream import SSE_DONE, Delta, StreamAssembler, iter_sse_payloads


class IterSsePayloadTests(unittest.TestCase):
    """SSE 行 → data 载荷。"""

    def test_basic_payloads(self):
        lines = [
            b'data: {"a": 1}\n',
            b"\n",
            b": keep-alive\n",
            b"event: message\n",
            b"id: 1\n",
            b'data:{"b": 2}\r\n',
            b"data:   \n",
        ]
        self.assertEqual(
            list(iter_sse_payloads(lines)),
            ['{"a": 1}', '{"b": 2}'],
        )

    def test_done_stops_iteration(self):
        lines = [
            b'data: {"a": 1}\n',
            f"data: {SSE_DONE}\n",
            b'data: {"never": true}\n',
        ]
        self.assertEqual(list(iter_sse_payloads(lines)), ['{"a": 1}'])

    def test_string_lines(self):
        self.assertEqual(
            list(iter_sse_payloads(['data: {"a": 1}', "data: [DONE]"])),
            ['{"a": 1}'],
        )

    def test_ignores_content_lines_without_data_prefix(self):
        self.assertEqual(list(iter_sse_payloads([b"\n", b"foo\n"])), [])


class StreamAssemblerTests(unittest.TestCase):
    """流式 chunk 累积。"""

    def test_content_and_reasoning_deltas(self):
        assembler = StreamAssembler()
        delta = assembler.feed(
            {"choices": [{"delta": {"reasoning_content": "想", "content": "你"}}]}
        )
        self.assertEqual(delta, Delta(content="你", reasoning="想"))

        delta = assembler.feed(
            {"choices": [{"delta": {"reasoning_content": "一想", "content": "好"}}]}
        )
        self.assertEqual(delta, Delta(content="好", reasoning="一想"))

        result = assembler.result()
        self.assertEqual(result["content"], "你好")
        self.assertEqual(result["reasoning_content"], "想一想")
        self.assertEqual(result["tool_calls"], [])

    def test_reasoning_alias(self):
        assembler = StreamAssembler()
        delta = assembler.feed({"choices": [{"delta": {"reasoning": "想"}}]})
        self.assertEqual(delta.reasoning, "想")
        self.assertEqual(assembler.result()["reasoning_content"], "想")

    def test_reasoning_content_preferred_over_alias(self):
        assembler = StreamAssembler()
        delta = assembler.feed(
            {"choices": [{"delta": {"reasoning_content": "A", "reasoning": "B"}}]}
        )
        self.assertEqual(delta.reasoning, "A")
        self.assertEqual(assembler.result()["reasoning_content"], "A")

    def test_tool_call_accumulation(self):
        assembler = StreamAssembler()
        assembler.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read_", "arguments": '{"pa'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        assembler.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "file", "arguments": 'th":"a"}'},
                                },
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "function": {"name": "sleep", "arguments": "{}"},
                                },
                            ]
                        }
                    }
                ]
            }
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [
                {
                    "id": "call_1",
                    "function_name": "read_file",
                    "arguments": '{"path":"a"}',
                },
                {"id": "call_2", "function_name": "sleep", "arguments": "{}"},
            ],
        )

    def test_tool_call_defaults_and_empty_arguments(self):
        assembler = StreamAssembler()
        assembler.feed(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"function": {"name": "noop"}}]}}
                ]
            }
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [{"id": "", "function_name": "noop", "arguments": ""}],
        )

    def test_finish_reason_keeps_last_non_empty(self):
        assembler = StreamAssembler()
        assembler.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        assembler.feed({"choices": [{"delta": {}, "finish_reason": None}]})
        assembler.feed({"choices": [{"delta": {}, "finish_reason": "length"}]})
        self.assertEqual(assembler.result()["finish_reason"], "length")

    def test_usage_chunk_without_choices(self):
        assembler = StreamAssembler()
        assembler.feed({"choices": [], "usage": {"prompt_tokens": 12}})
        assembler.feed(
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}}
        )
        self.assertEqual(assembler.result()["prompt_tokens"], 12)
        self.assertEqual(assembler.result()["completion_tokens"], 7)

    def test_usage_ignores_missing_fields(self):
        assembler = StreamAssembler()
        assembler.feed({"usage": {"prompt_tokens": 5}})
        assembler.feed({"usage": None})
        assembler.feed({"usage": {"completion_tokens": 3}})
        self.assertEqual(assembler.result()["prompt_tokens"], 5)
        self.assertEqual(assembler.result()["completion_tokens"], 3)

    def test_empty_result_defaults(self):
        result = StreamAssembler().result()
        self.assertEqual(
            result,
            {
                "content": None,
                "reasoning_content": None,
                "tool_calls": [],
                "finish_reason": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )

    def test_non_dict_chunk_ignored(self):
        assembler = StreamAssembler()
        self.assertEqual(assembler.feed(["not", "a", "dict"]), Delta("", ""))
        self.assertEqual(assembler.feed(None), Delta("", ""))
        self.assertIsNone(assembler.result()["content"])

    def test_missing_delta_or_choices(self):
        assembler = StreamAssembler()
        self.assertEqual(assembler.feed({"choices": [{}]}), Delta("", ""))
        self.assertEqual(assembler.feed({"choices": "bad"}), Delta("", ""))
        self.assertIsNone(assembler.result()["content"])


if __name__ == "__main__":
    unittest.main()
