"""``stream`` 模块单元测试：SDK chunk 累积器（正文/思考/工具调用/usage/finish）。"""

import unittest

import support  # noqa: F401 - 导入即把 dist/ 加入 sys.path

from support import (
    fake_model,
    make_chunk,
    make_delta,
    make_tool_call,
    make_usage_chunk,
)
from spiration_openai.stream import Delta, StreamAssembler


class ContentAndReasoningTests(unittest.TestCase):
    """正文与思考过程增量。"""

    def test_content_deltas(self):
        assembler = StreamAssembler()
        self.assertEqual(
            assembler.feed(make_chunk(make_delta(content="你"))),
            Delta(content="你", reasoning=""),
        )
        self.assertEqual(
            assembler.feed(make_chunk(make_delta(content="好"))),
            Delta(content="好", reasoning=""),
        )
        result = assembler.result()
        self.assertEqual(result["content"], "你好")
        self.assertIsNone(result["reasoning_content"])

    def test_reasoning_model_extra(self):
        assembler = StreamAssembler()
        self.assertEqual(
            assembler.feed(make_chunk(make_delta(reasoning_content="想"))),
            Delta(content="", reasoning="想"),
        )
        self.assertEqual(
            assembler.feed(make_chunk(make_delta(reasoning_content="一想"))),
            Delta(content="", reasoning="一想"),
        )
        self.assertEqual(assembler.result()["reasoning_content"], "想一想")

    def test_reasoning_direct_attribute(self):
        assembler = StreamAssembler()
        delta = assembler.feed(
            make_chunk(fake_model(content=None, reasoning_content="直接属性"))
        )
        self.assertEqual(delta.reasoning, "直接属性")
        self.assertEqual(assembler.result()["reasoning_content"], "直接属性")

    def test_reasoning_alias(self):
        assembler = StreamAssembler()
        delta = assembler.feed(make_chunk(make_delta(reasoning="别名")))
        self.assertEqual(delta.reasoning, "别名")
        self.assertEqual(assembler.result()["reasoning_content"], "别名")

    def test_reasoning_content_preferred_over_alias(self):
        assembler = StreamAssembler()
        delta = assembler.feed(
            make_chunk(
                fake_model(
                    content=None, _extra={"reasoning_content": "A", "reasoning": "B"}
                )
            )
        )
        self.assertEqual(delta.reasoning, "A")
        self.assertEqual(assembler.result()["reasoning_content"], "A")

    def test_content_and_reasoning_in_same_chunk(self):
        assembler = StreamAssembler()
        self.assertEqual(
            assembler.feed(make_chunk(make_delta(content="你", reasoning_content="想"))),
            Delta(content="你", reasoning="想"),
        )

    def test_empty_string_content_not_reported(self):
        assembler = StreamAssembler()
        self.assertEqual(
            assembler.feed(make_chunk(make_delta(content=""))), Delta("", "")
        )
        self.assertIsNone(assembler.result()["content"])


class ToolCallTests(unittest.TestCase):
    """工具调用分片累积。"""

    def test_single_chunk_tool_call(self):
        assembler = StreamAssembler()
        assembler.feed(
            make_chunk(
                make_delta(
                    tool_calls=[
                        make_tool_call(0, "call_1", "read_file", '{"path":"a"}')
                    ]
                )
            )
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [
                {
                    "id": "call_1",
                    "function_name": "read_file",
                    "arguments": '{"path":"a"}',
                }
            ],
        )

    def test_fragments_accumulated(self):
        assembler = StreamAssembler()
        assembler.feed(
            make_chunk(
                make_delta(
                    tool_calls=[
                        make_tool_call(0, "call_1", "read_", '{"pa')
                    ]
                )
            )
        )
        assembler.feed(
            make_chunk(
                make_delta(tool_calls=[make_tool_call(0, None, "file", 'th":"a"}')])
            )
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [
                {
                    "id": "call_1",
                    "function_name": "read_file",
                    "arguments": '{"path":"a"}',
                }
            ],
        )

    def test_multiple_calls_sorted_by_index(self):
        assembler = StreamAssembler()
        assembler.feed(
            make_chunk(
                make_delta(
                    tool_calls=[
                        make_tool_call(1, "call_2", "sleep", "{}"),
                        make_tool_call(0, "call_1", "echo", '{"t":"x"}'),
                    ]
                )
            )
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [
                {"id": "call_1", "function_name": "echo", "arguments": '{"t":"x"}'},
                {"id": "call_2", "function_name": "sleep", "arguments": "{}"},
            ],
        )

    def test_missing_index_defaults_to_zero(self):
        assembler = StreamAssembler()
        assembler.feed(
            make_chunk(
                make_delta(
                    tool_calls=[
                        fake_model(
                            id="call_1",
                            function=fake_model(name="noop", arguments="{}"),
                        )
                    ]
                )
            )
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [{"id": "call_1", "function_name": "noop", "arguments": "{}"}],
        )

    def test_id_only_fragment_completed_later(self):
        """首个分片可能只有 id（function 为 None），后续分片再补齐名称与参数。"""
        assembler = StreamAssembler()
        assembler.feed(
            make_chunk(make_delta(tool_calls=[fake_model(index=0, id="call_1")]))
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [{"id": "call_1", "function_name": "", "arguments": ""}],
        )
        assembler.feed(
            make_chunk(
                make_delta(tool_calls=[make_tool_call(0, None, "noop", "{}")])
            )
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [{"id": "call_1", "function_name": "noop", "arguments": "{}"}],
        )

    def test_empty_arguments_kept_empty(self):
        assembler = StreamAssembler()
        assembler.feed(
            make_chunk(
                make_delta(
                    tool_calls=[make_tool_call(0, "call_1", "noop", "")]
                )
            )
        )
        self.assertEqual(
            assembler.result()["tool_calls"],
            [{"id": "call_1", "function_name": "noop", "arguments": ""}],
        )


class FinishAndUsageTests(unittest.TestCase):
    """finish_reason 与 token 用量。"""

    def test_finish_reason_keeps_last_non_empty(self):
        assembler = StreamAssembler()
        assembler.feed(make_chunk(finish_reason="stop"))
        assembler.feed(make_chunk(finish_reason=None))
        assembler.feed(make_chunk(finish_reason="length"))
        self.assertEqual(assembler.result()["finish_reason"], "length")

    def test_usage_chunk_without_choices(self):
        assembler = StreamAssembler()
        assembler.feed(make_usage_chunk(12, 7))
        self.assertEqual(assembler.result()["prompt_tokens"], 12)
        self.assertEqual(assembler.result()["completion_tokens"], 7)

    def test_usage_partial_fields_keep_previous(self):
        assembler = StreamAssembler()
        assembler.feed(make_usage_chunk(5))
        assembler.feed(make_chunk())
        assembler.feed(make_usage_chunk(completion_tokens=3))
        self.assertEqual(assembler.result()["prompt_tokens"], 5)
        self.assertEqual(assembler.result()["completion_tokens"], 3)

    def test_usage_none_ignored(self):
        assembler = StreamAssembler()
        assembler.feed(make_chunk(make_delta(content="hi"), usage=None))
        self.assertEqual(assembler.result()["content"], "hi")
        self.assertEqual(assembler.result()["prompt_tokens"], 0)


class DefensiveInputTests(unittest.TestCase):
    """异常/缺省 chunk 的健壮性。"""

    def test_empty_result_defaults(self):
        self.assertEqual(
            StreamAssembler().result(),
            {
                "content": None,
                "reasoning_content": None,
                "tool_calls": [],
                "finish_reason": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )

    def test_none_chunk_ignored(self):
        assembler = StreamAssembler()
        self.assertEqual(assembler.feed(None), Delta("", ""))
        self.assertIsNone(assembler.result()["content"])

    def test_chunk_without_choices(self):
        assembler = StreamAssembler()
        self.assertEqual(assembler.feed(fake_model(usage=None)), Delta("", ""))
        self.assertEqual(
            assembler.feed(fake_model(choices=[], usage=None)), Delta("", "")
        )

    def test_choice_without_delta(self):
        assembler = StreamAssembler()
        chunk = fake_model(choices=[fake_model(delta=None, finish_reason=None)])
        self.assertEqual(assembler.feed(chunk), Delta("", ""))

    def test_choice_without_usage_attribute(self):
        assembler = StreamAssembler()
        chunk = fake_model(choices=[fake_model(delta=make_delta(content="ok"))])
        self.assertEqual(assembler.feed(chunk), Delta("ok", ""))
        self.assertEqual(assembler.result()["content"], "ok")


if __name__ == "__main__":
    unittest.main()
