"""测试共享辅助：导入路径、假宿主对象、假 openai SDK 对象与请求构造。

假 SDK 对象模拟 openai 3.37.0 的实测行为：

- 模型对象把服务端扩展字段（如 ``reasoning_content``）放进 ``model_extra``，
  同时也可通过 ``getattr`` 读取；
- 流式响应是可迭代对象，并提供 ``close()``；
- SDK 异常用真实的 ``openai`` 异常类构造，响应对象用 ``SimpleNamespace`` 模拟。

测试前需要让 Python 能找到 openai 3.37.0（例如设置
``PYTHONPATH=<repo>/.tmp/sdk``），但所有测试都不联网。
"""

import os
import sys
from types import SimpleNamespace

import openai

# dist/ 是拓展运行时由加载器加入 sys.path 的目录，这里为测试显式加入。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(ROOT_DIR, "dist")
if DIST_DIR not in sys.path:
    sys.path.insert(0, DIST_DIR)

#: 测试请求使用的端点（不会真的发起网络请求：provider 注入假 client）。
DEFAULT_ENDPOINT = "https://api.example.invalid/v1"


class FakeStream:
    """模拟加载器注入的 stream 对象，记录增量并支持模拟用户中止。"""

    def __init__(self, stop_after_deltas=None):
        """``stop_after_deltas``：收到指定数量的正文增量后开始报告中止。"""
        self.content_chunks = []
        self.reasoning_chunks = []
        self._stop_after = stop_after_deltas

    def delta(self, text):
        """正文增量回调。"""
        self.content_chunks.append(text)

    def reasoning_delta(self, text):
        """思考过程增量回调。"""
        self.reasoning_chunks.append(text)

    def should_stop(self):
        """用户是否已请求中止（测试用：按已收正文增量数模拟）。"""
        if self._stop_after is None:
            return False
        return len(self.content_chunks) >= self._stop_after

    @property
    def content(self):
        """已收到的正文增量拼接结果。"""
        return "".join(self.content_chunks)

    @property
    def reasoning(self):
        """已收到的思考过程增量拼接结果。"""
        return "".join(self.reasoning_chunks)


def fake_model(**fields):
    """构造模拟 SDK 模型对象的 :class:`types.SimpleNamespace`。

    ``_extra`` 中的键写入 ``model_extra``（模拟服务端扩展字段），其余字段
    作为普通属性；未提供的字段用 ``getattr(..., default)`` 读取。
    """
    extra = fields.pop("_extra", {})
    obj = SimpleNamespace(model_extra=dict(extra or {}))
    for name, value in fields.items():
        setattr(obj, name, value)
    return obj


def make_usage(prompt_tokens=None, completion_tokens=None):
    """构造模拟 ``CompletionUsage`` 的假对象。"""
    return fake_model(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


def make_delta(content=None, reasoning_content=None, reasoning=None, tool_calls=None):
    """构造模拟 ``ChoiceDelta`` 的假对象；思考字段放在 ``model_extra`` 中。"""
    extra = {}
    if reasoning_content is not None:
        extra["reasoning_content"] = reasoning_content
    if reasoning is not None:
        extra["reasoning"] = reasoning
    return fake_model(content=content, tool_calls=tool_calls, _extra=extra)


def make_tool_call(index=0, id=None, name=None, arguments=None):
    """构造模拟 ``ChoiceDeltaToolCall`` 的假对象（function 可能为 None）。"""
    function = fake_model(name=name, arguments=arguments)
    return fake_model(index=index, id=id, function=function, type="function")


def make_message(content=None, reasoning_content=None, reasoning=None, tool_calls=None):
    """构造模拟 ``ChatCompletionMessage`` 的假对象；思考字段放进 ``model_extra``。"""
    extra = {}
    if reasoning_content is not None:
        extra["reasoning_content"] = reasoning_content
    if reasoning is not None:
        extra["reasoning"] = reasoning
    return fake_model(content=content, tool_calls=tool_calls, _extra=extra)


def make_chunk(delta=None, finish_reason=None, usage=None):
    """构造一个 ``ChatCompletionChunk`` 假对象。"""
    if delta is None:
        delta = make_delta()
    choice = fake_model(delta=delta, finish_reason=finish_reason)
    return fake_model(choices=[choice], usage=usage)


def make_usage_chunk(prompt_tokens=None, completion_tokens=None):
    """构造只有 usage、``choices`` 为空的最后一个流式 chunk。"""
    return fake_model(
        choices=[],
        usage=make_usage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def make_completion(message=None, finish_reason="stop", usage=None):
    """构造非流式 ``ChatCompletion`` 假对象。"""
    if message is None:
        message = make_message()
    choice = fake_model(message=message, finish_reason=finish_reason)
    return fake_model(choices=[choice], usage=usage)


class FakeSdkStream:
    """模拟 SDK 流式响应：可迭代、可 ``close()``、可在迭代末尾抛错。"""

    def __init__(self, chunks, error=None):
        self._chunks = list(chunks)
        self._error = error
        self.closed = False
        self.consumed = 0

    def __iter__(self):
        for chunk in self._chunks:
            self.consumed += 1
            yield chunk
        if self._error is not None:
            raise self._error

    def close(self):
        """记录关闭动作（provider 在中止/异常路径应调用它）。"""
        self.closed = True


class FakeClient:
    """模拟 ``openai.OpenAI``：记录 ``create`` 的关键字参数并返回预定结果。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeClientFactory:
    """模拟 ``openai.OpenAI`` 构造函数：记录参数并分发 fake client。

    传多个 client 时按顺序消费；只传一个时总是返回同一个（用于缓存复用测试）。
    """

    def __init__(self, clients):
        if not isinstance(clients, (list, tuple)):
            clients = [clients]
        self._clients = list(clients)
        self.init_kwargs = []

    def __call__(self, **kwargs):
        self.init_kwargs.append(dict(kwargs))
        if len(self._clients) > 1:
            return self._clients.pop(0)
        return self._clients[0]


def make_status_error(status_code=401, text=None, message=None, body=None):
    """用真实的 ``openai.APIStatusError`` 构造 HTTP 状态错误。

    :param text: 模拟的 ``e.response.text``；``None`` 时用默认错误 JSON。
    :param body: 模拟的 ``e.body``；``None`` 时用默认错误对象。
    """
    if text is None:
        text = '{"error": {"message": "invalid api key"}}'
    response = SimpleNamespace(
        status_code=status_code, headers={}, text=text, request=None
    )
    if body is None:
        body = {"error": {"message": "invalid api key"}}
    return openai.APIStatusError(
        message or f"Error code: {status_code}",
        response=response,
        body=body,
    )


def make_connection_error(message="Connection error."):
    """构造真实的 ``openai.APIConnectionError``。"""
    return openai.APIConnectionError(message=message, request=None)


def make_timeout_error():
    """构造真实的 ``openai.APITimeoutError``。"""
    return openai.APITimeoutError(request=None)


def make_request(endpoint=DEFAULT_ENDPOINT, **overrides):
    """构造一份宿主补全请求 dict，``overrides`` 覆盖默认字段。"""
    request = {
        "model": "test-model",
        "endpoint": endpoint,
        "api_key": "sk-test",
        "timeout_seconds": 5,
        "max_tokens": 0,
        "temperature": 0.2,
        "stream": False,
        "reasoning": "none",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
    }
    request.update(overrides)
    return request
