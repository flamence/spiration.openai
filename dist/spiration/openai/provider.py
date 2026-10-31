"""OpenAIProvider：用官方 ``openai`` SDK 执行一次 Chat Completions 补全。

``execute`` 在宿主工作线程被调用。本类只持有一个「按端点/密钥/超时缓存客户端」
的字典（由 :class:`threading.Lock` 保护），一次请求的全部状态（流式累积器）
都是调用内的局部变量，因此可以并发执行。
"""

import threading

import openai

from . import stream as stream_module
from . import wire

#: 请求未指定（或给出非法值）时使用的超时秒数，与宿主默认值保持一致。
DEFAULT_TIMEOUT_SECONDS = 120.0

#: HTTP 错误响应体在异常消息中最多保留的字符数（足够诊断即可）。
ERROR_BODY_PREVIEW = 300

#: 宿主未配置 api_key 时使用的占位密钥。
#: 官方 SDK 拒绝空密钥构造客户端，而部分本地兼容端点并不校验鉴权，
#: 占位值可让这些端点继续工作。
ANONYMOUS_API_KEY = "spiration-anonymous"


def normalize_timeout(timeout_seconds):
    """把宿主给的超时值归一化为正浮点数；非法值回退到默认值。"""
    if isinstance(timeout_seconds, bool):
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return timeout if timeout > 0 else DEFAULT_TIMEOUT_SECONDS


def normalize_endpoint(endpoint):
    """校验并归一化宿主配置的 endpoint（直接作为 SDK 的 ``base_url``）。"""
    url = str(endpoint or "").strip()
    if not url:
        raise RuntimeError("endpoint 为空，无法发送 OpenAI 请求")
    return url


def normalize_api_key(api_key):
    """归一化 api_key；未配置时返回占位值，避免 SDK 拒绝构造客户端。"""
    key = str(api_key or "").strip()
    return key or ANONYMOUS_API_KEY


class OpenAIProvider:
    """OpenAI Chat Completions 格式的补全 provider。

    注册名为 ``openai``；请求经官方 ``openai`` SDK 转发到 OpenAI Chat
    Completions 兼容端点，不读写宿主界面状态。
    """

    #: 注册到宿主的 provider 名称。
    name = "openai"

    def __init__(self, client_factory=None):
        """构造 provider。

        :param client_factory: 可选的客户端工厂（默认 ``openai.OpenAI``），
                               测试可注入桩对象以完全离线。
        """
        self._client_factory = (
            client_factory if client_factory is not None else openai.OpenAI
        )
        self._clients = {}
        self._clients_lock = threading.Lock()

    def reasoning_levels(self):
        """本 provider 支持的思考挡位（按界面显示顺序）。"""
        return list(wire.REASONING_LEVELS)

    def execute(self, request, stream):
        """执行一次补全并返回宿主响应结构。

        :param request: 宿主补全请求（dict，见实现契约）。
        :param stream: 加载器创建的流式回调对象（delta/reasoning_delta/should_stop）。
        """
        if not isinstance(request, dict):
            raise RuntimeError("补全请求必须是 JSON 对象")

        endpoint = normalize_endpoint(request.get("endpoint"))
        kwargs = wire.build_request(request)
        try:
            client = self._client(
                endpoint,
                normalize_api_key(request.get("api_key")),
                normalize_timeout(request.get("timeout_seconds")),
            )
            if kwargs.get("stream"):
                return _execute_stream(client, kwargs, stream)
            return _execute_once(client, kwargs)
        except openai.APIStatusError as err:
            raise RuntimeError(_status_error_message(err)) from err
        except openai.APITimeoutError as err:
            raise RuntimeError(f"OpenAI 请求超时：{err}") from err
        except openai.APIConnectionError as err:
            raise RuntimeError(f"OpenAI 连接失败：{err}") from err
        except openai.APIError as err:
            raise RuntimeError(f"OpenAI 请求失败：{err}") from err
        except openai.OpenAIError as err:
            # 客户端构造阶段的错误（如缺少凭据）也统一为宿主可读的 RuntimeError。
            raise RuntimeError(f"OpenAI 客户端错误：{err}") from err

    def _client(self, endpoint, api_key, timeout):
        """按 (endpoint, api_key, timeout) 取缓存客户端，没有则创建一个。"""
        key = (endpoint, api_key, timeout)
        with self._clients_lock:
            client = self._clients.get(key)
            if client is None:
                client = self._client_factory(
                    api_key=api_key,
                    base_url=endpoint,
                    timeout=timeout,
                    max_retries=0,
                )
                self._clients[key] = client
            return client


def _execute_once(client, kwargs):
    """非流式：一次性请求并解析响应。"""
    return wire.parse_response(client.chat.completions.create(**kwargs))


def _execute_stream(client, kwargs, host_stream):
    """流式：逐块读取 SDK chunk，上报增量并累积完整响应。"""
    assembler = stream_module.StreamAssembler()
    sdk_stream = client.chat.completions.create(**kwargs)
    try:
        for chunk in sdk_stream:
            delta = assembler.feed(chunk)
            if delta.content:
                host_stream.delta(delta.content)
            if delta.reasoning:
                host_stream.reasoning_delta(delta.reasoning)

            # 每个 chunk 处理完立即检查用户是否请求中止，尽快停止读取。
            if host_stream.should_stop():
                break
    finally:
        _close_quietly(sdk_stream)
    return assembler.result()


def _close_quietly(sdk_stream):
    """尽量关闭 SDK 流；关闭阶段异常不应掩盖正常结果或原始异常。"""
    close = getattr(sdk_stream, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - 关闭失败不影响已累积的响应
        pass


def _status_error_message(err):
    """整理 ``APIStatusError``：状态码 + 响应体摘要（取不到时回退到异常文本）。"""
    status = getattr(err, "status_code", None)
    preview = _error_preview(err)
    if status is None:
        return f"OpenAI 请求失败：{preview or err}"
    if preview:
        return f"OpenAI 请求失败：HTTP {status}；响应体：{preview}"
    return f"OpenAI 请求失败：HTTP {status}：{err}"


def _error_preview(err):
    """尽量取错误响应体里的可读摘要（截断到 :data:`ERROR_BODY_PREVIEW`）。"""
    response = getattr(err, "response", None)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()[:ERROR_BODY_PREVIEW]
    body = getattr(err, "body", None)
    if body is not None:
        return str(body)[:ERROR_BODY_PREVIEW]
    return ""
