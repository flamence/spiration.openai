"""HTTP 传输层：用标准库 urllib 与 OpenAI Chat Completions 兼容端点通信。

本模块只关心「请求怎么发、字节流怎么读、失败怎么描述」，不解释请求/响应
的业务语义（那部分在 ``wire`` / ``stream`` / ``provider`` 中）。
"""

import http.client
import json
import urllib.error
import urllib.request

#: 请求未指定（或给出非法值）时使用的超时秒数，与宿主默认值保持一致。
DEFAULT_TIMEOUT_SECONDS = 120.0

#: 错误响应体在异常消息中最多保留的字符数（足够诊断即可）。
ERROR_BODY_PREVIEW = 300

_STREAM_ACCEPT = "text/event-stream"


def chat_completions_url(endpoint):
    """把宿主配置的 endpoint 归一化为 Chat Completions 的完整 URL。

    - 去掉尾部多余的 ``/``；
    - 已经以 ``/chat/completions`` 结尾时原样使用，否则追加该路径。
    """
    url = str(endpoint or "").strip().rstrip("/")
    if not url:
        raise RuntimeError("endpoint 为空，无法发送 OpenAI 请求")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    return url


def request_headers(api_key, stream):
    """构造请求头；``api_key`` 为空时不携带鉴权头。"""
    headers = {"Content-Type": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    if stream:
        headers["Accept"] = _STREAM_ACCEPT
    return headers


def normalize_timeout(timeout_seconds):
    """把宿主给超时值归一化为正浮点数；非法值回退到默认值。"""
    if isinstance(timeout_seconds, bool):
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return timeout if timeout > 0 else DEFAULT_TIMEOUT_SECONDS


def open_chat_completions(endpoint, payload, api_key, timeout_seconds, stream):
    """POST 一次 Chat Completions 请求并返回可逐行读取的响应对象。

    超时同时作用于连接与后续的读取操作（socket 级超时）。调用方负责关闭
    返回的响应对象。

    失败时抛出 :class:`RuntimeError`，消息中包含状态码与响应体摘要。
    """
    url = chat_completions_url(endpoint)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers(api_key, stream),
        method="POST",
    )
    timeout = normalize_timeout(timeout_seconds)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as err:
        raise RuntimeError(_http_error_message(url, err)) from err
    except (urllib.error.URLError, OSError, ValueError) as err:
        raise RuntimeError(f"OpenAI 请求失败（{url}）：{_reason_text(err)}") from err


def read_all(response):
    """读取非流式响应的完整字节内容；读取失败统一转换为 RuntimeError。"""
    try:
        return response.read()
    except (OSError, http.client.HTTPException) as err:
        raise RuntimeError(f"读取 OpenAI 响应失败：{err}") from err


def iter_response_lines(response):
    """按行读取响应字节；网络错误统一转换为 RuntimeError。"""
    try:
        for line in response:
            yield line
    except (OSError, http.client.HTTPException) as err:
        raise RuntimeError(f"读取 OpenAI 流式响应失败：{err}") from err


def _http_error_message(url, err):
    """把 HTTPError 整理成包含状态码与响应体摘要的消息。"""
    body = b""
    try:
        body = err.read() or b""
    except (OSError, http.client.HTTPException):
        body = b""
    finally:
        try:
            err.close()
        except OSError:
            pass
    preview = body.decode("utf-8", errors="replace").strip()[:ERROR_BODY_PREVIEW]
    detail = f"；响应体：{preview}" if preview else ""
    return f"OpenAI 请求失败：HTTP {err.code} {err.reason}（{url}）{detail}"


def _reason_text(err):
    """取异常中最具诊断价值的文本（URLError 的原因可能嵌套）。"""
    reason = getattr(err, "reason", None)
    if reason is not None:
        return str(reason)
    return str(err)
