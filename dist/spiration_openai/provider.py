"""OpenAIProvider：把宿主补全请求转发到 OpenAI Chat Completions 兼容端点。"""

import json

from . import stream as stream_module
from . import transport
from . import wire


class OpenAIProvider:
    """OpenAI Chat Completions 格式的补全 provider。

    注册名为 ``openai``；``execute`` 在宿主工作线程被调用，因此本类内部
    只使用标准库，不调用宿主 ``spiration`` 模块的界面能力。
    """

    #: 注册到宿主的 provider 名称。
    name = "openai"

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
        payload = wire.build_request(request)
        timeout = transport.normalize_timeout(request.get("timeout_seconds"))
        if payload.get("stream"):
            return self._execute_stream(request, payload, timeout, stream)
        return self._execute_once(request, payload, timeout)

    def _execute_once(self, request, payload, timeout):
        """非流式：一次性读取响应并解析。"""
        response = transport.open_chat_completions(
            request.get("endpoint"),
            payload,
            request.get("api_key"),
            timeout,
            stream=False,
        )
        try:
            body = transport.read_all(response)
        finally:
            response.close()

        try:
            document = json.loads(body.decode("utf-8", errors="replace"))
        except ValueError as err:
            preview = body.decode("utf-8", errors="replace").strip()[:200]
            raise RuntimeError(f"解析 OpenAI 响应失败：{preview}") from err

        error = wire.extract_error(document)
        if error:
            raise RuntimeError(error)
        return wire.parse_response(document)

    def _execute_stream(self, request, payload, timeout, stream):
        """流式：逐块读取 SSE，上报增量并累积完整响应。"""
        response = transport.open_chat_completions(
            request.get("endpoint"),
            payload,
            request.get("api_key"),
            timeout,
            stream=True,
        )
        assembler = stream_module.StreamAssembler()
        try:
            for data in stream_module.iter_sse_payloads(
                transport.iter_response_lines(response)
            ):
                chunk = _parse_chunk(data)
                if chunk is None:
                    continue

                error = wire.extract_error(chunk)
                if error:
                    raise RuntimeError(error)

                delta = assembler.feed(chunk)
                if delta.content:
                    stream.delta(delta.content)
                if delta.reasoning:
                    stream.reasoning_delta(delta.reasoning)

                # 每个 chunk 处理完立即检查用户是否请求中止，尽快停止读取。
                if stream.should_stop():
                    break
        finally:
            response.close()
        return assembler.result()


def _parse_chunk(payload):
    """解析单个 SSE data 载荷；非法 JSON 返回 None（按契约跳过该 chunk）。"""
    try:
        chunk = json.loads(payload)
    except ValueError:
        return None
    return chunk if isinstance(chunk, dict) else None
