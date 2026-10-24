"""宿主补全请求/响应与 OpenAI Chat Completions 线上格式之间的转换。

本模块不发起网络请求，纯函数便于单元测试；流式解析见 ``stream`` 模块。
"""

import json

#: 本 provider 声明的思考挡位（与宿主 reasoning_level 名称一致）。
REASONING_LEVELS = ("none", "low", "medium", "high")

#: 思考挡位 → OpenAI ``reasoning_effort``。
#: ``xhigh`` / ``max`` 虽未声明，但若被传入则降级为 ``high``。
_REASONING_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

_DEFAULT_TOOL_PARAMETERS = {"type": "object", "properties": {}}


def build_request(request):
    """把宿主补全请求（dict）转换为 OpenAI Chat Completions 请求体。"""
    request = request or {}
    stream = bool(request.get("stream"))
    body = {
        "model": request.get("model") or "",
        "messages": [
            convert_message(message)
            for message in request.get("messages") or []
            if isinstance(message, dict)
        ],
        "stream": stream,
        "temperature": _temperature(request.get("temperature")),
    }

    max_tokens = _positive_int(request.get("max_tokens"))
    if max_tokens:
        body["max_tokens"] = max_tokens

    tools = [
        convert_tool(tool)
        for tool in request.get("tools") or []
        if isinstance(tool, dict)
    ]
    if tools:
        body["tools"] = tools

    if stream:
        # 请求最后一个 chunk 携带 token 用量；不支持的端点会忽略该字段。
        body["stream_options"] = {"include_usage": True}

    effort = reasoning_effort(request.get("reasoning"))
    if effort:
        body["reasoning_effort"] = effort
    return body


def reasoning_effort(reasoning):
    """把宿主思考挡位映射为 ``reasoning_effort``；返回 None 表示不携带。"""
    level = str(reasoning or "").strip().lower()
    if level in ("", "none"):
        return None
    return _REASONING_EFFORT.get(level, "medium")


def convert_message(message):
    """把单条宿主消息转换为 OpenAI Chat Completions 消息。

    ``assistant`` 消息的 ``reasoning_content`` 不会发回服务端：不同兼容
    实现对该字段的处理并不一致。
    """
    role = message.get("role") or "user"
    content = message.get("content")

    if role == "assistant":
        converted = {"role": "assistant", "content": content}
        tool_calls = convert_assistant_tool_calls(message.get("tool_calls"))
        if tool_calls:
            converted["tool_calls"] = tool_calls
        return converted

    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.get("tool_call_id") or "",
            "content": content,
        }

    return {"role": role, "content": content}


def convert_assistant_tool_calls(tool_calls):
    """把宿主 assistant 消息中的工具调用转换为 OpenAI function 调用格式。"""
    converted = []
    for item in tool_calls or []:
        if not isinstance(item, dict):
            continue
        arguments = item.get("arguments")
        converted.append(
            {
                "id": item.get("id") or "",
                "type": "function",
                "function": {
                    "name": item.get("function_name") or "",
                    "arguments": arguments if isinstance(arguments, str) else "",
                },
            }
        )
    return converted


def convert_tool(tool):
    """把宿主工具定义转换为 OpenAI ``type=function`` 工具格式。"""
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        parameters = dict(_DEFAULT_TOOL_PARAMETERS)
    return {
        "type": "function",
        "function": {
            "name": tool.get("name") or "",
            "description": tool.get("description") or "",
            "parameters": parameters,
        },
    }


def parse_response(payload):
    """解析非流式响应的 JSON 文档为宿主响应结构。"""
    payload = payload if isinstance(payload, dict) else {}
    response = {
        "content": None,
        "reasoning_content": None,
        "tool_calls": [],
        "finish_reason": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    choice = _first_choice(payload)
    if choice is not None:
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        response["content"] = _text_or_none(message.get("content"))
        # 部分兼容端点把思考过程命名为 reasoning。
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        response["reasoning_content"] = _text_or_none(reasoning)
        response["tool_calls"] = parse_tool_calls(message.get("tool_calls"))
        response["finish_reason"] = choice.get("finish_reason") or ""

    usage = payload.get("usage")
    if isinstance(usage, dict):
        response["prompt_tokens"] = token_count(usage.get("prompt_tokens"))
        response["completion_tokens"] = token_count(usage.get("completion_tokens"))
    return response


def parse_tool_calls(raw_calls):
    """解析 OpenAI 响应中的 ``tool_calls`` 数组为宿主工具调用结构。"""
    calls = []
    for item in raw_calls or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments")
        calls.append(
            {
                "id": item.get("id") or "",
                "function_name": function.get("name") or "",
                "arguments": arguments if isinstance(arguments, str) else "",
            }
        )
    return calls


def token_count(value):
    """把 token 计数字段归一化为非负整数；缺省或非法值按 0 处理。"""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float) and value > 0:
        return int(value)
    return 0


def extract_error(payload):
    """从响应 JSON 中提取服务端错误摘要；没有错误时返回 None。

    个别兼容端点以 HTTP 200 + ``{"error": {...}}`` 的形式报错，因此需要
    与非流式响应解析分开检查。
    """
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        detail = (
            error.get("message")
            or error.get("detail")
            or json.dumps(error, ensure_ascii=False)
        )
        code = error.get("code") or error.get("type")
        if code:
            return f"服务端返回错误（{code}）：{detail}"
        return f"服务端返回错误：{detail}"
    return f"服务端返回错误：{error}"


def _first_choice(payload):
    """取 ``choices[0]``，不存在时返回 None。"""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return None


def _text_or_none(value):
    """取值：非空字符串原样返回，其余（含空串、null）统一返回 None。"""
    if isinstance(value, str) and value:
        return value
    return None


def _temperature(value):
    """温度归一化：合法数值原样返回，缺省/非法值回退到 0（贪心）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return value


def _positive_int(value):
    """取正整数（允许整数值的浮点数）；非正数或非法值返回 0。"""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float) and value > 0:
        return int(value)
    return 0
