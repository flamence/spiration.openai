"""宿主补全请求/响应与官方 ``openai`` SDK 对象之间的转换。

本模块不发起网络请求，纯函数便于单元测试；SDK 流式 chunk 的累积见
``stream`` 模块。

协议要点（openai 3.37.0 实测）：

- 非流式响应是 ``ChatCompletion`` 对象，流式响应是可迭代的
  ``ChatCompletionChunk`` 序列；
- 服务端扩展字段（如 DeepSeek 的 ``reasoning_content``）由 SDK 放进模型对象的
  ``model_extra``，同时也能通过 ``getattr`` 直接读取，统一用
  :func:`reasoning_text` 提取；
- 现代 Chat Completions 用 ``max_completion_tokens`` 表达输出上限。
"""

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

#: 工具未提供 parameters 时的缺省 JSON Schema。
_DEFAULT_TOOL_PARAMETERS = {"type": "object", "properties": {}}


def build_request(request):
    """把宿主补全请求（dict）转换为 ``chat.completions.create`` 的关键字参数。"""
    request = request or {}
    stream = bool(request.get("stream"))
    kwargs = {
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
        # 现代 OpenAI Chat Completions 的输出上限字段；DeepSeek 等兼容端点
        # 实测同样接受该字段。
        kwargs["max_completion_tokens"] = max_tokens

    tools = [
        convert_tool(tool)
        for tool in request.get("tools") or []
        if isinstance(tool, dict)
    ]
    if tools:
        kwargs["tools"] = tools

    if stream:
        # 请求最后一个 chunk 携带 token 用量；不支持的端点会忽略该字段。
        kwargs["stream_options"] = {"include_usage": True}

    effort = reasoning_effort(request.get("reasoning"))
    if effort:
        kwargs["reasoning_effort"] = effort
    return kwargs


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


def parse_response(completion):
    """把非流式 ``ChatCompletion`` 对象解析为宿主响应结构。

    ``content`` 为 None（例如思考模型把 token 全用在思考过程上）时原样保留
    None，不伪造空串。
    """
    response = {
        "content": None,
        "reasoning_content": None,
        "tool_calls": [],
        "finish_reason": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    choices = getattr(completion, "choices", None)
    choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
    if choice is not None:
        message = getattr(choice, "message", None)
        if message is not None:
            response["content"] = _text_or_none(getattr(message, "content", None))
            response["reasoning_content"] = reasoning_text(message)
            response["tool_calls"] = parse_tool_calls(
                getattr(message, "tool_calls", None)
            )
        finish_reason = getattr(choice, "finish_reason", None)
        response["finish_reason"] = (
            finish_reason if isinstance(finish_reason, str) else ""
        )

    usage = getattr(completion, "usage", None)
    if usage is not None:
        response["prompt_tokens"] = token_count(getattr(usage, "prompt_tokens", None))
        response["completion_tokens"] = token_count(
            getattr(usage, "completion_tokens", None)
        )
    return response


def parse_tool_calls(raw_calls):
    """解析 SDK 响应中的 ``tool_calls`` 数组为宿主工具调用结构。"""
    if not isinstance(raw_calls, (list, tuple)):
        return []
    calls = []
    for item in raw_calls:
        if item is None:
            continue
        function = getattr(item, "function", None)
        name = getattr(function, "name", None) if function is not None else None
        arguments = (
            getattr(function, "arguments", None) if function is not None else None
        )
        calls.append(
            {
                "id": _text_or_none(getattr(item, "id", None)) or "",
                "function_name": _text_or_none(name) or "",
                "arguments": arguments if isinstance(arguments, str) else "",
            }
        )
    return calls


def reasoning_text(obj):
    """读取服务端扩展字段中的思考过程；没有（或为空）时返回 None。

    SDK 3.37.0 会同时以直接属性和 ``model_extra`` 键暴露 ``reasoning_content``，
    这里兼容两种形态（并兼容部分兼容端点使用的 ``reasoning`` 别名）。
    读取顺序：``reasoning_content`` 直接属性 → ``reasoning`` 直接属性 →
    ``model_extra`` 中的同名字段；空串/None 一律归一为 None。
    """
    if obj is None:
        return None
    for name in ("reasoning_content", "reasoning"):
        text = _text_or_none(getattr(obj, name, None))
        if text is not None:
            return text
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        for name in ("reasoning_content", "reasoning"):
            text = _text_or_none(extra.get(name))
            if text is not None:
                return text
    return None


def token_count(value):
    """把 token 计数字段归一化为非负整数；缺省或非法值按 0 处理。"""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float) and value > 0:
        return int(value)
    return 0


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
