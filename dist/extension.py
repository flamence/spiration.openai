"""Spiration OpenAI 格式支持拓展的入口。

加载器（``com.flamence.spiration.python``）会按 ``extension.json`` 中
``python.requirements`` 的声明安装官方 ``openai`` SDK，注入 ``spiration``
模块后调用本文件的 ``initialize`` / ``shutdown``；``initialize`` 向宿主注册
名为 ``openai`` 的智能体 provider，由 :class:`spiration.openai.provider.OpenAIProvider`
负责把补全请求转发到 OpenAI Chat Completions 兼容端点。
"""

import spiration

from spiration.openai.provider import OpenAIProvider

#: 拓展元信息（加载器读取后展示在拓展列表中）。
name = "OpenAI"
description = (
    "Provides OpenAI Chat Completions compatible model providers "
    "via the official OpenAI Python SDK."
)
version = "0.1.0"

#: 注册到宿主的 provider 名称，与 models.json 的 "provider" 字段对应。
PROVIDER_NAME = "openai"

#: 当前注册的 provider 实例；仅在注册成功后持有。
_provider = None


def initialize():
    """向宿主注册 ``openai`` provider。

    :return: 注册成功（或宿主已具备该能力并完成注册）返回 ``True``；
             宿主不支持智能体接口或注册失败时返回 ``False``。
    """
    global _provider

    agent = getattr(spiration, "agent", None)
    if agent is None or not agent.supported():
        spiration.log_error(
            "OpenAI extension: host does not provide the spiration.agent interface"
        )
        return False

    provider = OpenAIProvider()
    try:
        agent.register_provider(PROVIDER_NAME, provider)
    except RuntimeError as err:
        spiration.log_error(
            f"OpenAI extension: failed to register provider '{PROVIDER_NAME}': {err}"
        )
        return False

    _provider = provider
    spiration.log_info(
        f"OpenAI extension: provider '{PROVIDER_NAME}' registered "
        f"(reasoning levels: {', '.join(provider.reasoning_levels())})"
    )
    return True


def shutdown():
    """注销 provider；关闭阶段的异常一律忽略。

    加载器保证注销接口对不存在的名称幂等，这里仍兜底捕获，避免拓展卸载
    过程因为宿主接口异常而中断。
    """
    global _provider
    _provider = None

    agent = getattr(spiration, "agent", None)
    if agent is None:
        return
    try:
        agent.unregister_provider(PROVIDER_NAME)
    except Exception:  # noqa: BLE001 - 关闭阶段不再抛出任何异常
        pass
