"""``extension.py`` 入口行为测试：用桩对象替代宿主注入的 ``spiration`` 模块。"""

import importlib.util
import os
import sys
import unittest

from support import DIST_DIR

EXTENSION_PATH = os.path.join(DIST_DIR, "extension.py")


class _FakeAgent:
    """模拟 ``spiration.agent`` 子模块。"""

    def __init__(self, supported=True, register_error=None, unregister_error=None):
        self._supported = supported
        self._register_error = register_error
        self._unregister_error = unregister_error
        self.registered = []
        self.unregistered = []

    def supported(self):
        """宿主是否提供智能体接口。"""
        return self._supported

    def register_provider(self, name, provider):
        """记录注册调用，或按配置抛出异常。"""
        if self._register_error is not None:
            raise self._register_error
        self.registered.append((name, provider))

    def unregister_provider(self, name):
        """记录注销调用，或按配置抛出异常。"""
        if self._unregister_error is not None:
            raise self._unregister_error
        self.unregistered.append(name)


class _FakeSpiration:
    """模拟加载器注入的 ``spiration`` 模块。"""

    def __init__(self, agent="default"):
        self.agent = _FakeAgent() if agent == "default" else agent
        self.infos = []
        self.errors = []

    def log_info(self, message):
        """记录 info 日志。"""
        self.infos.append(message)

    def log_error(self, message):
        """记录 error 日志。"""
        self.errors.append(message)


def _load_extension(fake_spiration):
    """把 ``extension.py`` 作为独立模块加载，宿主模块用桩替代。"""
    previous = sys.modules.get("spiration")
    sys.modules["spiration"] = fake_spiration
    try:
        spec = importlib.util.spec_from_file_location(
            "openai_extension_under_test", EXTENSION_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is not None:
            sys.modules["spiration"] = previous
        else:
            sys.modules.pop("spiration", None)
    return module


class ExtensionEntryTests(unittest.TestCase):
    """入口元信息。"""

    def test_metadata(self):
        module = _load_extension(_FakeSpiration())
        for attribute in ("name", "description", "version"):
            self.assertIsInstance(getattr(module, attribute), str, attribute)
            self.assertTrue(getattr(module, attribute), attribute)
        self.assertIn("OpenAI", module.description)
        self.assertIn("SDK", module.description)
        self.assertEqual(module.PROVIDER_NAME, "openai")

    def test_entry_does_not_construct_a_client(self):
        """加载入口只注册 provider，不构造 SDK 客户端（首次加载无需联网）。"""
        spiration = _FakeSpiration()
        module = _load_extension(spiration)
        self.assertTrue(module.initialize())
        _, provider = spiration.agent.registered[0]
        self.assertEqual(provider._clients, {})


class InitializeTests(unittest.TestCase):
    """``initialize()`` 的注册与失败路径。"""

    def test_unsupported_host(self):
        spiration = _FakeSpiration(agent=_FakeAgent(supported=False))
        module = _load_extension(spiration)
        self.assertFalse(module.initialize())
        self.assertEqual(spiration.agent.registered, [])
        self.assertEqual(len(spiration.errors), 1)
        self.assertIn("spiration.agent", spiration.errors[0])

    def test_missing_agent_module(self):
        spiration = _FakeSpiration(agent=None)
        module = _load_extension(spiration)
        self.assertFalse(module.initialize())
        self.assertEqual(len(spiration.errors), 1)

    def test_registers_provider(self):
        spiration = _FakeSpiration()
        module = _load_extension(spiration)
        self.assertTrue(module.initialize())

        self.assertEqual(len(spiration.agent.registered), 1)
        provider_name, provider = spiration.agent.registered[0]
        self.assertEqual(provider_name, "openai")
        self.assertTrue(callable(provider.execute))
        self.assertEqual(
            provider.reasoning_levels(), ["none", "low", "medium", "high"]
        )
        self.assertEqual(len(spiration.infos), 1)
        self.assertIn("openai", spiration.infos[0])

    def test_register_failure(self):
        spiration = _FakeSpiration(
            agent=_FakeAgent(register_error=RuntimeError("name already exists"))
        )
        module = _load_extension(spiration)
        self.assertFalse(module.initialize())
        self.assertIn("name already exists", spiration.errors[0])


class ShutdownTests(unittest.TestCase):
    """``shutdown()`` 的注销与异常兜底。"""

    def test_unregister_after_initialize(self):
        spiration = _FakeSpiration()
        module = _load_extension(spiration)
        self.assertTrue(module.initialize())
        module.shutdown()
        self.assertEqual(spiration.agent.unregistered, ["openai"])

    def test_shutdown_without_agent_module(self):
        module = _load_extension(_FakeSpiration(agent=None))
        module.shutdown()  # 不应抛出

    def test_shutdown_swallows_errors(self):
        spiration = _FakeSpiration(
            agent=_FakeAgent(unregister_error=RuntimeError("host gone"))
        )
        module = _load_extension(spiration)
        module.shutdown()  # 不应抛出


if __name__ == "__main__":
    unittest.main()
