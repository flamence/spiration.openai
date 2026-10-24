"""测试共享辅助：准备导入路径、模拟流对象与默认宿主请求。"""

import os
import sys

# dist/ 是拓展运行时由加载器加入 sys.path 的目录，这里为测试显式加入。
DIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist"
)
if DIST_DIR not in sys.path:
    sys.path.insert(0, DIST_DIR)


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
        """用户是否已请求中止（测试用：按已收增量数模拟）。"""
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


def make_request(endpoint, **overrides):
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
