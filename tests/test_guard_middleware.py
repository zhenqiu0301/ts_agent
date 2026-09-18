from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ts_agent.tools.middleware import (
    MAX_TOOL_ROUNDS,
    TRUNCATED_SUFFIX,
    count_tool_rounds,
    limit_tool_rounds_core,
    truncate_tool_result,
)


class TruncateToolResultTests(unittest.TestCase):
    def test_short_result_passes_through_unchanged(self) -> None:
        self.assertEqual(truncate_tool_result("正常结果"), "正常结果")
        self.assertEqual(truncate_tool_result(None), None)
        self.assertEqual(truncate_tool_result({"a": 1}), {"a": 1})

    def test_long_result_is_truncated_with_suffix(self) -> None:
        result = truncate_tool_result("x" * 5000, limit=100)
        self.assertEqual(len(result), 100 + len(TRUNCATED_SUFFIX))
        self.assertTrue(result.endswith("不要重复调用]"))

    def test_limit_can_be_disabled_via_zero(self) -> None:
        text = "y" * 300
        self.assertEqual(truncate_tool_result(text, limit=10**9), text)


class CountToolRoundsTests(unittest.TestCase):
    def test_counts_ai_messages_with_tool_calls(self) -> None:
        messages = [
            HumanMessage(content="问题"),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {}, "id": "1"}],
            ),
            ToolMessage(content="结果", tool_call_id="1"),
            AIMessage(content="最终回答"),
        ]
        self.assertEqual(count_tool_rounds(messages), 1)

    def test_empty_history_has_zero_rounds(self) -> None:
        self.assertEqual(count_tool_rounds([HumanMessage(content="你好")]), 0)


class _FakeModelRequest:
    """模拟 ModelRequest 的最小接口：override 返回新对象。"""

    def __init__(self, tools, messages):
        self.tools = tools
        self.state = {"messages": messages}
        self._overridden = {}

    def override(self, **kwargs):
        clone = _FakeModelRequest(self.tools, self.state["messages"])
        clone._overridden = kwargs
        for key, value in kwargs.items():
            setattr(clone, key, value)
        return clone


class LimitToolRoundsTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _run(request):
        captured = {}

        async def handler(req):
            captured["request"] = req
            return "done"

        result = await limit_tool_rounds_core(request, handler)
        return result, captured["request"]

    async def test_under_limit_keeps_tools(self) -> None:
        request = _FakeModelRequest(tools=["t1"], messages=[HumanMessage(content="hi")])
        result, passed = await self._run(request)
        self.assertEqual(result, "done")
        self.assertEqual(passed.tools, ["t1"])

    async def test_over_limit_strips_all_tools(self) -> None:
        history = [
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {}, "id": str(i)}],
            )
            for i in range(MAX_TOOL_ROUNDS)
        ]
        request = _FakeModelRequest(tools=["t1", "t2"], messages=history)
        result, passed = await self._run(request)
        self.assertEqual(result, "done")
        self.assertEqual(passed.tools, [])


class MemoryInjectionTests(unittest.IsolatedAsyncioTestCase):
    """记忆上下文经 state 注入子 agent 消息前缀。"""

    def _agent_with_state(self, memory_context: str):
        from ts_agent.agents.main_graph_agent import MainGraphAgent

        state = {
            "recent_messages": [HumanMessage(content="推荐一台扫地机")],
            "summary": "旧摘要",
            "memory_context": memory_context,
        }
        messages = MainGraphAgent._build_messages(None, state)
        return messages

    def test_memory_context_becomes_first_system_prefix(self) -> None:
        messages = self._agent_with_state("预算3000元；家有宠物")
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("已知用户背景", messages[0].content)
        self.assertIn("预算3000元", messages[0].content)
        self.assertIn("历史摘要", messages[1].content)

    def test_empty_memory_skips_prefix(self) -> None:
        messages = self._agent_with_state("")
        # 记忆为空时不产生"已知用户背景"前缀，summary 前缀行为不变
        self.assertFalse(any("已知用户背景" in str(m.content) for m in messages))
        self.assertEqual(messages[0].content, "历史摘要：旧摘要")
        self.assertEqual(messages[1].content, "推荐一台扫地机")


if __name__ == "__main__":
    unittest.main()
