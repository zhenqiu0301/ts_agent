from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from ts_agent.agents import memory_utils
from ts_agent.agents.main_graph_agent import MainGraphAgent
from ts_agent.agents.persistence import build_persistent_backends


class SummaryModel:
    async def ainvoke(self, _messages, config=None):
        return AIMessage(content="用户需要一台适合养猫家庭的扫地机器人，预算3000元。")


class StructuredMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_episode_profile_retrieval_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            messages = [
                HumanMessage(content="我家90平方米，养猫，预算3000元，电话13800000000"),
                AIMessage(content="可以优先看防缠绕机型。"),
            ]
            try:
                changed = await memory_utils.save_memory_episode(
                    backends.store,
                    SummaryModel(),
                    "user-1",
                    "thread-1",
                    messages,
                    "purchase",
                )
                self.assertTrue(changed)
                snapshot = await memory_utils.list_user_memories(backends.store, "user-1")
                facts = {item["key"]: item for item in snapshot["profile"]}
                self.assertEqual(facts["home_area_sqm"]["value"], 90)
                self.assertEqual(facts["budget_cny"]["value"], 3000)
                self.assertTrue(facts["has_pets"]["confirmed"])
                self.assertNotIn("13800000000", str(snapshot))

                await memory_utils.save_memory_episode(
                    backends.store,
                    SummaryModel(),
                    "user-1",
                    "thread-2",
                    [HumanMessage(content="这次预算4000元")],
                    "purchase",
                )
                changed_snapshot = await memory_utils.list_user_memories(
                    backends.store, "user-1"
                )
                budget_candidates = [
                    item
                    for item in changed_snapshot["profile"]
                    if item["key"] == "budget_cny"
                ]
                self.assertTrue(any(item.get("conflict") for item in budget_candidates))
                self.assertTrue(any(item["value"] == 3000 for item in budget_candidates))

                relevant = await memory_utils.retrieve_relevant_memories(
                    backends.store, "user-1", "养猫选购", top_k=3
                )
                self.assertLessEqual(len(relevant), 3)
                self.assertTrue(relevant)

                await memory_utils.clear_user_memories(backends.store, "user-1")
                empty = await memory_utils.list_user_memories(backends.store, "user-1")
                self.assertEqual(empty, {"profile": [], "episodes": []})
            finally:
                await backends.close()


class PendingGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_purchase_bypasses_normal_router(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            agent = MainGraphAgent(backends, router_model=SummaryModel())
            try:
                pending = agent._build_review_state(
                    [{"name": "create_purchase_order", "args": {"product_model": "X1"}}]
                )
                await agent._save_pending_review(
                    "purchase", "thread-1:purchase", pending
                )
                result = await agent._pending_gate_node(
                    {"recent_messages": [HumanMessage(content="确认执行")]},
                    {"configurable": {"thread_id": "thread-1"}},
                )
                self.assertEqual(result["route"], "purchase")
                executing = await agent._update_pending_review(
                    "purchase", "thread-1:purchase", pending, "executing"
                )
                self.assertEqual(executing["status"], "executing")
                await agent._complete_pending_review(
                    "purchase", "thread-1:purchase", executing
                )
                self.assertIsNone(
                    await agent._get_pending_review("purchase", "thread-1:purchase")
                )
                history = await backends.store.aget(
                    ("hitl_history", "purchase"), "thread-1:purchase"
                )
                self.assertEqual(history.value["status"], "completed")
            finally:
                await agent.close()
