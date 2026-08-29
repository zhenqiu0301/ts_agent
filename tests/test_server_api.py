"""服务端 API 测试：注入伪 Agent，不依赖真实模型与网络。"""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from ts_agent.server.app import create_app


class FakeCheckpointer:
    def __init__(self) -> None:
        self.checkpoints: dict[str, dict] = {}
        self.aget_calls = 0

    async def aget(self, config):
        self.aget_calls += 1
        thread_id = config.get("configurable", {}).get("thread_id")
        return self.checkpoints.get(thread_id)


class FakeAgent:
    MAX_RECENT_MESSAGES = 10

    def __init__(self, chunks=None, error: Exception | None = None) -> None:
        self.checkpointer = FakeCheckpointer()
        self.chunks = chunks if chunks is not None else ["你", "好"]
        self.error = error
        self.bootstrap_calls: list[tuple[str, str]] = []
        self.cleared_users: list[str] = []
        self.finalized: list[dict] = []

    async def load_user_memory_summary(self, user_id: str, query: str) -> str:
        self.bootstrap_calls.append((user_id, query))
        return "用户画像：预算3000"

    async def list_user_memories(self, user_id: str) -> dict:
        return {"profile": [{"key": "budget_cny", "value": 3000}], "episodes": []}

    async def clear_user_memories(self, user_id: str) -> None:
        self.cleared_users.append(user_id)

    async def finalize_thread(
        self, thread_id: str, user_id: str, recent_messages=None
    ) -> bool:
        self.finalized.append(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "messages": list(recent_messages or []),
            }
        )
        return True

    async def execute_stream(
        self,
        query: str,
        thread_id: str,
        user_id: str,
        bootstrap_summary=None,
        event_callback=None,
    ):
        if self.error is not None:
            raise self.error
        if event_callback is not None:
            event_callback({"type": "tools", "names": ["compare_prices"]})
            event_callback({"type": "nodes", "names": ["purchase"]})
        # 模拟真实执行：对话完成后线程产生 checkpoint
        self.checkpointer.checkpoints[thread_id] = {"channel_values": {}}
        for chunk in self.chunks:
            yield chunk


def parse_sse_events(response) -> list[dict]:
    """把 SSE 响应解析为 [{"type": ..., **data}] 事件列表。"""
    events = []
    event_type = ""
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
        elif line == "" and event_type:
            payload = "\n".join(data_lines)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"raw": payload}
            events.append({"type": event_type, **data})
            event_type, data_lines = "", []
    return events


class ServerApiTests(unittest.TestCase):
    def _client(self, agent: FakeAgent) -> TestClient:
        return TestClient(create_app(agent=agent))

    def test_health(self) -> None:
        with self._client(FakeAgent()) as client:
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_streams_chunks_tools_nodes_and_done(self) -> None:
        with self._client(FakeAgent(chunks=["你", "好"])) as client:
            with client.stream(
                "POST",
                "/api/chat",
                json={"thread_id": "t1", "user_id": "u1", "message": "你好"},
            ) as response:
                self.assertEqual(response.status_code, 200)
                self.assertTrue(
                    response.headers["content-type"].startswith("text/event-stream")
                )
                events = parse_sse_events(response)

        types = [event["type"] for event in events]
        self.assertEqual(types[-1], "done")
        self.assertEqual(events[-1]["response"], "你好")
        deltas = [event["delta"] for event in events if event["type"] == "chunk"]
        self.assertEqual("".join(deltas), "你好")
        self.assertIn("tool", types)
        self.assertIn("node", types)
        tool_event = next(event for event in events if event["type"] == "tool")
        self.assertEqual(tool_event["names"], ["compare_prices"])
        node_event = next(event for event in events if event["type"] == "node")
        self.assertEqual(node_event["names"], ["purchase"])

    def test_bootstrap_summary_loaded_only_on_first_turn(self) -> None:
        agent = FakeAgent(chunks=["好的"])
        with self._client(agent) as client:
            for _ in range(2):
                with client.stream(
                    "POST",
                    "/api/chat",
                    json={"thread_id": "t2", "user_id": "u1", "message": "选扫地机"},
                ) as response:
                    events = parse_sse_events(response)
                    self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(len(agent.bootstrap_calls), 1)
        self.assertEqual(agent.bootstrap_calls[0], ("u1", "选扫地机"))
        self.assertEqual(agent.checkpointer.aget_calls, 2)

    def test_chat_error_emits_error_event(self) -> None:
        agent = FakeAgent(error=RuntimeError("模型服务不可用"))
        with self._client(agent) as client:
            with client.stream(
                "POST",
                "/api/chat",
                json={"thread_id": "t3", "user_id": "u1", "message": "你好"},
            ) as response:
                events = parse_sse_events(response)
        self.assertEqual([event["type"] for event in events], ["error"])
        self.assertIn("失败", events[0]["message"])

    def test_memory_list_and_clear(self) -> None:
        agent = FakeAgent()
        with self._client(agent) as client:
            response = client.get("/api/memory/u1")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["profile"][0]["value"], 3000)

            response = client.delete("/api/memory/u1")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "cleared"})
        self.assertEqual(agent.cleared_users, ["u1"])

    def test_finalize_maps_roles_and_filters_invalid(self) -> None:
        agent = FakeAgent()
        payload = {
            "thread_id": "t9",
            "user_id": "u1",
            "messages": [
                {"role": "user", "content": " 问题一 "},
                {"role": "system", "content": "应被忽略"},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": "回答一"},
            ],
        }
        with self._client(agent) as client:
            response = client.post("/api/session/finalize", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["changed"])

        recorded = agent.finalized[0]
        self.assertEqual(recorded["thread_id"], "t9")
        self.assertEqual(len(recorded["messages"]), 2)
        self.assertIsInstance(recorded["messages"][0], HumanMessage)
        self.assertEqual(recorded["messages"][0].content, "问题一")
        self.assertIsInstance(recorded["messages"][1], AIMessage)
        self.assertEqual(recorded["messages"][1].content, "回答一")

    def test_finalize_windows_to_recent_messages(self) -> None:
        agent = FakeAgent()
        messages = [{"role": "user", "content": f"消息{i}"} for i in range(15)]
        with self._client(agent) as client:
            response = client.post(
                "/api/session/finalize",
                json={"thread_id": "t8", "user_id": "u1", "messages": messages},
            )
        self.assertTrue(response.json()["changed"])
        recorded = agent.finalized[0]
        self.assertEqual(len(recorded["messages"]), agent.MAX_RECENT_MESSAGES)
        self.assertEqual(
            [message.content for message in recorded["messages"]],
            [f"消息{i}" for i in range(5, 15)],
        )


if __name__ == "__main__":
    unittest.main()
