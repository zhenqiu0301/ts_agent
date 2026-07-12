from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from ts_agent.agents.main_graph_agent import MainGraphAgent
from ts_agent.agents.persistence import build_persistent_backends
from ts_agent.model.factory import get_chat_model, get_embeddings
from ts_agent.rag.vector_store import VectorStoreService
from ts_agent.tools.tools import get_rag_service, get_tavily_search
from ts_agent.utils.file_handler import listdir_with_allowed_type


class LazyInitializationTests(unittest.TestCase):
    def test_clients_are_not_created_at_import_time(self) -> None:
        get_embeddings.cache_clear()
        self.assertEqual(get_embeddings.cache_info().currsize, 0)
        self.assertTrue(callable(get_chat_model))
        self.assertTrue(callable(get_rag_service))
        self.assertTrue(callable(get_tavily_search))


class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_backends_can_be_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            await backends.close()
            with self.assertRaises(ValueError):
                await backends.checkpoint_connection.execute("SELECT 1")

    async def test_main_agent_uses_injected_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            agent = MainGraphAgent(persistence=backends, router_model=AsyncRouterModel())
            self.assertIs(agent.checkpointer, backends.checkpointer)
            self.assertIs(agent.store, backends.store)
            await agent.close()


class AsyncRouterModel:
    def __init__(self, route: str = "unclear") -> None:
        self.route = route

    async def ainvoke(self, _messages, config=None):
        return AIMessage(content=self.route)


class StreamingChildAgent:
    def __init__(self, response: str) -> None:
        self.model = FakeListChatModel(responses=[response])

    async def ainvoke(self, _payload, *, context, config):
        chunks = []
        async for chunk in self.model.astream("answer", config=config):
            chunks.append(str(chunk.content))
        return {"messages": [AIMessage(content="".join(chunks))]}


class AsyncExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_graph_streams_without_sync_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            agent = MainGraphAgent(backends, router_model=AsyncRouterModel())
            agent.graph = agent._build_graph()
            try:
                chunks = [
                    chunk
                    async for chunk in agent.execute_stream(
                        "这是什么",
                        "async-test",
                        "user",
                    )
                ]
                self.assertIn("请补充", "".join(chunks))
            finally:
                await agent.close()

    async def test_purchase_response_is_streamed_as_model_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            agent = MainGraphAgent(backends, router_model=AsyncRouterModel("purchase"))
            agent.purchase_agent = StreamingChildAgent("流式回答")
            agent.graph = agent._build_graph()
            events = []
            try:
                chunks = [
                    chunk
                    async for chunk in agent.execute_stream(
                        "帮我选一台",
                        "stream-test",
                        "user",
                        event_callback=events.append,
                    )
                ]
                self.assertGreater(len(chunks), 1)
                self.assertEqual("".join(chunks), "流式回答")
                self.assertTrue(any(event["type"] == "nodes" for event in events))
            finally:
                await agent.close()


class RoutingTests(unittest.TestCase):
    def test_sensitive_action_decisions(self) -> None:
        self.assertEqual(
            MainGraphAgent._parse_ticket_review_decision("确认执行"),
            {"type": "approve"},
        )
        self.assertEqual(
            MainGraphAgent._parse_ticket_review_decision("暂不执行"),
            {"type": "reject", "message": "用户暂不执行敏感售后操作，继续在线处理。"},
        )
        self.assertIsNone(MainGraphAgent._parse_ticket_review_decision("再考虑一下"))

    def test_pending_action_is_stored_as_structured_state(self) -> None:
        state = MainGraphAgent._build_review_state(
            [{"name": "create_purchase_order", "args": {"product_model": "X1"}}]
        )
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(state["count"], 1)
        self.assertEqual(state["tools"], ["create_purchase_order"])
        self.assertEqual(state["actions"][0]["args"]["product_model"], "X1")


class FileDiscoveryTests(unittest.TestCase):
    def test_missing_directory_returns_no_files(self) -> None:
        self.assertEqual(listdir_with_allowed_type("/not/a/real/path", ("txt",)), ())

    def test_extensions_are_case_insensitive_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.TXT").write_text("b", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "skip.md").write_text("x", encoding="utf-8")
            self.assertEqual(
                [Path(item).name for item in listdir_with_allowed_type(directory, ("txt",))],
                ["a.txt", "b.TXT"],
            )


class VectorIndexTests(unittest.TestCase):
    def test_index_sync_handles_changes_and_removals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "raw"
            data_dir.mkdir()
            first = data_dir / "first.txt"
            first.write_text("第一版知识内容。", encoding="utf-8")

            service = VectorStoreService(
                embeddings=DeterministicFakeEmbedding(size=8),
                persist_directory=root / "chroma",
                manifest_path=root / "manifest.json",
                data_path=data_dir,
                collection_name="test-agent",
            )
            service.load_documents()
            first_ids = set(service.vector_store.get(include=[])["ids"])
            self.assertTrue(first_ids)

            first.write_text("第二版知识内容，长度有所变化。", encoding="utf-8")
            service.load_documents()
            second_ids = set(service.vector_store.get(include=[])["ids"])
            self.assertTrue(second_ids)
            self.assertTrue(first_ids.isdisjoint(second_ids))

            first.unlink()
            second = data_dir / "second.txt"
            second.write_text("新增知识。", encoding="utf-8")
            service.load_documents()
            result = service.vector_store.get(include=["metadatas"])
            self.assertTrue(result["ids"])
            self.assertEqual(
                {metadata["source"] for metadata in result["metadatas"]},
                {"second.txt"},
            )


if __name__ == "__main__":
    unittest.main()
