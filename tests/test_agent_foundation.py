from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

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


class CountingSummaryModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages, config=None):
        self.calls += 1
        return AIMessage(content="合并后的摘要")


class SummarizeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _messages(count: int) -> list:
        return [
            HumanMessage(content=f"问题{i}") if i % 2 == 0 else AIMessage(content=f"回答{i}")
            for i in range(count)
        ]

    async def test_summarize_skips_at_exact_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            agent = MainGraphAgent(backends, router_model=CountingSummaryModel())
            try:
                updates = await agent._summarize_node(
                    {
                        "recent_messages": self._messages(agent.MAX_RECENT_MESSAGES),
                        "summary": "旧摘要",
                    },
                    {},
                )
                self.assertEqual(updates, {})
                self.assertEqual(agent.router_model.calls, 0)
            finally:
                await agent.close()

    async def test_summarize_merges_only_overflow_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            agent = MainGraphAgent(backends, router_model=CountingSummaryModel())
            try:
                updates = await agent._summarize_node(
                    {
                        "recent_messages": self._messages(agent.MAX_RECENT_MESSAGES + 2),
                        "summary": "旧摘要",
                    },
                    {},
                )
                self.assertEqual(agent.router_model.calls, 1)
                self.assertEqual(updates["summary"], "合并后的摘要")
                trimmed = updates["recent_messages"]
                self.assertIsInstance(trimmed[0], RemoveMessage)
                self.assertEqual(len(trimmed) - 1, agent.MAX_RECENT_MESSAGES)
            finally:
                await agent.close()


class RecordingChildAgent:
    """记录每次调用的子线程 id。"""

    def __init__(self) -> None:
        self.thread_ids: list[str] = []

    async def ainvoke(self, _payload, *, context, config):
        self.thread_ids.append(str(config["configurable"]["thread_id"]))
        return {"messages": [AIMessage(content="好的")]}


class SubThreadIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_turn_uses_fresh_child_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = await build_persistent_backends(Path(directory))
            agent = MainGraphAgent(backends, router_model=AsyncRouterModel())
            child = RecordingChildAgent()
            agent.purchase_agent = child
            config = {"configurable": {"thread_id": "iso-main", "user_id": "user"}}
            try:
                for _ in range(2):
                    result = await agent._purchase_node(
                        {
                            "recent_messages": [HumanMessage(content="帮我选")],
                            "summary": "",
                        },
                        config,
                    )
                    self.assertEqual(result["response"], "好的")
                self.assertEqual(len(child.thread_ids), 2)
                self.assertNotEqual(child.thread_ids[0], child.thread_ids[1])
                for thread_id in child.thread_ids:
                    self.assertTrue(thread_id.startswith("iso-main:purchase:"))
            finally:
                await agent.close()

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

    def test_index_sync_clears_stale_chunks_when_file_becomes_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "raw"
            data_dir.mkdir()
            doc = data_dir / "doc.txt"
            doc.write_text("第一版知识内容。", encoding="utf-8")

            service = VectorStoreService(
                embeddings=DeterministicFakeEmbedding(size=8),
                persist_directory=root / "chroma",
                manifest_path=root / "manifest.json",
                data_path=data_dir,
                collection_name="test-agent-empty",
            )
            service.load_documents()
            self.assertTrue(set(service.vector_store.get(include=[])["ids"]))

            doc.write_text("", encoding="utf-8")
            service.load_documents()
            self.assertEqual(service.vector_store.get(include=[])["ids"], [])
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"]["doc.txt"]["ids"], [])

            # md5 已更新，二次同步不再重复处理
            service.load_documents()
            manifest_again = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["files"], manifest_again["files"])


if __name__ == "__main__":
    unittest.main()
