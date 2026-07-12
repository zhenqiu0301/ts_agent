from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from agents.main_graph_agent import MainGraphAgent
from agents.persistence import build_persistent_backends
from model.factory import get_chat_model, get_embeddings
from rag.vector_store import VectorStoreService
from tools.tools import get_rag_service, get_tavily_search
from utils.file_handler import listdir_with_allowed_type


class LazyInitializationTests(unittest.TestCase):
    def test_clients_are_not_created_at_import_time(self) -> None:
        get_chat_model.cache_clear()
        get_embeddings.cache_clear()
        get_rag_service.cache_clear()
        get_tavily_search.cache_clear()
        self.assertEqual(get_chat_model.cache_info().currsize, 0)
        self.assertEqual(get_embeddings.cache_info().currsize, 0)
        self.assertEqual(get_rag_service.cache_info().currsize, 0)
        self.assertEqual(get_tavily_search.cache_info().currsize, 0)


class PersistenceTests(unittest.TestCase):
    def test_backends_can_be_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = build_persistent_backends(Path(directory))
            backends.close()
            with self.assertRaises(sqlite3.ProgrammingError):
                backends.checkpoint_connection.execute("SELECT 1")

    def test_main_agent_uses_injected_persistence(self) -> None:
        os.environ["MCP_DISABLE_EXTERNAL"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            backends = build_persistent_backends(Path(directory))
            agent = MainGraphAgent(persistence=backends)
            self.assertIs(agent.checkpointer, backends.checkpointer)
            self.assertIs(agent.store, backends.store)
            agent.close()


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
