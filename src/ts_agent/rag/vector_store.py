"""Chroma vector-store access and repeatable knowledge-base indexing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ts_agent.model.factory import get_embeddings
from ts_agent.utils.config_handler import chroma_conf, rag_conf
from ts_agent.utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from ts_agent.utils.logger_handler import logger
from ts_agent.utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(
        self,
        embeddings: Embeddings | None = None,
        persist_directory: str | Path | None = None,
        manifest_path: str | Path | None = None,
        data_path: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedding_model_name = str(rag_conf["embedding_model_name"])
        self.persist_directory = Path(
            persist_directory or get_abs_path(chroma_conf["persist_directory"])
        ).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(
            manifest_path
            or get_abs_path(
                chroma_conf.get("manifest_store", "data/db/chroma_manifest.json")
            )
        ).resolve()
        self.data_root = Path(
            data_path or get_abs_path(chroma_conf["data_path"])
        ).resolve()
        self.vector_store = Chroma(
            collection_name=collection_name or chroma_conf["collection_name"],
            embedding_function=embeddings or get_embeddings(),
            persist_directory=str(self.persist_directory),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"embedding_model": self.embedding_model_name, "files": {}}
        with self.manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
            raise ValueError(f"向量索引清单格式错误: {self.manifest_path}")
        return data

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    @staticmethod
    def _load_file(path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return txt_loader(str(path))
        if suffix == ".pdf":
            return pdf_loader(str(path))
        return []

    @staticmethod
    def _chunk_ids(source: str, md5_hex: str, count: int) -> list[str]:
        return [
            hashlib.sha256(f"{source}:{md5_hex}:{index}".encode()).hexdigest()
            for index in range(count)
        ]

    def _delete_ids(self, ids: list[str]) -> None:
        if ids:
            self.vector_store.delete(ids=ids)

    def rebuild(self) -> None:
        """Delete all existing vectors and rebuild from configured source files."""

        existing = self.vector_store.get(include=[]).get("ids", [])
        self._delete_ids(list(existing))
        self._save_manifest({"embedding_model": self.embedding_model_name, "files": {}})
        self.load_documents()

    def load_documents(self) -> None:
        """Synchronize changed, added, and removed knowledge files with Chroma."""

        if not self.manifest_path.is_file():
            existing_ids = self.vector_store.get(include=[]).get("ids", [])
            if existing_ids:
                raise RuntimeError(
                    "检测到无索引清单的旧 Chroma 数据。为避免重复向量，请运行 "
                    "`python -m ts_agent.rag.vector_store --rebuild`。"
                )
        manifest = self._load_manifest()
        if manifest.get("embedding_model") != self.embedding_model_name:
            raise RuntimeError(
                "Embedding 模型已变化，请运行 "
                "`python -m ts_agent.rag.vector_store --rebuild` 重建索引。"
            )

        files_manifest: dict[str, dict[str, Any]] = manifest["files"]
        source_files = [
            Path(item)
            for item in listdir_with_allowed_type(
                str(self.data_root), tuple(chroma_conf["allow_knowledge_file_type"])
            )
        ]
        current_sources = {str(path.relative_to(self.data_root)) for path in source_files}

        for removed_source in set(files_manifest) - current_sources:
            self._delete_ids(list(files_manifest[removed_source].get("ids", [])))
            del files_manifest[removed_source]

        for path in source_files:
            source = str(path.relative_to(self.data_root))
            md5_hex = get_file_md5_hex(str(path))
            if not md5_hex:
                logger.warning("[加载知识库]无法计算文件摘要，跳过: %s", path)
                continue
            previous = files_manifest.get(source, {})
            if previous.get("md5") == md5_hex:
                continue

            documents = self._load_file(path)
            chunks = self.splitter.split_documents(documents) if documents else []
            if not chunks:
                # 文件已变化但解析不出分片：清掉旧向量并落新 md5，
                # 避免脏内容一直可检索、且每次同步重复告警
                self._delete_ids(list(previous.get("ids", [])))
                files_manifest[source] = {"md5": md5_hex, "ids": []}
                logger.warning("[加载知识库]文件无有效分片，已清理旧分片: %s", path)
                continue

            self._delete_ids(list(previous.get("ids", [])))
            ids = self._chunk_ids(source, md5_hex, len(chunks))
            for index, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {"source": source, "source_md5": md5_hex, "chunk_index": index}
                )
            self.vector_store.add_documents(chunks, ids=ids)
            files_manifest[source] = {"md5": md5_hex, "ids": ids}
            logger.info("[加载知识库]同步成功: %s (%s 个分片)", path, len(chunks))

        manifest["embedding_model"] = self.embedding_model_name
        self._save_manifest(manifest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="同步或重建 Chroma 知识库")
    parser.add_argument("--rebuild", action="store_true", help="清空并重建索引")
    args = parser.parse_args()
    service = VectorStoreService()
    if args.rebuild:
        service.rebuild()
    else:
        service.load_documents()
