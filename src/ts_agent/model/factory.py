"""Lazy factories for the application's chat and embedding models."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ts_agent.utils.config_handler import rag_conf
from ts_agent.utils.path_tool import get_abs_path

load_dotenv(get_abs_path(".env"))


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少 {name}。请在项目根目录 .env 中配置。")
    return value


def get_chat_model() -> BaseChatModel:
    """Create a chat client for the caller's current async event loop."""

    return ChatOpenAI(
        model=rag_conf["chat_model_name"],
        api_key=_require_env("DEEPSEEK_API_KEY"),
        base_url=rag_conf.get("chat_base_url", "https://api.deepseek.com"),
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Create the DashScope embedding client on first RAG use."""

    # langchain-community 的 BATCH_SIZE 表只登记了旧模型名，未知模型名会按 25 批量
    # 发送；qwen3.7-text-embedding 服务端上限是 20，这里统一按安全批量切分。
    return BatchedEmbeddings(
        DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=_require_env("DASHSCOPE_API_KEY"),
        ),
        batch_size=10,
    )


class BatchedEmbeddings(Embeddings):
    """按固定批量切分调用底层 embeddings，规避服务端批量上限。"""

    def __init__(self, inner: Embeddings, batch_size: int = 10) -> None:
        self.inner = inner
        self.batch_size = max(1, batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self.inner.embed_documents(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)
