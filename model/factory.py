"""Lazy factories for the application's chat and embedding models."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path

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

    return DashScopeEmbeddings(
        model=rag_conf["embedding_model_name"],
        dashscope_api_key=_require_env("DASHSCOPE_API_KEY"),
    )
