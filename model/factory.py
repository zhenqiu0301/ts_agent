from abc import ABC, abstractmethod
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path

load_dotenv(get_abs_path(".env"))


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError(
                "缺少 DASHSCOPE_API_KEY。请在项目根目录 .env 中配置，"
                "例如：DASHSCOPE_API_KEY=your_key"
            )
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
