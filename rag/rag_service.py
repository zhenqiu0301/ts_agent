"""Retrieval-augmented response service."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import get_chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts


class RagSummarizeService:
    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        self.model = model or get_chat_model()
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = "\n".join(
            f"【参考资料{index}】{doc.page_content} | 元数据：{doc.metadata}"
            for index, doc in enumerate(context_docs, start=1)
        )
        return self.chain.invoke({"input": query, "context": context})


if __name__ == "__main__":
    service = RagSummarizeService()
    print(service.rag_summarize("大户型适合哪些扫地机器人"))
