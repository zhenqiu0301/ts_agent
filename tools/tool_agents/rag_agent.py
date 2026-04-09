"""
StateGraph version of a standalone RAG agent.

This module is intentionally isolated and does not modify existing project flow.
"""

import os
import sys
from typing import TypedDict

# Compatible with running this file directly.
if __package__ is None or __package__ == "":
    # 当前文件在 tools/tool_agents/ 下，需要回退三级到项目根目录。
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts


class RagGraphState(TypedDict, total=False):
    query: str
    context_docs: list[Document]
    context: str
    answer: str


class RagStateGraphAgent:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.system_prompt = load_rag_prompts()
        self.model = chat_model
        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(RagGraphState)
        graph_builder.add_node("retrieve_docs", self._retrieve_docs_node)
        graph_builder.add_node("build_context", self._build_context_node)
        graph_builder.add_node("generate_answer", self._generate_answer_node)

        graph_builder.add_edge(START, "retrieve_docs")
        graph_builder.add_edge("retrieve_docs", "build_context")
        graph_builder.add_edge("build_context", "generate_answer")
        graph_builder.add_edge("generate_answer", END)

        return graph_builder.compile()

    def _retrieve_docs_node(self, state: RagGraphState) -> RagGraphState:
        query = state["query"]
        docs = self.retriever.invoke(query)
        return {"context_docs": docs}

    def _build_context_node(self, state: RagGraphState) -> RagGraphState:
        docs = state.get("context_docs", [])
        context_parts = []
        for i, doc in enumerate(docs, start=1):
            context_parts.append(
                f"【参考资料{i}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}"
            )
        return {"context": "\n".join(context_parts)}

    def _generate_answer_node(self, state: RagGraphState) -> RagGraphState:
        query = state["query"]
        context = state.get("context", "")
        user_prompt = f"用户问题：{query}\n\n参考资料：\n{context}"

        response = self.model.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer}

    def invoke(self, query: str) -> str:
        result = self.graph.invoke({"query": query})
        return result.get("answer", "")

    def stream(self, query: str):
        last_answer = ""
        for chunk in self.graph.stream({"query": query}, stream_mode="values"):
            answer = chunk.get("answer", "")
            if not answer:
                continue
            if answer.startswith(last_answer):
                delta = answer[len(last_answer) :]
            else:
                delta = answer
            if delta:
                yield delta
            last_answer = answer


if __name__ == "__main__":
    agent = RagStateGraphAgent()
    print(agent.invoke("大户型适合哪些扫地机器人"))
