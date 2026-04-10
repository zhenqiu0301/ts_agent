from __future__ import annotations

import os
import re
import sys
from typing import Annotated, TypedDict

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langchain_core.runnables import RunnableConfig

try:
    from model.factory import chat_model
    from .sub_agents import AfterSalesAgent, PurchaseAgent
except ImportError:
    from model.factory import chat_model
    from agents_v2.sub_agents import AfterSalesAgent, PurchaseAgent


class MainGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    route: str
    response: str


class MainGraphAgentV2:
    """主控 Agent：路由到选购/售后子 Agent。"""

    MAX_RECENT_MESSAGES = 8

    def __init__(self):
        self.router_model = chat_model
        self.purchase_agent = PurchaseAgent().agent
        self.after_sales_agent = AfterSalesAgent().agent
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MainGraphState)

        builder.add_node("analyze", self._analyze_node)
        builder.add_node("purchase", self._purchase_node)
        builder.add_node("after_sales", self._after_sales_node)
        builder.add_node("summarize", self._summarize_node)

        builder.add_edge(START, "analyze")
        builder.add_conditional_edges(
            "analyze",
            self._route_selector,
            {
                "purchase": "purchase",
                "after_sales": "after_sales",
                "unclear": END,
            },
        )
        builder.add_edge("purchase", "summarize")
        builder.add_edge("after_sales", "summarize")
        builder.add_edge("summarize", END)

        return builder.compile(checkpointer=InMemorySaver())

    def _build_messages(self, state: MainGraphState) -> str | list[BaseMessage]:
        summary = state.get("summary", "").strip()
        messages = list(state.get("messages", []))
        if summary:
            return [SystemMessage(content=f"历史摘要：{summary}"), *messages]
        return messages

    def _analyze_node(self, state: MainGraphState):
        query = self._build_messages(state)
        system_prompt = (
            "你是客服路由分析器。请判断用户意图属于以下二选一：\n"
            "1) purchase: 选购/推荐/对比/预算/户型匹配\n"
            "2) after_sales: 故障排查/维修/售后/保养/报告查询\n\n"
            "如果用户语义不清、信息不足以判断，输出 unclear。\n"
            "只输出一个标签：purchase、after_sales 或 unclear，不要输出其他内容。"
        )
        user_prompt = f"用户问题：{query}"
        result = self.router_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        text = (result.content or "").strip().lower()

        match = re.search(r"\b(purchase|after_sales|unclear)\b", text)
        route = match.group(1) if match else "unclear"
        if route == "unclear":
            return {
                "route": "unclear",
                "response": "我还不太确定你的诉求是选购还是售后。请补充一下你的目标或问题细节。",
            }
        return {"route": route}

    def _route_selector(self, state: MainGraphState) -> str:
        route = state.get("route", "unclear")
        return route if route in {"purchase", "after_sales", "unclear"} else "unclear"

    def _purchase_node(self, state: MainGraphState, config: RunnableConfig):
        main_thread_id = config.get("configurable", {}).get("thread_id", "default")
        result = self.purchase_agent.invoke(
            {"messages": self._build_messages(state)},
            context={"route": "purchase", "report": False},
            config={"configurable": {"thread_id": f"{main_thread_id}:purchase"}},
        )
        reply = next(
            (
                msg.content
                for msg in reversed(result["messages"])
                if isinstance(msg, AIMessage)
            ),
            "",
        )
        return {"messages": [AIMessage(content=reply)], "response": reply}

    def _after_sales_node(self, state: MainGraphState, config: RunnableConfig):
        main_thread_id = config.get("configurable", {}).get("thread_id", "default")
        result = self.after_sales_agent.invoke(
            {"messages": self._build_messages(state)},
            context={"route": "after_sales", "report": False},
            config={"configurable": {"thread_id": f"{main_thread_id}:after-sales"}},
        )
        reply = 下一处(
            (
                msg.content
                for msg in reversed(result["messages"])
                if isinstance(msg, AIMessage)
            ),
            "",
        )
        return {"messages": [AIMessage(content=reply)], "response": reply}

    @staticmethod
    def _messages_to_plain_text(messages: list[BaseMessage]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _merge_summary(self, old_summary: str, old_messages: list[BaseMessage]) -> str:
        conversation = self._messages_to_plain_text(old_messages)
        system_prompt = (
            "你是对话摘要助手，请将历史客服对话压缩为简洁摘要，保留事实与约束。\n"
            "输出要求：\n"
            "1) 50-150字\n"
            "2) 包含：用户目标、已确认条件、未解决问题\n"
            "3) 不要编造\n"
        )
        user_prompt = (
            f"已有摘要：{old_summary or '无'}\n"
            f"新增历史对话：\n{conversation or '无'}\n"
            "请输出更新后的摘要："
        )
        result = self.router_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        text = (result.content or "").strip()
        return text or old_summary

    def _summarize_node(self, state: MainGraphState):
        messages = state.get("messages", [])
        summary = state.get("summary", "")
        if len(messages) <= self.MAX_RECENT_MESSAGES:
            return {}

        old_messages = messages[: -self.MAX_RECENT_MESSAGES]
        recent_messages = messages[-self.MAX_RECENT_MESSAGES :]
        new_summary = self._merge_summary(summary, old_messages)
        return {
            "summary": new_summary,
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages],
        }

    def execute_stream(self, query: str, thread_id: str):
        last_emitted = ""
        for chunk in self.graph.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values",
            config={"configurable": {"thread_id": thread_id}},
        ):
            text = chunk.get("response", "").strip()
            if not text:
                continue

            if text.startswith(last_emitted):
                delta = text[len(last_emitted) :]
            else:
                delta = text

            if delta:
                yield delta
                last_emitted = text


if __name__ == "__main__":
    agent = MainGraphAgentV2()
    for piece in agent.execute_stream(
        "我家120平有宠物，想买台适合的扫拖机器人", "demo"
    ):
        print(piece, end="", flush=True)
