from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import Command

from ts_agent.agents import memory_utils
from ts_agent.agents.persistence import PersistentBackends, build_persistent_backends
from ts_agent.agents.sub_agents import AfterSalesAgent, PurchaseAgent
from ts_agent.model.factory import get_chat_model
from ts_agent.utils.logger_handler import logger


class MainGraphState(TypedDict):
    recent_messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    route: str
    response: str


class MainGraphAgent:
    """主控 Agent"""

    MAX_RECENT_MESSAGES = 10

    def __init__(
        self,
        persistence: PersistentBackends,
        router_model: Any | None = None,
    ):
        self._persistence = persistence
        self.checkpointer = self._persistence.checkpointer
        self.store = self._persistence.store
        self.router_model = router_model or get_chat_model()
        self.purchase_agent = None
        self.after_sales_agent = None
        self.graph = None

    @classmethod
    async def create(
        cls,
        persistence: PersistentBackends | None = None,
        router_model: Any | None = None,
    ) -> MainGraphAgent:
        backends = persistence or await build_persistent_backends()
        instance: MainGraphAgent | None = None
        try:
            instance = cls(backends, router_model=router_model)
            instance.purchase_agent = (
                await PurchaseAgent.create(backends.checkpointer, instance.router_model)
            ).agent
            instance.after_sales_agent = (
                await AfterSalesAgent.create(backends.checkpointer, instance.router_model)
            ).agent
            instance.graph = instance._build_graph()
        except Exception:
            # 构建失败时释放自建资源，避免 SQLite 连接与模型客户端泄漏；
            # 调用方传入的资源仍由调用方持有并自行管理。
            if persistence is None:
                if instance is not None and router_model is None:
                    try:
                        await cls._close_model_client(instance.router_model)
                    except Exception:
                        logger.debug("[main agent]释放路由模型客户端失败", exc_info=True)
                try:
                    await backends.close()
                except Exception:
                    logger.debug("[main agent]关闭持久化后端失败", exc_info=True)
            raise
        return instance

    @staticmethod
    async def _close_model_client(model: Any) -> None:
        client = getattr(model, "root_async_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            await close()

    async def close(self) -> None:
        try:
            await self._close_model_client(self.router_model)
        finally:
            await self._persistence.close()

    def _pending_review_namespace(self, route: str) -> tuple[str, ...]:
        return ("hitl_reviews", route)

    async def _get_pending_review(
        self, route: str, thread_id: str
    ) -> dict[str, Any] | None:
        item = await self.store.aget(self._pending_review_namespace(route), thread_id)
        return dict(item.value) if item and isinstance(item.value, dict) else None

    async def _save_pending_review(
        self, route: str, thread_id: str, value: dict[str, Any]
    ) -> None:
        await self.store.aput(self._pending_review_namespace(route), thread_id, value)

    async def _delete_pending_review(self, route: str, thread_id: str) -> None:
        await self.store.adelete(self._pending_review_namespace(route), thread_id)

    async def _update_pending_review(
        self, route: str, thread_id: str, pending: dict[str, Any], status: str
    ) -> dict[str, Any]:
        updated = {
            **pending,
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        await self._save_pending_review(route, thread_id, updated)
        return updated

    async def _complete_pending_review(
        self, route: str, thread_id: str, pending: dict[str, Any]
    ) -> None:
        completed = {
            **pending,
            "status": "completed",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        await self.store.aput(("hitl_history", route), thread_id, completed)
        await self._delete_pending_review(route, thread_id)

    async def _find_pending_route(self, main_thread_id: str) -> str | None:
        for route, suffix in (("purchase", "purchase"), ("after_sales", "after-sales")):
            pending = await self._get_pending_review(route, f"{main_thread_id}:{suffix}")
            if pending and pending.get("status") in {
                "awaiting_review",
                "executing",
                "failed",
            }:
                return route
        return None

    @staticmethod
    def _parse_ticket_review_decision(text: str) -> dict[str, Any] | None:
        normalized = re.sub(r"\s+", "", (text or "").lower())
        approve_patterns = (
            "确认创建工单",
            "同意创建工单",
            "可以创建工单",
            "确认建单",
            "同意建单",
            "确认退货申请",
            "同意退货申请",
            "可以退货申请",
            "确认执行",
            "同意执行",
        )
        reject_patterns = (
            "暂不创建工单",
            "不要创建工单",
            "先不创建工单",
            "暂不建单",
            "不要建单",
            "先不建单",
            "暂不退货申请",
            "不要退货申请",
            "先不退货申请",
            "暂不执行",
            "不要执行",
            "先不执行",
        )
        if any(p in normalized for p in approve_patterns):
            return {"type": "approve"}
        if any(p in normalized for p in reject_patterns):
            return {"type": "reject", "message": "用户暂不执行敏感售后操作，继续在线处理。"}
        return None

    @staticmethod
    def _build_review_state(
        action_requests: list[Any], sub_thread_id: str = ""
    ) -> dict[str, Any]:
        """将待审批工具调用保存为显式、可恢复的业务状态。"""

        actions = []
        for action in action_requests:
            if not isinstance(action, dict):
                continue
            name = str(action.get("name", "")).strip()
            args = action.get("args", {})
            if name:
                actions.append({"name": name, "args": args if isinstance(args, dict) else {}})
        return {
            "status": "awaiting_review",
            "count": len(actions) or 1,
            "tools": [action["name"] for action in actions],
            "actions": actions,
            "sub_thread_id": sub_thread_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _build_graph(self):
        builder = StateGraph(MainGraphState)

        builder.add_node("pending_gate", self._pending_gate_node)
        builder.add_node("analyze", self._analyze_node)
        builder.add_node("purchase", self._purchase_node)
        builder.add_node("after_sales", self._after_sales_node)
        builder.add_node("summarize", self._summarize_node)

        builder.add_edge(START, "pending_gate")
        builder.add_conditional_edges(
            "pending_gate",
            self._pending_gate_selector,
            {
                "analyze": "analyze",
                "purchase": "purchase",
                "after_sales": "after_sales",
            },
        )
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

        return builder.compile(checkpointer=self.checkpointer, store=self.store)

    async def _pending_gate_node(
        self, state: MainGraphState, config: RunnableConfig
    ) -> dict[str, str]:
        self._node_log("pending_gate", "检查待审批业务动作", state)
        main_thread_id = str(
            config.get("configurable", {}).get("thread_id", "default")
        )
        route = await self._find_pending_route(main_thread_id)
        return {"route": route or "analyze"}

    @staticmethod
    def _pending_gate_selector(state: MainGraphState) -> str:
        route = state.get("route", "analyze")
        return route if route in {"purchase", "after_sales"} else "analyze"

    @staticmethod
    def _node_log(node: str, action: str, state: MainGraphState | None = None) -> None:
        msg_count = 0
        if isinstance(state, dict):
            msgs = state.get("recent_messages", [])
            if isinstance(msgs, list):
                msg_count = len(msgs)
        logger.info(
            f"[graph node]当前节点：{node}，即将执行：{action}，"
            f"state.recent_messages数量：{msg_count}"
        )

    async def load_user_memory_summary(
        self, user_id: str = "default_user", query: str = ""
    ) -> str:
        """按当前问题检索少量相关画像和历史事件。"""
        if self.store is None:
            return ""
        memories = await memory_utils.retrieve_relevant_memories(
            self.store, user_id, query, top_k=5
        )
        return memory_utils.format_memory_context(memories)

    async def list_user_memories(self, user_id: str) -> dict[str, list[dict]]:
        return await memory_utils.list_user_memories(self.store, user_id)

    async def clear_user_memories(self, user_id: str) -> None:
        await memory_utils.clear_user_memories(self.store, user_id)

    def _build_messages(self, state: MainGraphState) -> list[BaseMessage]:
        summary = state.get("summary", "").strip()
        messages = list(state.get("recent_messages", []))
        prefixes: list[BaseMessage] = []
        if summary:
            prefixes.append(SystemMessage(content=f"历史摘要：{summary}"))
        return [*prefixes, *messages]

    async def _analyze_node(self, state: MainGraphState):
        self._node_log("analyze", "调用路由模型判断用户意图", state)
        query = self._build_messages(state)
        system_prompt = (
            "你是客服路由分析器。请判断用户意图属于以下二选一：\n"
            "1) purchase: 选购/推荐/对比/预算/信息匹配\n"
            "2) after_sales: 故障排查/维修/售后/保养/报告查询\n\n"
            "如果用户语义不清、信息不足以判断，输出 unclear。\n"
            "只输出一个标签：purchase、after_sales 或 unclear，不要输出其他内容。"
        )
        result = await self.router_model.ainvoke(
            [SystemMessage(content=system_prompt), *query],
            config={"tags": ["router"]},
        )
        text = (result.content or "").strip().lower()

        # 防御模型输出额外解释：从文本中提取合法标签
        match = re.search(r"\b(purchase|after_sales|unclear)\b", text)
        route = match.group(1) if match else "unclear"
        # logger.info(f"[graph node]节点analyze执行完成，路由结果：{route}")
        if route == "unclear":
            return {
                "route": "unclear",
                "response": "我还不太确定你的诉求是选购还是售后。请补充一下你的目标或问题细节。",
            }
        return {"route": route}

    def _route_selector(self, state: MainGraphState) -> str:
        route = state.get("route", "unclear")
        return route if route in {"purchase", "after_sales", "unclear"} else "unclear"

    async def _cleanup_child_thread(self, sub_thread_id: str) -> None:
        """删除已完成轮次的子线程 checkpoint，避免子线程数据随会话无限累积。"""
        try:
            await self.checkpointer.adelete_thread(sub_thread_id)
        except Exception as e:
            logger.debug(f"[graph node]清理子线程checkpoint失败(可忽略)：{sub_thread_id} {e}")

    async def _purchase_node(self, state: MainGraphState, config: RunnableConfig):
        self._node_log("purchase", "调用PurchaseAgent处理选购咨询", state)
        configurable = config.get("configurable", {})
        main_thread_id = configurable.get("thread_id", "default")
        user_id = configurable.get("user_id", "default_user")
        # 审批记录使用稳定键；子线程每轮全新，避免子 agent 历史跨轮重复累积
        review_key = f"{main_thread_id}:purchase"
        pending = await self._get_pending_review("purchase", review_key)
        resuming_pending = False
        if pending:
            # 审批恢复必须回到中断时的子线程；旧记录无此字段时回退到稳定键
            sub_thread_id = str(pending.get("sub_thread_id") or "") or review_key
            latest_user_text = ""
            for msg in reversed(state.get("recent_messages", [])):
                if isinstance(msg, HumanMessage):
                    latest_user_text = str(getattr(msg, "content", "") or "")
                    break
            decision = self._parse_ticket_review_decision(latest_user_text)
            if decision is None:
                logger.info(
                    "[hitl] 审批回复无效，thread_id=%s，input_length=%s",
                    sub_thread_id,
                    len(latest_user_text.strip()),
                )
                return {
                    "response": "当前有待确认的购买操作。请回复“确认执行”或“暂不执行”。"
                }
            logger.info(
                "[hitl] 收到审批决定，thread_id=%s，decision=%s，tools=%s",
                sub_thread_id,
                decision.get("type"),
                pending.get("tools", []),
            )
            review_count = int(pending.get("count", 1) or 1)
            payload: Command | dict = Command(
                resume={"decisions": [decision for _ in range(review_count)]}
            )
            pending = await self._update_pending_review(
                "purchase", review_key, pending, "executing"
            )
            resuming_pending = True
        else:
            sub_thread_id = f"{review_key}:{uuid4().hex[:8]}"
            payload = {"messages": self._build_messages(state)}

        child_config = dict(config)
        child_config["tags"] = [*config.get("tags", []), "user-response"]
        child_config["configurable"] = {
            **configurable,
            "thread_id": sub_thread_id,
            "user_id": user_id,
        }
        try:
            result = await self.purchase_agent.ainvoke(
                payload,
                context={"route": "purchase", "report": False},
                config=child_config,
            )
        except Exception:
            if resuming_pending and pending:
                await self._update_pending_review(
                    "purchase", review_key, pending, "failed"
                )
            raise

        interrupts = result.get("__interrupt__")
        if interrupts:
            first_interrupt = interrupts[0]
            payload = getattr(first_interrupt, "value", None)
            if isinstance(payload, dict):
                action_requests = payload.get("action_requests", [])
                review_state = self._build_review_state(action_requests, sub_thread_id)
                tool_names = review_state["tools"]
                await self._save_pending_review(
                    "purchase",
                    review_key,
                    review_state,
                )
                logger.info(
                    "[hitl] 触发人工审批，thread_id=%s，count=%s，tools=%s",
                    sub_thread_id,
                    len(action_requests) or 1,
                    tool_names,
                )
            return {
                "response": "检测到敏感购买操作需要人工确认。请回复“确认执行”或“暂不执行”。"
            }

        reply = next(
            (
                msg.content
                for msg in reversed(result["messages"])
                if isinstance(msg, AIMessage)
            ),
            "",
        )
        # logger.info(f"[graph node]节点purchase执行完成，回复长度：{len(str(reply))}")
        ai_msg = AIMessage(content=reply)
        if resuming_pending and pending:
            await self._complete_pending_review("purchase", review_key, pending)
        await self._cleanup_child_thread(sub_thread_id)
        return {
            "recent_messages": [ai_msg],
            "response": reply,
        }

    async def _after_sales_node(self, state: MainGraphState, config: RunnableConfig):
        self._node_log("after_sales", "调用AfterSalesAgent处理售后咨询", state)
        configurable = config.get("configurable", {})
        main_thread_id = configurable.get("thread_id", "default")
        user_id = configurable.get("user_id", "default_user")
        # 审批记录使用稳定键；子线程每轮全新，避免子 agent 历史跨轮重复累积
        review_key = f"{main_thread_id}:after-sales"
        pending = await self._get_pending_review("after_sales", review_key)
        resuming_pending = False
        if pending:
            # 审批恢复必须回到中断时的子线程；旧记录无此字段时回退到稳定键
            sub_thread_id = str(pending.get("sub_thread_id") or "") or review_key
            latest_user_text = ""
            for msg in reversed(state.get("recent_messages", [])):
                if isinstance(msg, HumanMessage):
                    latest_user_text = str(getattr(msg, "content", "") or "")
                    break
            decision = self._parse_ticket_review_decision(latest_user_text)
            if decision is None:
                logger.info(
                    "[hitl] 审批回复无效，thread_id=%s，input_length=%s",
                    sub_thread_id,
                    len(latest_user_text.strip()),
                )
                return {
                    "response": "当前有待确认的售后操作。请回复“确认执行”或“暂不执行”。"
                }
            logger.info(
                "[hitl] 收到审批决定，thread_id=%s，decision=%s，tools=%s",
                sub_thread_id,
                decision.get("type"),
                pending.get("tools", []),
            )
            review_count = int(pending.get("count", 1) or 1)
            payload: Command | dict = Command(
                resume={"decisions": [decision for _ in range(review_count)]}
            )
            pending = await self._update_pending_review(
                "after_sales", review_key, pending, "executing"
            )
            resuming_pending = True
        else:
            sub_thread_id = f"{review_key}:{uuid4().hex[:8]}"
            payload = {"messages": self._build_messages(state)}

        child_config = dict(config)
        child_config["tags"] = [*config.get("tags", []), "user-response"]
        child_config["configurable"] = {
            **configurable,
            "thread_id": sub_thread_id,
            "user_id": user_id,
        }
        try:
            result = await self.after_sales_agent.ainvoke(
                payload,
                context={"route": "after_sales", "report": False},
                config=child_config,
            )
        except Exception:
            if resuming_pending and pending:
                await self._update_pending_review(
                    "after_sales", review_key, pending, "failed"
                )
            raise

        interrupts = result.get("__interrupt__")
        if interrupts:
            first_interrupt = interrupts[0]
            payload = getattr(first_interrupt, "value", None)
            if isinstance(payload, dict):
                action_requests = payload.get("action_requests", [])
                review_state = self._build_review_state(action_requests, sub_thread_id)
                tool_names = review_state["tools"]
                await self._save_pending_review(
                    "after_sales",
                    review_key,
                    review_state,
                )
                logger.info(
                    "[hitl] 触发人工审批，thread_id=%s，count=%s，tools=%s",
                    sub_thread_id,
                    len(action_requests) or 1,
                    tool_names,
                )
            return {
                "response": "检测到敏感售后操作需要人工确认。请回复“确认执行”或“暂不执行”。"
            }

        reply = next(
            (
                msg.content
                for msg in reversed(result["messages"])
                if isinstance(msg, AIMessage)
            ),
            "",
        )
        # logger.info(f"[graph node]节点after_sales执行完成，回复长度：{len(str(reply))}")
        ai_msg = AIMessage(content=reply)
        if resuming_pending and pending:
            await self._complete_pending_review("after_sales", review_key, pending)
        await self._cleanup_child_thread(sub_thread_id)
        return {
            "recent_messages": [ai_msg],
            "response": reply,
        }

    async def _summarize_node(
        self,
        state: MainGraphState,
        config: RunnableConfig,
    ):
        self._node_log("summarize", "执行会话摘要压缩与长期记忆增量整理", state)
        messages = state.get("recent_messages", [])
        summary = state.get("summary", "")

        # 只有超过窗口长度才压缩，恰好等于阈值时 old_messages 为空，
        # 继续执行会用空增量调用模型并覆盖已有摘要
        if len(messages) <= self.MAX_RECENT_MESSAGES:
            logger.info(
                f"[graph node]节点summarize跳过，当前消息数{len(messages)}"
                f"未超过阈值{self.MAX_RECENT_MESSAGES}"
            )
            return {}

        old_messages = messages[: -self.MAX_RECENT_MESSAGES]
        recent_messages = messages[-self.MAX_RECENT_MESSAGES :]
        updates: dict = {}

        new_summary = await memory_utils.merge_summary(
            self.router_model, summary, old_messages
        )
        updates["summary"] = new_summary
        updates["recent_messages"] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent_messages,
        ]
        logger.info(
            f"[graph node]节点summarize执行完成，历史消息裁剪为最近{len(recent_messages)}条"
        )

        return updates

    async def finalize_thread(
        self,
        thread_id: str,
        user_id: str = "default_user",
        recent_messages: list[BaseMessage] | None = None,
    ) -> bool:
        """在线程结束时写入业务事件并更新结构化用户画像。"""
        if self.store is None:
            return False

        normalized_thread_id = (thread_id or "default").strip() or "default"
        tail_messages = [
            msg for msg in (recent_messages or []) if isinstance(msg, BaseMessage)
        ]
        if not tail_messages:
            return False
        checkpoint = await self.checkpointer.aget(
            {"configurable": {"thread_id": normalized_thread_id}}
        )
        route = "unknown"
        if checkpoint:
            route = str(checkpoint.get("channel_values", {}).get("route", "unknown"))
        return await memory_utils.save_memory_episode(
            self.store,
            self.router_model,
            user_id,
            normalized_thread_id,
            tail_messages,
            route,
        )

    async def execute_stream(
        self,
        query: str,
        thread_id: str,
        user_id: str = "default_user",
        bootstrap_summary: str | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        streamed_text = ""
        final_response = ""
        user_msg = HumanMessage(content=query)
        input_payload: dict = {
            "recent_messages": [user_msg],
            "response": "",
        }
        if bootstrap_summary is not None:
            input_payload["summary"] = bootstrap_summary

        async for mode, data in self.graph.astream(
            # 重置 response，避免新一轮开始时复用上轮持久化状态里的旧回答。
            input_payload,
            stream_mode=["messages", "updates", "values"],
            config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
        ):
            if mode == "messages":
                chunk, metadata = data
                tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
                content = getattr(chunk, "content", "")
                tool_chunks = getattr(chunk, "tool_call_chunks", []) or []
                if tool_chunks and event_callback:
                    names = [item.get("name") for item in tool_chunks if item.get("name")]
                    if names:
                        event_callback({"type": "tools", "names": names})
                if (
                    "user-response" in tags
                    and not tool_chunks
                    and isinstance(content, str)
                    and content
                ):
                    streamed_text += content
                    yield content
            elif mode == "updates" and isinstance(data, dict):
                if event_callback:
                    event_callback({"type": "nodes", "names": list(data)})
            elif mode == "values" and isinstance(data, dict):
                final_response = str(data.get("response", "") or "").strip()

        if final_response and final_response != streamed_text:
            if final_response.startswith(streamed_text):
                remainder = final_response[len(streamed_text) :]
                if remainder:
                    yield remainder
            elif not streamed_text:
                yield final_response
