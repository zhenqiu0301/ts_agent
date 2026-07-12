from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig


def get_user_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable", {})
    user_id = str(configurable.get("user_id", "")).strip()
    return user_id or "default_user"


def memory_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "profile")


THREAD_DELTA_SEP = "::delta::"


def delta_memory_key(thread_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    safe_thread = (thread_id or "default").strip() or "default"
    return f"{safe_thread}{THREAD_DELTA_SEP}{ts}:{uuid4().hex[:8]}"


def is_delta_memory_key(key: str) -> bool:
    return THREAD_DELTA_SEP in str(key or "")


def parse_thread_id_from_delta_key(key: str) -> str:
    text = str(key or "")
    if THREAD_DELTA_SEP not in text:
        return ""
    return text.split(THREAD_DELTA_SEP, 1)[0].strip()


def list_namespace_items(store: Any, namespace: tuple[str, ...], batch_size: int = 100):
    items = []
    offset = 0
    while True:
        chunk = store.search(namespace, limit=batch_size, offset=offset)
        if not chunk:
            break
        items.extend(chunk)
        if len(chunk) < batch_size:
            break
        offset += batch_size
    items.sort(key=lambda it: (str(getattr(it, "updated_at", "")), it.key))
    return items


def list_delta_items(store: Any, namespace: tuple[str, ...], batch_size: int = 100):
    items = list_namespace_items(store, namespace, batch_size=batch_size)
    return [it for it in items if is_delta_memory_key(it.key)]


def list_thread_delta_items(
    store: Any,
    namespace: tuple[str, ...],
    thread_id: str,
    batch_size: int = 100,
):
    wanted = (thread_id or "default").strip() or "default"
    items = list_delta_items(store, namespace, batch_size=batch_size)
    return [it for it in items if parse_thread_id_from_delta_key(it.key) == wanted]


def load_thread_messages(checkpointer: Any, thread_id: str) -> list[BaseMessage]:
    checkpoint_tuple = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
    if not checkpoint_tuple:
        return []
    values = checkpoint_tuple.checkpoint.get("channel_values", {})
    messages = values.get("recent_messages", [])
    if not isinstance(messages, list):
        return []
    return [msg for msg in messages if isinstance(msg, BaseMessage)]


def messages_to_plain_text(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def long_memory_to_summary(model: Any, long_memory: str) -> str:
    system_prompt = (
        "你是会话启动摘要助手。请将给定的长期记忆转换为本轮会话可直接使用的历史摘要。\n"
        "输出要求：100-200字中文，不分点，保留稳定约束与关键结论，不要编造。"
    )
    user_prompt = f"长期记忆：{long_memory}\n请输出可用于对话上下文的历史摘要："
    result = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    text = (result.content or "").strip()
    return text or long_memory


def summarize_long_memory_delta(model: Any, messages: list[BaseMessage]) -> str:
    conversation = messages_to_plain_text(messages)
    system_prompt = (
        "你是长期记忆增量提炼助手。只基于新增对话提炼稳定、可复用事实。\n"
        "保留：用户信息、偏好、预算、历史故障、购买/售后等关键结论。\n"
        "删除：寒暄、一次性情绪、与后续无关细节。\n"
        "输出要求：20-100字中文，不分点。若无新增稳定事实，输出“无”。"
    )
    user_prompt = f"新增对话：\n{conversation or '无'}\n请输出长期记忆增量："
    result = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    text = (result.content or "").strip()
    return "" if text == "无" else text


def compact_long_memory(model: Any, memory_text: str) -> str:
    if not memory_text:
        return ""
    system_prompt = (
        "你是长期记忆压缩整理助手。请在不丢失关键稳定事实的前提下，去重、合并、消歧。\n"
        "保留：预算、用户信息、偏好、历史问题结论、明确禁忌。\n"
        "删除：寒暄、一次性情绪、重复表达。\n"
        "输出要求：200-300字中文，不分点。"
    )
    user_prompt = f"待整理长期记忆：{memory_text}\n请输出整理后的长期记忆："
    result = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    text = (result.content or "").strip()
    return text or memory_text


def merge_summary(model: Any, old_summary: str, old_messages: list[BaseMessage]) -> str:
    conversation = messages_to_plain_text(old_messages)
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
    result = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    text = (result.content or "").strip()
    return text or old_summary
