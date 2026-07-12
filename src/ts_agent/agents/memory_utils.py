"""Structured user-profile and episodic-memory helpers."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig


def get_user_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable", {})
    return str(configurable.get("user_id", "")).strip() or "default_user"


def profile_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "profile_facts")


def episode_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "episodes")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _redact_sensitive(text: str) -> str:
    text = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1\d{10}(?!\d)", "[PHONE]", text)
    return re.sub(r"\b\d{15,18}[0-9Xx]\b", "[ID]", text)


def messages_to_plain_text(messages: list[BaseMessage], *, redact: bool = True) -> str:
    lines = []
    for message in messages:
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        content = str(message.content)
        lines.append(f"{role}: {_redact_sensitive(content) if redact else content}")
    return "\n".join(lines)


def _extract_profile_candidates(messages: list[BaseMessage], thread_id: str) -> list[dict]:
    text = "\n".join(
        str(message.content) for message in messages if isinstance(message, HumanMessage)
    )
    candidates: dict[str, Any] = {}
    area = re.search(r"(\d{2,3})\s*(?:㎡|平方米|平)", text)
    if area:
        candidates["home_area_sqm"] = int(area.group(1))
    budget = re.search(r"预算[^\d]{0,8}(\d{3,6})", text)
    if budget:
        candidates["budget_cny"] = int(budget.group(1))
    if any(word in text for word in ("养猫", "有猫", "猫毛", "养狗", "有狗", "宠物")):
        candidates["has_pets"] = True
    floor_types = [
        floor for floor in ("木地板", "瓷砖", "地毯", "复合地板", "大理石") if floor in text
    ]
    if floor_types:
        candidates["floor_types"] = floor_types

    message_ids = [str(message.id) for message in messages if getattr(message, "id", None)]
    facts = []
    for key, value in candidates.items():
        expires_at = None
        if key == "budget_cny":
            expires_at = (datetime.now().astimezone() + timedelta(days=90)).isoformat(
                timespec="seconds"
            )
        facts.append({
            "key": key,
            "value": value,
            "source_thread_id": thread_id,
            "source_message_ids": message_ids,
            "source": "explicit_user_message",
            "confidence": 1.0,
            "confirmed": True,
            "sensitivity": "normal",
            "updated_at": _now(),
            "expires_at": expires_at,
            "conflict": False,
        })
    return facts


async def save_memory_episode(
    store: Any,
    model: Any,
    user_id: str,
    thread_id: str,
    messages: list[BaseMessage],
    route: str = "unknown",
) -> bool:
    if not messages:
        return False
    facts = _extract_profile_candidates(messages, thread_id)
    summary = await summarize_long_memory_delta(model, messages)
    if summary:
        event_id = f"evt-{uuid4().hex}"
        await store.aput(
            episode_namespace(user_id),
            event_id,
            {
                "event_id": event_id,
                "event_type": f"conversation_{route}",
                "thread_id": thread_id,
                "summary": _redact_sensitive(summary),
                "source": "conversation_summary",
            "source_message_count": len(messages),
            "source_message_ids": [
                str(message.id) for message in messages if getattr(message, "id", None)
            ],
                "confidence": 0.8,
                "sensitivity": "normal",
                "created_at": _now(),
                "expires_at": (
                    datetime.now().astimezone() + timedelta(days=365)
                ).isoformat(timespec="seconds"),
            },
        )
    for fact in facts:
        namespace = profile_namespace(user_id)
        existing = await store.aget(namespace, fact["key"])
        if existing and existing.value.get("value") != fact["value"]:
            fact = {
                **fact,
                "confirmed": False,
                "confidence": 0.7,
                "conflict": True,
                "previous_value": existing.value.get("value"),
            }
            candidate_key = f"{fact['key']}::candidate::{uuid4().hex[:8]}"
            await store.aput(namespace, candidate_key, fact)
        else:
            await store.aput(namespace, fact["key"], fact)
    return bool(summary or facts)


def _query_terms(query: str) -> set[str]:
    normalized = re.sub(r"\s+", "", query.lower())
    return {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}


async def retrieve_relevant_memories(
    store: Any, user_id: str, query: str, *, top_k: int = 5
) -> list[dict]:
    profile_items = await store.asearch(profile_namespace(user_id), limit=100)
    episode_items = await store.asearch(episode_namespace(user_id), limit=100)
    terms = _query_terms(query)
    ranked = []
    now = datetime.now().astimezone()
    for item in [*profile_items, *episode_items]:
        value = dict(item.value) if isinstance(item.value, dict) else {}
        expires_at = value.get("expires_at")
        if expires_at:
            try:
                if datetime.fromisoformat(str(expires_at)) <= now:
                    continue
            except ValueError:
                pass
        searchable = str(value.get("value", "")) + str(value.get("summary", ""))
        overlap = len(terms & _query_terms(searchable)) if terms else 0
        kind_bonus = 3 if "value" in value else 0
        timestamp = str(value.get("updated_at") or value.get("created_at") or "")
        ranked.append((overlap + kind_bonus, timestamp, value))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [value for _, _, value in ranked[:top_k]]


def format_memory_context(memories: list[dict]) -> str:
    lines = []
    for memory in memories:
        if "key" in memory:
            prefix = "待确认画像" if memory.get("conflict") else "用户画像"
            lines.append(f"{prefix} {memory['key']}: {memory.get('value')}")
        elif memory.get("summary"):
            lines.append(f"历史事件: {memory['summary']}")
    return "\n".join(lines)


async def list_user_memories(store: Any, user_id: str) -> dict[str, list[dict]]:
    profiles = await store.asearch(profile_namespace(user_id), limit=100)
    episodes = await store.asearch(episode_namespace(user_id), limit=100)
    return {
        "profile": [dict(item.value) for item in profiles],
        "episodes": [dict(item.value) for item in episodes],
    }


async def clear_user_memories(store: Any, user_id: str) -> None:
    for namespace in (
        profile_namespace(user_id),
        episode_namespace(user_id),
        ("users", user_id, "profile"),
    ):
        for item in await store.asearch(namespace, limit=1000):
            await store.adelete(namespace, item.key)


async def summarize_long_memory_delta(model: Any, messages: list[BaseMessage]) -> str:
    conversation = messages_to_plain_text(messages)
    system_prompt = (
        "你是客服业务事件摘要器。仅保留用户目标、产品、故障、订单/工单结果和待跟进事项。\n"
        "禁止输出手机号、详细地址、身份证号。若无值得保留的业务事件，输出“无”。"
    )
    result = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=conversation or "无")],
        config={"tags": ["memory"]},
    )
    text = _redact_sensitive(str(result.content or "").strip())
    return "" if text == "无" else text


async def merge_summary(
    model: Any, old_summary: str, old_messages: list[BaseMessage]
) -> str:
    conversation = messages_to_plain_text(old_messages)
    result = await model.ainvoke(
        [
            SystemMessage(
                content="压缩当前会话工作记忆，保留目标、已收集字段和待完成任务，不保留手机和地址。"
            ),
            HumanMessage(content=f"旧摘要：{old_summary or '无'}\n新对话：{conversation}"),
        ],
        config={"tags": ["memory"]},
    )
    return _redact_sensitive(str(result.content or "").strip()) or old_summary
