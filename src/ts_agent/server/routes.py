"""Agent 服务的 HTTP API 路由。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from ts_agent.server.schemas import (
    ChatRequest,
    ClearMemoryResponse,
    FinalizeRequest,
    FinalizeResponse,
    HealthResponse,
)
from ts_agent.utils.logger_handler import logger

router = APIRouter(prefix="/api")


def _agent(request: Request) -> Any:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 未就绪")
    return agent


def _sse_event(kind: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": kind, "data": json.dumps(data, ensure_ascii=False)}


def _drain_callback_events(pending: list[dict[str, Any]]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    while pending:
        event = pending.pop(0)
        kind = "tool" if event.get("type") == "tools" else "node"
        events.append(_sse_event(kind, {"names": event.get("names", [])}))
    return events


async def _first_turn_bootstrap(
    agent: Any, thread_id: str, user_id: str, message: str
) -> str | None:
    """仅在线程首轮（无 checkpoint）时检索长期记忆摘要作为对话引导。"""

    checkpoint = await agent.checkpointer.aget({"configurable": {"thread_id": thread_id}})
    if checkpoint is not None:
        return None
    return await agent.load_user_memory_summary(user_id, message)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> EventSourceResponse:
    agent = _agent(request)

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        pending: list[dict[str, Any]] = []

        def on_event(event: dict[str, Any]) -> None:
            pending.append(event)

        chunks: list[str] = []
        try:
            bootstrap = await _first_turn_bootstrap(
                agent, payload.thread_id, payload.user_id, payload.message
            )
            async for delta in agent.execute_stream(
                payload.message,
                payload.thread_id,
                payload.user_id,
                bootstrap,
                event_callback=on_event,
            ):
                # 回调事件先于其后的 chunk 触发，按到达顺序下发
                for event in _drain_callback_events(pending):
                    yield event
                if delta:
                    chunks.append(delta)
                    yield _sse_event("chunk", {"delta": delta})
            for event in _drain_callback_events(pending):
                yield event
            yield _sse_event("done", {"response": "".join(chunks)})
        except Exception as e:
            logger.error(f"[api]处理对话失败: {e}", exc_info=True)
            yield _sse_event("error", {"message": "服务端处理消息失败，请稍后重试。"})

    return EventSourceResponse(event_stream())


@router.get("/memory/{user_id}")
async def list_memory(user_id: str, request: Request) -> dict:
    agent = _agent(request)
    return await agent.list_user_memories(user_id)


@router.delete("/memory/{user_id}", response_model=ClearMemoryResponse)
async def clear_memory(user_id: str, request: Request) -> ClearMemoryResponse:
    agent = _agent(request)
    await agent.clear_user_memories(user_id)
    return ClearMemoryResponse(status="cleared")


@router.post("/session/finalize", response_model=FinalizeResponse)
async def finalize_session(
    payload: FinalizeRequest, request: Request
) -> FinalizeResponse:
    agent = _agent(request)
    window = int(getattr(agent, "MAX_RECENT_MESSAGES", 10) or 10)
    recent_messages: list[BaseMessage] = []
    for item in payload.messages[-window:]:
        content = item.content.strip()
        if not content:
            continue
        if item.role == "user":
            recent_messages.append(HumanMessage(content=content))
        elif item.role == "assistant":
            recent_messages.append(AIMessage(content=content))
    changed = await agent.finalize_thread(
        payload.thread_id,
        payload.user_id,
        recent_messages=recent_messages,
    )
    return FinalizeResponse(changed=changed)
