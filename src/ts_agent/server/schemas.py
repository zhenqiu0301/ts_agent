"""服务端 API 的请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    user_id: str = Field(default="default_user", min_length=1)
    message: str = Field(min_length=1)


class MessageInput(BaseModel):
    role: str
    content: str


class FinalizeRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    user_id: str = Field(default="default_user", min_length=1)
    messages: list[MessageInput] = Field(default_factory=list)


class FinalizeResponse(BaseModel):
    changed: bool


class ClearMemoryResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
