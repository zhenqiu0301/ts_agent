"""FastAPI 服务端入口：承载共享 MainGraphAgent 并暴露 SSE 流式 API。

启动方式：

    uv run uvicorn ts_agent.server.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ts_agent.agents.main_graph_agent import MainGraphAgent
from ts_agent.server.routes import router
from ts_agent.utils.logger_handler import enable_file_logging, logger

enable_file_logging()

_DEFAULT_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"


def _allowed_origins() -> list[str]:
    raw = os.getenv("TS_AGENT_ALLOWED_ORIGINS", "").strip()
    if not raw:
        raw = _DEFAULT_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(agent: Any = None) -> FastAPI:
    """构建 FastAPI 应用。

    注入 agent 时其生命周期归调用方所有（用于测试）；
    不注入时由 lifespan 自建共享 Agent 并在退出时关闭。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_agent = agent is None
        app.state.agent = agent
        if owns_agent:
            app.state.agent = await MainGraphAgent.create()
            logger.info("[server]共享 MainGraphAgent 已创建")
        try:
            yield
        finally:
            if owns_agent:
                await app.state.agent.close()
                logger.info("[server]共享 MainGraphAgent 已关闭")

    application = FastAPI(title="TS Agent 智能客服 API", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
