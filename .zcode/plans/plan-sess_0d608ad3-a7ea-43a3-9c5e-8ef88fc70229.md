将项目改造为服务端/客户端架构：新增 FastAPI 服务端承载共享 Agent，SSE 流式推送；Streamlit 保留为客户端，内部从直调 Agent 改为 HTTP 调用 API。

## 总体架构

```
浏览器 ──► Streamlit 客户端(8501) ──HTTP/SSE──► FastAPI 服务端(8000)
                                              ├─ 共享 MainGraphAgent（进程内单例）
                                              ├─ SQLite checkpoint/store、Chroma、MCP
                                              └─ GET /docs 交互式 API 文档
```

服务端持有共享 `MainGraphAgent`（lifespan 创建/关闭），所有请求复用——uvicorn 单事件循环，彻底消除此前 Streamlit 每消息新建循环带来的跨循环隐患；客户端不再 import 任何 Agent 代码。

## 新增：`src/ts_agent/server/` 包

### `schemas.py`（Pydantic 模型）
- `ChatRequest{thread_id, user_id="default_user", message}`
- `MessageInput{role, content}`、`FinalizeRequest{thread_id, user_id, messages}`
- `FinalizeResponse{changed}`、`HealthResponse{status}`

### `routes.py`（APIRouter，agent 取自 `request.app.state.agent`）
- `GET /api/health` → `{"status": "ok"}`
- `POST /api/chat` → SSE 流（`sse-starlette` 的 `EventSourceResponse`，自动 keepalive ping、断连取消）。服务端职责：
  - **bootstrap 判定**：`agent.checkpointer.aget(thread_id)` 无 checkpoint（首轮）→ 服务端调 `load_user_memory_summary(user_id, message)` 作为 bootstrap_summary 传入 `execute_stream`；非首轮传 None。客户端不再负责。
  - 事件映射：`execute_stream` 的 chunk → `event: chunk {"delta"}`；event_callback 的 nodes/tools → `event: node/tool {"names"}`；结束 → `event: done {"response": 全文}`；异常 → `event: error {"message"}` + logger.error（对应现在 app.py 的兜底行为）。
- `GET /api/memory/{user_id}` → `{profile, episodes}`
- `DELETE /api/memory/{user_id}` → `{"status": "cleared"}`
- `POST /api/session/finalize` → 服务端完成 role→HumanMessage/AIMessage 映射、按 `agent.MAX_RECENT_MESSAGES` 截窗（窗口逻辑从客户端移到服务端），调 `finalize_thread` → `{"changed": bool}`

### `app.py`
- 模块级 `enable_file_logging()`（沿用现有日志约定）
- `create_app(agent: MainGraphAgent | None = None)` 工厂：lifespan 中未注入时自建 agent、关闭时只关闭自建的（注入的归调用方，与 MainGraphAgent.create 的所有权规则一致）；CORS 中间件（origins 来自 `TS_AGENT_ALLOWED_ORIGINS`，默认 localhost:8501，为将来浏览器客户端预留）
- 模块级 `app = create_app()` 作为 uvicorn 入口：`uv run uvicorn ts_agent.server.app:app`

## 修改：`app.py`（Streamlit 客户端）
- 删除 `MainGraphAgent` 导入与所有 async generator（`capture/load_memories/clear_memories/finalize_session`）、bootstrap 相关 session_state——全部换成同步 httpx 调用。
- 新增 `stream_chat()` 同步生成器：POST `/api/chat` 流式读取并解析 SSE 事件行（httpx 已是直接依赖）。
- API 地址：`TS_AGENT_API_URL` 环境变量，默认 `http://127.0.0.1:8000`；首屏做一次 `/api/health` 探测，不可达时 st.warning 提示先启动服务端。
- 聊天流程：`write_stream(stream_chat 渲染生成器)`，node/tool 事件沿用现有进度文案映射；error 事件渲染友好提示；done 事件兜底全文。
- 侧边栏三按钮改为 GET/DELETE/POST 调用，httpx 异常兜底为友好提示；finalize_notice 机制、thread_id 显示、记忆 JSON 面板等交互保持不变。

## 修改：`pyproject.toml` + `uv.lock`
- dependencies 增加 `sse-starlette>=3,<4`（已随锁文件安装 3.4.5，声明为直接依赖）；执行 `uv lock` 刷新（CI 有 `uv lock --check`）。若锁更新失败则回退为手写 SSE（StreamingResponse + 手工编码，不加依赖）。

## 修改：`start.sh`
- 后台启动 uvicorn（127.0.0.1:8000，端口可用 `TS_AGENT_API_PORT` 覆盖），轮询 `/api/health` 就绪后前台启动 streamlit；`trap` EXIT 清理服务端进程；`MCP_DISABLE_EXTERNAL` 等环境变量透传。

## 修改：`README.md` / `.env.example`
- 架构图与启动说明（两个进程）、API 端点一览与 `/docs` 说明、`TS_AGENT_API_URL` 注释。

## 新增：`tests/test_server_api.py`
注入 FakeAgent（实现 execute_stream/load_user_memory_summary/list_user_memories/clear_user_memories/finalize_thread/checkpointer/MAX_RECENT_MESSAGES）+ `create_app(agent=fake)`，用 `fastapi.testclient.TestClient` 的 `stream()` 解析 SSE：
1. health 正常；
2. chat SSE：chunk 拼接 == done 全文、node/tool 事件透传；
3. bootstrap 仅首轮触发（FakeCheckpointer.aget 首次 None、二次返回对象）；
4. execute_stream 抛异常 → 收到 error 事件且流正常结束；
5. memory GET/DELETE 命中 FakeAgent；
6. finalize：role 映射正确、窗口截取最后 10 条、返回 changed。

## 验证
1. `MCP_DISABLE_EXTERNAL=1 .venv/bin/python -m unittest discover -s tests`（现有 31 个 + 新增 server 测试全绿）
2. `.venv/bin/python -m ruff check .`
3. `uv lock --check`
4. 真实启动冒烟：后台起 uvicorn → curl `/api/health` 与 `/docs` → 关闭；`streamlit run app.py --server.headless true` 短暂启动确认客户端无导入/渲染异常（不真实调用 LLM）。

## 不做的事
- 不引入鉴权/多租户（本地演示项目）；
- 不改 `MainGraphAgent` 及工具层任何语义——本次只是把它从"Streamlit 进程内"搬到"服务端进程内"；
- 不删除 Streamlit（作为客户端保留）；纯网页客户端将来可直接基于该 API 另行添加。