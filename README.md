# TS Agent 智能客服

基于 LangGraph、DeepSeek、DashScope Embedding、Chroma 的智能客服示例，包含选购咨询、售后处理、RAG 知识检索和长期记忆。
项目采用服务端/客户端架构：FastAPI 服务端持有共享 Agent 并通过 SSE 流式推送回答与进度事件，Streamlit 作为客户端负责渲染。
模型、子 Agent、MCP 工具、流式输出和 SQLite 持久化使用原生异步调用链。
回答支持 Token 级流式展示和节点/工具进度事件；外部比价 MCP 仅在真正调用价格工具时延迟连接。
售后工单创建具有参数校验和线程级幂等写入保护。

## 架构

```text
浏览器 ──► Streamlit 客户端(8501) ──HTTP/SSE──► FastAPI 服务端(8000)
                                              ├─ 共享 MainGraphAgent
                                              ├─ SQLite checkpoint/store
                                              ├─ Chroma 向量库
                                              └─ 外部 MCP（延迟连接）
```

## 环境要求

- Python 3.10+
- `uv`
- Node.js 16+（默认的淘客 MCP 工具需要）
- DeepSeek、DashScope 与 Tavily API Key

## 本地启动

```bash
cp .env.example .env
# 编辑 .env，填入本地密钥
uv sync

# 终端 1：启动 API 服务端（默认 127.0.0.1:8000）
./start_server.sh

# 终端 2：启动 Streamlit 客户端（默认 8501）
./start_client.sh
```

服务端地址与端口通过 `TS_AGENT_API_HOST` / `TS_AGENT_API_PORT` 覆盖；客户端要连接的服务端地址通过 `TS_AGENT_API_URL` 覆盖（未设置时按同一组 host/port 变量拼出默认值）。

## HTTP API

服务端启动后可访问 `http://127.0.0.1:8000/docs` 查看交互式文档。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/chat` | 对话（SSE 流式：chunk/node/tool/done/error 事件） |
| GET | `/api/memory/{user_id}` | 查看用户长期记忆 |
| DELETE | `/api/memory/{user_id}` | 清除用户长期记忆 |
| POST | `/api/session/finalize` | 结束会话并整理长期记忆 |

客户端通过 `TS_AGENT_API_URL` 指定服务端地址（默认 `http://127.0.0.1:8000`）；服务端 CORS 白名单通过 `TS_AGENT_ALLOWED_ORIGINS` 配置（默认仅允许本机 Streamlit 来源）。

`.env`、虚拟环境、日志、Chroma 索引、SQLite 记忆和业务 JSONL 均为本地运行数据，不会提交到 Git。

## 外部 MCP

项目默认会通过 Node.js 启动 `src/ts_agent/vendor/taoke-mcp-main/dist/cli.js`，配置位于
`config/mcp.yml`。仓库保留 `dist/` 作为可直接运行的第三方构建产物；更新该目录时，应同时核对
`package.json`、`package-lock.json` 和上游版本。

首次使用前需要安装 MCP 的 Node 依赖（`dotenv` 等，该目录不随 Git 提交）：

```bash
cd src/ts_agent/vendor/taoke-mcp-main && npm install --omit=dev
```

MCP 服务端暴露约 31 个工具，项目通过白名单只加载比价所需的 `jd.goods.query` 与
`pdd.goods.search`（可用 `MCP_PRICE_COMPARE_TOOL_WHITELIST` 覆盖）。无需账号凭证即可
使用拼多多侧工具；京东侧工具需要 在 `config/mcp.yml` 填入 `JD_KEY`/`JD_PID`，否则该平台
返回不可用，模型会按提示词规则以 `web_search` 兜底。

如果本地不需要外部 MCP，可在启动前禁用：

```bash
MCP_DISABLE_EXTERNAL=1 ./start_server.sh
```

## 主要结构

```text
src/ts_agent/          主图、子 Agent、RAG、工具、Prompt 和内置 MCP
src/ts_agent/server/   FastAPI 服务端（SSE 流式 API）
config/                模型、RAG、Prompt 路径和 MCP 外部配置
data/                  知识文档与本地运行数据
evals/                 100 条确定性 benchmark（路由/工具纪律/售后流程/HITL/安全）
sft/                   智谱 GLM 采样的 SFT 对话数据集（与 benchmark 不重叠）
tests/                 不依赖真实模型请求的测试
app.py                 Streamlit 客户端入口
```

## 数据资产：benchmark 与 SFT

- `evals/`：100 条端到端确定性评测（意图路由、比价工具纪律、排障、建单 HITL、
  安全与健壮性），运行方式见 `evals/README.md`；换模型、改提示词前后各跑一遍做对比。
- `sft/`：用智谱 GLM 采样的对话 SFT 数据集，用户输入经指纹去重保证与 benchmark
  不重叠；system 提示词直接复用线上 prompt。构建方式见 `sft/README.md`。

## 工作流与记忆

- 主图按意图路由到对应子 Agent，每轮使用全新子线程，工具中间产物不污染主会话。
- 比价和使用报告由确定性组合工具保证调用顺序，不依赖模型自行编排关键步骤。
- 长期记忆分为结构化用户画像和业务事件，包含来源、可信度、敏感级别和有效期。
- 长期记忆按用户 ID 隔离；在客户端侧边栏修改"用户 ID"即可切换身份（会自动开启新会话）。
- 新对话按当前问题检索 Top-5 相关记忆；用户可在侧边栏查看或清除长期记忆。

## 知识库索引

同步新增、修改和删除的知识文件：

```bash
uv run python -m ts_agent.rag.vector_store
```

首次启用索引清单或更换 Embedding 模型后执行完整重建：

```bash
uv run python -m ts_agent.rag.vector_store --rebuild
```

重建会将 `data/raw` 中知识文件的文本分片发送到 DashScope Embedding API。
执行前请确认这些文件允许上传到第三方服务。

## 测试

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv lock --check
```

CI 会在 Python 3.10、3.11 和 3.12 上运行同样的检查。

## 安全说明

- 不要提交 `.env` 或真实 API Key。
- 工具日志默认只记录参数字段名，不记录手机号、姓名或地址。
- 设置 `TS_LOG_MESSAGE_CONTENT=1` 才会记录模型输入预览，仅建议本地调试使用。
- 工单、长期记忆和向量索引默认保存在 `data/`，属于本地数据。
