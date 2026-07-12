# TS Agent 智能客服

基于 LangGraph、DeepSeek、DashScope Embedding、Chroma 和 Streamlit 的智能客服示例，包含选购咨询、售后处理、RAG 知识检索、人工审批和长期记忆。
模型、子 Agent、MCP 工具、流式输出和 SQLite 持久化使用原生异步调用链。

## 环境要求

- Python 3.10+
- `uv`
- Node.js 16+ï¼默认的淘客 MCP 工具需要ï¼
- DeepSeek、DashScope 与 Tavily API Key

## 本地启动

```bash
cp .env.example .env
# 编辑 .env，填入本地密钥
uv sync
./start.sh
```

`.env`、虚拟环境、日志、Chroma 索引、SQLite 记忆和业务 JSONL 均为本地运行数据，不会提交到 Git。

## 外部 MCP

项目默认会通过 Node.js 启动 `taoke-mcp-main/dist/cli.js`，配置位于
`config/mcp.yml`。仓库保留 `dist/` 作为可直接运行的第三方构建产物；更新该目录时，应同时核对
`package.json`、`package-lock.json` 和上游版本。

如果本地不需要外部 MCP，可在启动前禁用：

```bash
MCP_DISABLE_EXTERNAL=1 ./start.sh
```

## 主要结构

```text
agents/      主图、子 Agent、人工审批和长期记忆
config/      模型、RAG、提示词和 MCP 配置
model/       DeepSeek 与 Embedding 延迟初始化
prompts/     系统与任务提示词
rag/         Chroma 检索与知识库索引
tools/       订单、售后、搜索和 MCP 工具
utils/       配置、路径、日志和文件加载
tests/       不依赖真实模型请求的基础测试
```

## 知识库索引

同步新增、修改和删除的知识文件：

```bash
uv run python -m rag.vector_store
```

首次启用索引清单或更换 Embedding 模型后执行完整重建：

```bash
uv run python -m rag.vector_store --rebuild
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
- 工单、订单、退货、长期记忆和向量索引默认保存在 `data/`，属于本地数据。
