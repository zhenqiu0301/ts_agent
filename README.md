# TS Agent 智能客服

基于 LangGraph、DeepSeek、DashScope Embedding、Chroma 和 Streamlit 的智能客服示例，包含选购咨询、售后处理、RAG 知识检索、人工审批和长期记忆。
模型、子 Agent、MCP 工具、流式输出和 SQLite 持久化使用原生异步调用链。
回答支持 Token 级流式展示和节点/工具进度事件；外部比价 MCP 仅在真正调用价格工具时延迟连接。
下单、售后工单和退货申请具有参数校验、显式审批状态和线程级幂等写入保护。

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
./start.sh
```

`.env`、虚拟环境、日志、Chroma 索引、SQLite 记忆和业务 JSONL 均为本地运行数据，不会提交到 Git。

## 外部 MCP

项目默认会通过 Node.js 启动 `src/ts_agent/vendor/taoke-mcp-main/dist/cli.js`，配置位于
`config/mcp.yml`。仓库保留 `dist/` 作为可直接运行的第三方构建产物；更新该目录时，应同时核对
`package.json`、`package-lock.json` 和上游版本。

如果本地不需要外部 MCP，可在启动前禁用：

```bash
MCP_DISABLE_EXTERNAL=1 ./start.sh
```

## 主要结构

```text
src/ts_agent/  主图、子 Agent、RAG、工具、Prompt 和内置 MCP
config/        模型、RAG、Prompt 路径和 MCP 外部配置
data/          知识文档与本地运行数据
tests/         不依赖真实模型请求的测试
app.py         Streamlit 应用入口
```

## 工作流与记忆

- 主图在意图识别前先检查待审批动作，“确认执行”会直接恢复正确的子 Agent。
- 审批动作使用 `awaiting_review -> executing -> completed/failed` 状态机，失败时保留恢复信息。
- 比价和使用报告由确定性组合工具保证调用顺序，不依赖模型自行编排关键步骤。
- 长期记忆分为结构化用户画像和业务事件，包含来源、可信度、敏感级别和有效期。
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
- 工单、订单、退货、长期记忆和向量索引默认保存在 `data/`，属于本地数据。
