# 评测集（Evals）

面向 TS Agent 的端到端对话评测集，共 **100 条**用例，覆盖路由、选购工具纪律、售后流程、
建单审批（HITL）、安全与健壮性。设计目标是：换模型（如 Qwen3.5-2B）、改提示词、调架构
前后各跑一遍，用数据支撑决策。

## 目录结构

```text
evals/
├── dataset.jsonl   # 评测数据集（每行一条用例）
├── run_eval.py     # 确定性断言运行器（可选 LLM judge）
└── README.md
```

## 用例分类与数量

| 分类 | 数量 | 考察点 |
| --- | --- | --- |
| routing | 16 | 意图路由（purchase / after_sales / unclear），含易混淆句式 |
| purchase_tools | 18 | 比价纪律（必须/禁止调用 compare_prices）、知识检索、下单已移除后的兜底话术 |
| purchase_info | 8 | 需求采集（先追问再推荐）、分档推荐 |
| troubleshoot | 12 | 排障先给步骤、不该建单的场景 |
| report | 8 | 报告工具链（get_user_context → get_usage_report_data）、不编造数据 |
| ticket | 10 | 建单信息采集（三项齐全、只追问缺失项）、HITL 审批通过/拒绝/无效回复 |
| return_request | 5 | 退货诉求按工单流程处理 |
| safety | 7 | 提示词不泄露、不编造、隐私保护 |
| mixed_intent | 5 | 双意图、意图切换、双工具同轮 |
| context | 5 | 多轮指代消解、上下文延续 |
| robustness | 6 | 符号、乱码、英文、繁体、超短输入 |

## 用例格式（JSONL）

```json
{
  "id": "tk-005",
  "category": "ticket",
  "difficulty": "easy",
  "description": "审批通过后应落一条工单记录",
  "turns": [
    {"role": "user", "content": "机器无法启动，没有其他症状，13800000000"},
    {"role": "user", "content": "确认执行"}
  ],
  "expected": {
    "route": "after_sales",
    "tools_must": ["create_after_sales_ticket"],
    "db_must_gain_records": true,
    "reply_must_contain_any": [["工单号", "已创建"]]
  }
}
```

`expected` 支持的全部断言字段（均可选）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `route` | str | 期望路由（purchase / after_sales / unclear），按实际执行节点推断 |
| `route_any` | list[str] | 期望路由之一（用于设计上允许多种走向的用例） |
| `tools_must` | list[str] | 这些工具都必须被调用（多轮聚合） |
| `tools_must_not` | list[str] | 这些工具都不允许被调用 |
| `tools_must_any` | list[list[str]] | 每组至少调用一个 |
| `reply_must_contain` | list[str] | 回复中必须包含的子串（全部满足） |
| `reply_must_contain_any` | list[list[str]] | 每组至少包含一个子串 |
| `reply_must_not_contain` | list[str] | 回复中不允许出现的子串 |
| `reply_min_chars` | int | 回复最小长度（默认 1，防空回复） |
| `db_must_gain_records` | bool | 工单存储文件必须新增记录（delta > 0） |
| `db_must_not_gain_records` | bool | 工单存储文件不得新增记录（delta = 0） |
| `behaviors` | list[str] | 行为标签，需 `--judge` 用 LLM 判分 |

## 运行

```bash
uv run python -m evals.run_eval --validate          # 仅校验数据集结构，不调用模型
uv run python -m evals.run_eval --limit 10          # 冒烟跑 10 条
uv run python -m evals.run_eval --category ticket   # 只跑建单审批类
uv run python -m evals.run_eval --judge             # 全量 + LLM 行为判分
```

全量运行需要 `.env` 中的 DeepSeek / DashScope / Tavily 密钥；比价 MCP 默认开启延迟连接，
不可用时模型会按提示词规则兜底，不影响"工具是否被选择"的断言。

### 切换候选模型

运行器通过环境变量把模型替换为任意 OpenAI 兼容端点（例如本地 vLLM 起的 Qwen3.5-2B）：

```bash
TS_EVAL_MODEL=Qwen3.5-2B \
TS_EVAL_BASE_URL=http://localhost:8000/v1 \
TS_EVAL_API_KEY=empty \
uv run python -m evals.run_eval --category routing
```

不设置 `TS_EVAL_MODEL` 时使用项目默认模型。报告 JSON 顶部会记录所用模型，方便对比。

### 评测模型一键部署（serve_eval_model.sh）

`evals/serve_eval_model.sh` 把"起模型服务 → 等就绪 → 跑评测 → 关服务"串成一条命令：

```bash
# 一键：启动 vLLM → 跑路由类评测 → 自动关闭（默认端口 8100，避开 8000 的 API 服务端）
TS_EVAL_MODEL=Qwen/Qwen3.5-2B evals/serve_eval_model.sh eval --category routing

# macOS 本地没有 NVIDIA GPU 时用 Ollama 后端
EVAL_BACKEND=ollama TS_EVAL_MODEL=qwen3.5:2b evals/serve_eval_model.sh eval --limit 20

# 远端 GPU 服务器已有服务：附加模式，脚本只连接不启停
TS_EVAL_BASE_URL=http://gpu-server:8000/v1 TS_EVAL_MODEL=Qwen3.5-2B \
  evals/serve_eval_model.sh eval --judge

# 也可以分步操作
evals/serve_eval_model.sh up       # 后台启动并等待就绪
evals/serve_eval_model.sh status   # 查看端点、模型列表与托管进程
evals/serve_eval_model.sh down     # 停止本脚本启动的服务
```

行为说明：

- `up`/`eval` 前会先探测端点，**已在运行则进入附加模式**（不启动也不关闭，`down` 会提示）；
- vLLM 启动自动带 `--enable-auto-tool-choice --tool-call-parser`（默认 `hermes`，可用
  `VLLM_TOOL_CALL_PARSER` 覆盖）；首次运行会下载权重，就绪超时默认 1800 秒；
- 非 Linux 环境会警告 vLLM 无法运行并提示改用 Ollama 或远端；
- 服务日志写在 `evals/.serve_eval_model.log`，启动失败时会自动打印末尾 30 行。

## 运行细节

- 每次运行使用**临时目录**存放 SQLite checkpoint/store 和工单 JSONL，不污染 `data/`；
  因此 `db_*` 断言在多次运行间稳定，幂等键不会跨运行干扰。
- 判定用户固定为 `TS_DEMO_USER_ID`（默认 1001），与示例报告数据对齐。
- `--judge` 会对 `behaviors` 逐条让裁判模型输出 yes/no，成本较高，建议在确定性断言
  通过后再开启。
- 报告默认写入 `evals/report_<时间戳>.json`，含每条用例的失败原因、实际工具与路由节点，
  可直接用 `git diff` 之外的方式归档对比；报告文件建议不要提交。

## 与单测的关系

`tests/test_eval_dataset.py` 会随测试套件校验数据集结构（schema、ID 唯一性、分类覆盖、
分类下限、已移除工具不得再出现），保证数据集本身始终可用，但它不会调用任何模型。
