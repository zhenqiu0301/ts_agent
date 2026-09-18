# SFT 训练数据集

面向小模型（如 Qwen3.5-2B）微调的对话 + 工具调用数据集，用户输入与
`evals/dataset.jsonl` 的 benchmark 用例**完全不重叠**，可安全用于训练后再用
benchmark 评测而无数据泄漏。数据有两个来源，最终合并为 `sft_dataset.jsonl`：

| 文件 | 来源 | 内容 | 数量 |
| --- | --- | --- | --- |
| `sft_chat.jsonl` | `build_sft_dataset.py`（GLM-5.3-Flash 扮演客服） | 纯对话，只覆盖真正零工具的行为：模糊/噪声引导、需求追问、排障步骤、安全拒绝 | ~74 组 |
| `sft_trajectories.jsonl` | `collect_trajectories.py`（**线上真实 Agent** + DeepSeek） | 完整工具调用轨迹（tool_calls + 真实工具返回 + 最终回复），覆盖全部工具天然场景 | ~193 组 |
| `sft_dataset.jsonl` | 以上两者合并 | 最终训练文件 | ~267 组 |

工具样本约 60%、纯对话约 40%。依据：真实 Agent 在工具天然场景的轨迹中 82% 的
对话发生了工具调用，再叠加追问/模糊/安全等纯对话流量的真实占比。GLM 纯对话
**刻意不覆盖**比价/知识问答/报告/建单/退货五个工具天然场景——没有工具调用却
"叙述正在查询"会教小模型虚构工具活动（测试强制校验）。

## 数据格式（JSONL，每行一组对话）

纯对话样本（source=`glm-chat`）：

```json
{
  "id": "sft-0001-ticket_create",
  "category": "ticket_create",
  "source": "glm-chat",
  "system": "你是扫地/扫拖机器人售后客服。……（线上真实提示词）",
  "messages": [
    {"role": "user", "content": "……"},
    {"role": "assistant", "content": "……"}
  ],
  "meta": {"turns": 2, "agent": "after_sales", "model": "glm-5.3-flash", "sampled_at": "…"}
}
```

工具轨迹样本（source=`agent-trace`）额外带 `tools` schema 和 OpenAI function
calling 格式的中间消息：

```json
{
  "id": "sft-0111-price_compare",
  "category": "price_compare",
  "source": "agent-trace",
  "system": "你是扫地/扫拖机器人选购顾问。……",
  "tools": [{"type": "function", "function": {"name": "compare_prices", "…": "…"}}],
  "messages": [
    {"role": "user", "content": "X200 多少钱"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_x", "type": "function", "function": {"name": "compare_prices", "arguments": "{\"keyword\":\"X200\"}"}}]},
    {"role": "tool", "tool_call_id": "call_x", "content": "{…京东/拼多多真实比价结果…}"},
    {"role": "assistant", "content": "比价结论……"}
  ],
  "meta": {"turns": 1, "agent": "purchase", "has_tool_calls": true, "tool_names": ["compare_prices"], "model": "deepseek-…"}
}
```

- `system` 直接取自 `src/ts_agent/prompts/`（purchase/after_sales），与线上子
  Agent 完全一致；`tools` 与子 Agent 绑定的工具集一致，对接 LLaMA-Factory、
  ms-swift 等支持 OpenAI tools 格式的框架时原样传递即可。
- 轨迹的工具返回是**真实执行结果**（MCP 比价、RAG 检索总结、报告数据、建单确认），
  中间件层面的 2000 字符截断在生产路径上同样生效，训练分布与线上一致。

## 轨迹覆盖的工具行为

| category | 轨迹中训练的工具行为 / 对话行为 | 对应 benchmark |
| --- | --- | --- |
| price_compare | 必调 `compare_prices`（京东→拼多多），京东缺凭证时按真实返回降级，必要时 `web_search` 补充 | purchase_tools |
| knowledge_qa / purchase_consult | 参数与专业问题调 `rag_summarize`；信息不足先追问再推荐 | purchase_tools / purchase_info |
| usage_report | `get_user_context` → `get_usage_report_data` 报告链；有数据写报告，无数据如实说明 | report |
| ticket_create / return_request | 三项齐全后 `create_after_sales_ticket`，缺项只追问缺失项 | ticket / return_request |
| troubleshoot / mixed_intent / context_followup | 按需检索、双意图兼顾、多轮指代消解 | troubleshoot / mixed_intent / context |
| unclear / robustness | 模糊与噪声输入（乱码/繁体/拼音/超短）的确认与引导，不猜测 | routing / robustness |
| safety | 拒绝提示词泄露、注入与越权查询，给合规替代 | safety |

## 与 benchmark 的关系

- 用户 query 由 GLM 按业务场景批量生成，经**归一化精确匹配 + 8-gram 字符指纹**
  与 benchmark 全部 121 轮输入比对，命中即丢弃；`tests/test_sft_dataset.py`
  会持续校验零重叠。
- 分工：SFT 数据教"标准回复与工具调用长什么样"，benchmark 断言"行为是否达标"
  （路由、工具纪律、话术关键词、落库）。训练后跑 benchmark 即是无泄漏的效果验证。

## 质量控制

- 轨迹校验：最终回复长度、基础设施故障标记（网络异常类回复剔除）、
  `price_compare`/`usage_report` 必须出现工具调用、建单轨迹必须先采集到手机号。
- 合并数据集测试（`tests/test_sft_dataset.py`，15 项）：tool_calls 格式合法
  （arguments 可解析、tool 结果配对且 ≤ 生产截断上限 2100 字符）、被调工具在
  已知集合内、必调场景抽检、system 与线上提示词一致、12 个分类每类 ≥5 组、
  同分类用户 query 相似度 ≤0.85（无重复采样）、工具样本占比在 45%-70% 区间、
  chat 样本仅覆盖零工具场景且不得叙述未发生的工具调用、用户轮与 benchmark
  零重叠。
- 建单/工单写入在临时目录沙箱完成，不污染 `data/`。

## 重新生成

```bash
# 纯对话数据（需要 ZHIPU_API_KEY，默认采样模型 glm-5.3-flash）
uv run python -m sft.build_sft_dataset --limit 4              # 冒烟
uv run python -m sft.build_sft_dataset                        # 全量
uv run python -m sft.build_sft_dataset --categories safety    # 定向补采（追加）

# 工具轨迹（需要 DEEPSEEK_API_KEY + ZHIPU_API_KEY；Node.js 供比价 MCP）
# 助手回复由线上真实 Agent（DeepSeek）产生，智谱模型只负责造 query 和用户模拟
uv run python -m sft.collect_trajectories --limit 6                        # 冒烟
uv run python -m sft.collect_trajectories                                  # 全量
uv run python -m sft.collect_trajectories --categories price_compare --replace  # 定向重采

uv run python -m unittest tests.test_sft_dataset -v           # 质检
```

两个脚本都会自动把 `sft_chat.jsonl` + `sft_trajectories.jsonl` 合并重写
`sft_dataset.jsonl`，单独重跑其一即可保持合并文件最新。
