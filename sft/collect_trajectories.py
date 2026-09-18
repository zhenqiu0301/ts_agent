"""用项目真实子 Agent（DeepSeek）采集含工具调用的 SFT 轨迹。

用法（在项目根目录）：
    uv run python -m sft.collect_trajectories --limit 6     # 冒烟 6 组
    uv run python -m sft.collect_trajectories               # 全量（约 180 组）

与 build_sft_dataset.py 的区别：这里不靠 GLM 扮演客服，而是把采到的用户 query
喂给线上真实的 PurchaseAgent / AfterSalesAgent，导出完整消息轨迹——包括
assistant 的 tool_calls、真实工具返回（rag_summarize / compare_prices MCP /
报告链 / 建单）和最终回复。工具调用格式即线上 LangChain 的 OpenAI function
calling 格式，供小模型 SFT 直接学习。

沙箱隔离：工单写入临时目录（TS_AFTER_SALES_TICKET_PATH），checkpoint/store
放在临时目录，不污染 data/。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from sft.build_sft_dataset import (
    BENCHMARK_PATH,
    REPO_ROOT,
    SCENARIOS,
    ContaminationGuard,
    Scenario,  # noqa: F401  (re-export 便于单测)
    ZhipuClient,
    generate_queries,
    load_env,
    pick_depth,
    simulate_next_user_turn,
)

sys.path.insert(0, str(REPO_ROOT / "src"))

TRAJECTORY_OUT = REPO_ROOT / "sft" / "sft_trajectories.jsonl"
CHAT_DATASET_PATH = REPO_ROOT / "sft" / "sft_chat.jsonl"
MERGED_OUT = REPO_ROOT / "sft" / "sft_dataset.jsonl"
EVAL_USER_ID = "1001"
TICKET_ENV = "TS_AFTER_SALES_TICKET_PATH"

# 轨迹采集的场景配额（unclear 无工具，由 GLM 对话数据覆盖，不采集）
TRAJECTORY_COUNTS = {
    "purchase_consult": 30,
    "price_compare": 28,
    "knowledge_qa": 24,
    "troubleshoot": 24,
    "ticket_create": 24,
    "context_followup": 14,
    "usage_report": 16,
    "return_request": 12,
    "mixed_intent": 8,
}

# 这些场景的 prompt 强制要求工具调用，轨迹里没有工具调用视为不合格
TOOL_REQUIRED_CATEGORIES = {"price_compare", "usage_report"}

# 轨迹多轮配比：工具轨迹 2 轮足够，不放 3 轮
DEPTH_WEIGHTS = (0.55, 0.45, 0.0)

# 基础设施故障标记：出现说明这次运行环境有问题，不能当作训练数据
INFRA_FAILURE_MARKERS = (
    "当前无法连接模型服务",
    "检索总结暂时不可用",
    "工单创建失败：写入工单存储时发生异常",
)


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def lc_message_to_openai(message: Any) -> dict[str, Any] | None:
    """LangChain 消息 → OpenAI 对话格式（保留 tool_calls / tool 角色）。"""
    if isinstance(message, SystemMessage):
        return None  # system 单独存字段
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": _content_to_str(message.content)}
    if isinstance(message, AIMessage):
        if getattr(message, "tool_calls", None):
            return {
                "role": "assistant",
                "content": _content_to_str(message.content) or "",
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(
                                tool_call["args"], ensure_ascii=False
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        return {"role": "assistant", "content": _content_to_str(message.content)}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _content_to_str(message.content),
        }
    return None


def tools_to_openai_schema(tools: list[Any]) -> list[dict[str, Any]]:
    schemas = []
    for tool in tools:
        parameters: dict[str, Any] = {}
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            parameters = args_schema.model_json_schema()
            parameters.pop("title", None)
            for prop in parameters.get("properties", {}).values():
                prop.pop("title", None)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.name),
                    "description": str(getattr(tool, "description", "") or "").strip(),
                    "parameters": parameters,
                },
            }
        )
    return schemas


def trajectory_is_valid(category: str, messages: list[dict[str, Any]]) -> str | None:
    """返回不合格原因，None 表示合格。"""
    assistant_turns = [
        m for m in messages if m["role"] == "assistant" and not m.get("tool_calls")
    ]
    if not assistant_turns or len(assistant_turns[-1]["content"].strip()) < 30:
        return "最终回复过短或缺失"
    combined = "\n".join(m["content"] for m in messages if m["role"] in {"assistant", "tool"})
    for marker in INFRA_FAILURE_MARKERS:
        if marker in combined:
            return f"基础设施故障：{marker}"
    tool_names_called = {
        tool_call["function"]["name"]
        for m in messages
        for tool_call in m.get("tool_calls", [])
    }
    if category in TOOL_REQUIRED_CATEGORIES and not tool_names_called:
        return "必调工具场景未出现工具调用"
    if "create_after_sales_ticket" in tool_names_called:
        # 建单必须发生在三项信息齐全之后：检查轨迹里用户是否给过手机号
        #（完整 11 位或打码格式均可，工具侧 _validate_phone 会做最终校验）
        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        if not re.search(r"1[3-9][\dxX＊*]{9}|1[3-9]\d[-\s]*\d{4}[-\s]*\d{4}", user_text):
            return "未采集到手机号就建单，属违规轨迹"
    return None


async def collect_one_conversation(
    agent_obj: Any,
    route: str,
    scenario: Scenario,
    first_turn: str,
    depth: int,
    client: ZhipuClient,
    rng,
    guard: ContaminationGuard,
) -> dict[str, Any] | None:
    thread_id = f"sft-trace-{rng.randrange(10**10):010d}"
    config = {
        "configurable": {"thread_id": thread_id, "user_id": EVAL_USER_ID},
        "tags": ["sft-collect"],
    }
    raw_messages: list[Any] = []
    stored_turns: list[dict[str, str]] = []
    for turn_index in range(depth):
        if turn_index == 0:
            user_turn = first_turn
        else:
            plain_dialogue = [
                m
                for m in (
                    lc_message_to_openai(message) for message in raw_messages
                )
                if m and m["role"] in {"user", "assistant"}
            ]
            user_turn = await simulate_next_user_turn(
                client, scenario, plain_dialogue, rng, guard
            )
        result = await agent_obj.ainvoke(
            {"messages": [HumanMessage(content=user_turn)]},
            context={"route": route, "report": False},
            config=config,
        )
        new_messages = list(result["messages"])[len(raw_messages):]
        if not new_messages:
            return None
        raw_messages.extend(new_messages)
        stored_turns.append({"role": "user", "content": user_turn})

    converted = [
        converted
        for message in raw_messages
        if (converted := lc_message_to_openai(message)) is not None
    ]
    if not converted or converted[0]["role"] != "user":
        return None

    tool_names = sorted(
        {
            tool_call["function"]["name"]
            for m in converted
            for tool_call in m.get("tool_calls", [])
        }
    )
    reason = trajectory_is_valid(scenario.name, converted)
    if reason:
        print(f"  [reject] {scenario.name}: {reason}")
        return None

    return {
        "category": scenario.name,
        "source": "agent-trace",
        "messages": converted,
        "meta": {
            "turns": depth,
            "agent": scenario.agent,
            "has_tool_calls": bool(tool_names),
            "tool_names": tool_names,
            "sampled_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


async def run(args: argparse.Namespace) -> int:
    load_env()
    import os
    import random
    import tempfile

    zhipu_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not zhipu_key or "your_" in zhipu_key:
        print("缺少 ZHIPU_API_KEY（用户模拟器需要），请在 .env 中配置。")
        return 1

    rng = random.Random(args.seed)
    benchmark_turns = [
        turn["content"]
        for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for turn in json.loads(line)["turns"]
    ]
    guard = ContaminationGuard(benchmark_turns)
    client = ZhipuClient(zhipu_key, args.sim_model, concurrency=4)

    scenarios = [
        replace(
            scenario, count=TRAJECTORY_COUNTS[scenario.name], depth_weights=DEPTH_WEIGHTS
        )
        for scenario in SCENARIOS
        if scenario.name in TRAJECTORY_COUNTS
    ]
    if args.categories:
        wanted = {name.strip() for name in args.categories.split(",") if name.strip()}
        unknown = wanted - {s.name for s in scenarios}
        if unknown:
            print(f"未知分类：{sorted(unknown)}，可选：{sorted(TRAJECTORY_COUNTS)}")
            return 1
        scenarios = [s for s in scenarios if s.name in wanted]

    # 定向补采时保留已有轨迹（--replace 时先剔除目标分类），避免整库重写
    dataset: list[dict[str, Any]] = []
    if args.categories and TRAJECTORY_OUT.is_file():
        dataset = [
            json.loads(line)
            for line in TRAJECTORY_OUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.replace:
            wanted = {name.strip() for name in args.categories.split(",") if name.strip()}
            before = len(dataset)
            dataset = [c for c in dataset if c["category"] not in wanted]
            print(
                f"载入已有轨迹 {len(dataset)} 组（--replace 剔除目标分类 "
                f"{before - len(dataset)} 组），将追加新采集结果"
            )
        else:
            print(f"载入已有轨迹 {len(dataset)} 组，将追加新采集结果")

    try:
        print("[阶段 1] 生成用户 query（与 benchmark 去重）...")
        plan: list[tuple[Scenario, str, int]] = []
        for scenario in scenarios:
            queries = await generate_queries(client, scenario, rng, guard)
            rng.shuffle(queries)
            for query in queries:
                plan.append((scenario, query, pick_depth(scenario, rng)))
        rng.shuffle(plan)
        if args.limit:
            plan = plan[: args.limit]
        print(f"[阶段 2] 采集 {len(plan)} 组真实 Agent 轨迹（并发 {args.concurrency}）...")

        from ts_agent.agents.persistence import build_persistent_backends
        from ts_agent.agents.sub_agents import (
            AfterSalesAgent,
            PurchaseAgent,
            after_sales_tools,
            purchase_tools,
        )
        from ts_agent.model.factory import get_chat_model
        from ts_agent.tools.mcp_tools import get_price_comparison_tool
        from ts_agent.utils.prompt_loader import (
            load_after_sales_prompts,
            load_purchase_prompts,
        )

        purchase_system = load_purchase_prompts().strip()
        after_sales_system = load_after_sales_prompts().strip()
        tool_schemas = {
            "purchase": tools_to_openai_schema(
                [*purchase_tools, get_price_comparison_tool()]
            ),
            "after_sales": tools_to_openai_schema(list(after_sales_tools)),
        }
        model_desc = "default"

        with tempfile.TemporaryDirectory(prefix="sft_trace_") as tmp:
            os.environ[TICKET_ENV] = str(Path(tmp) / "tickets.jsonl")
            try:
                backends = await build_persistent_backends(Path(tmp) / "state")
                model = get_chat_model()
                model_desc = getattr(model, "model_name", None) or "default"
                purchase_agent = await PurchaseAgent.create(backends.checkpointer, model)
                after_sales_agent = await AfterSalesAgent.create(
                    backends.checkpointer, model
                )
                agent_map = {
                    "purchase": (purchase_agent.agent, "purchase", purchase_system),
                    "after_sales": (after_sales_agent.agent, "after_sales", after_sales_system),
                }

                dataset_extend: list[dict[str, Any]] = []
                progress = 0
                sem = asyncio.Semaphore(args.concurrency)

                async def one(scenario: Scenario, query: str, depth: int) -> None:
                    nonlocal progress
                    agent_obj, route, _ = agent_map[scenario.agent]
                    async with sem:
                        try:
                            conv = await collect_one_conversation(
                                agent_obj, route, scenario, query, depth, client, rng, guard
                            )
                        except Exception as exc:  # noqa: BLE001
                            conv = None
                            print(f"  [error] {scenario.name}: {type(exc).__name__}: {exc}")
                    async with asyncio.Lock():
                        progress += 1
                        if conv:
                            conv["meta"]["model"] = model_desc
                            dataset_extend.append(conv)
                        print(
                            f"  [{progress}/{len(plan)}] "
                            f"{'ok' if conv else 'rejected'} {scenario.name}"
                        )

                await asyncio.gather(*(one(s, q, d) for s, q, d in plan))
                dataset.extend(dataset_extend)
            finally:
                os.environ.pop(TICKET_ENV, None)
    finally:
        await client.close()

    dataset.sort(key=lambda c: (c["category"],))
    for index, conv in enumerate(dataset, 1):
        conv["id"] = f"traj-{index:04d}-{conv['category']}"
        conv["system"] = (
            purchase_system if conv["meta"]["agent"] == "purchase" else after_sales_system
        )
        conv["tools"] = tool_schemas[conv["meta"]["agent"]]
    TRAJECTORY_OUT.write_text(
        "".join(json.dumps(conv, ensure_ascii=False) + "\n" for conv in dataset),
        encoding="utf-8",
    )

    print(
        f"\n轨迹采集完成：本次新增 {len(dataset_extend)} 组"
        f"（含工具调用 {sum(1 for c in dataset_extend if c['meta']['has_tool_calls'])} 组，"
        f"拒绝 {len(plan) - len(dataset_extend)} 组），"
        f"轨迹库共 {len(dataset)} 组 → {TRAJECTORY_OUT.relative_to(REPO_ROOT)}"
    )

    # ---- 合并 chat + trajectories → sft_dataset.jsonl
    merged: list[dict[str, Any]] = []
    if CHAT_DATASET_PATH.is_file():
        for line in CHAT_DATASET_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                conv = json.loads(line)
                conv.setdefault("source", "glm-chat")
                merged.append(conv)
        print(f"合并 GLM 对话数据 {len(merged)} 组")
    merged.extend(dataset)
    for index, conv in enumerate(merged, 1):
        conv["id"] = f"sft-{index:04d}-{conv['category']}"
    MERGED_OUT.write_text(
        "".join(json.dumps(conv, ensure_ascii=False) + "\n" for conv in merged),
        encoding="utf-8",
    )
    print(f"最终数据集：{len(merged)} 组 → {MERGED_OUT.relative_to(REPO_ROOT)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="采集含工具调用的真实 Agent SFT 轨迹")
    parser.add_argument("--limit", type=int, help="最多采集 N 组（冒烟用）")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--categories",
        help="只补采指定分类（逗号分隔，如 ticket_create），已有轨迹保留并追加",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="与 --categories 连用：先删除这些分类的已有轨迹再重新采集",
    )
    parser.add_argument(
        "--sim-model",
        default="glm-5.3-flash",
        help="用户模拟器/造 query 用的智谱模型（默认 glm-5.3-flash）",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
