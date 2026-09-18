import os
from collections.abc import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from ts_agent.tools.tools import reset_tool_runtime_context, set_tool_runtime_context
from ts_agent.utils.logger_handler import logger
from ts_agent.utils.prompt_loader import (
    load_after_sales_prompts,
    load_purchase_prompts,
    load_report_prompts,
    load_system_prompts,
)


def _safe_preview_content(content) -> str:
    """将不同类型的 message.content 安全转成短日志文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content[:3]:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip() or "item"
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(f"{item_type}:{text.strip()[:80]}")
                else:
                    parts.append(item_type)
            else:
                parts.append(str(item)[:80])
        return " | ".join(parts)
    if content is None:
        return ""
    return str(content).strip()

TOOL_RESULT_MAX_CHARS = int(os.getenv("TS_TOOL_RESULT_MAX_CHARS", "2000") or "2000")
MAX_TOOL_ROUNDS = int(os.getenv("TS_MAX_TOOL_ROUNDS", "6") or "6")
TRUNCATED_SUFFIX = "\n…[工具结果过长，已截断；请基于以上信息作答，不要重复调用]"


def truncate_tool_result(content, limit: int = TOOL_RESULT_MAX_CHARS):
    """超长工具结果截断，防止撑爆模型上下文（小模型尤其敏感）。"""
    if not isinstance(content, str) or len(content) <= limit:
        return content
    return content[:limit] + TRUNCATED_SUFFIX


def count_tool_rounds(messages) -> int:
    """统计已发生的工具调用轮数（一条带 tool_calls 的 AI 消息为一轮）。"""
    return sum(
        1
        for msg in messages
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
    )


async def limit_tool_rounds_core(request: ModelRequest, handler):
    """单轮内工具调用轮数超限后摘除全部工具定义，强制模型基于已有信息作答。"""
    if request.tools and count_tool_rounds(request.state["messages"]) >= MAX_TOOL_ROUNDS:
        logger.warning(
            "[tool loop guard]工具调用已达 %d 轮上限，本轮强制模型直接作答",
            MAX_TOOL_ROUNDS,
        )
        request = request.override(tools=[])
    return await handler(request)


limit_tool_rounds = wrap_model_call(limit_tool_rounds_core)


@wrap_tool_call
async def monitor_tool(
    # 请求的数据封装
    request: ToolCallRequest,
    # 执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:  # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    arguments = request.tool_call.get("args", {})
    argument_keys = sorted(arguments) if isinstance(arguments, dict) else []
    logger.info("[tool monitor]参数字段：%s", argument_keys)

    runtime_context = {}
    try:
        runtime_obj = getattr(request, "runtime", None)
        if runtime_obj is not None:
            runtime_context = dict(getattr(runtime_obj, "context", {}) or {})
            runtime_config = getattr(runtime_obj, "config", {}) or {}
            configurable = runtime_config.get("configurable", {})
            if isinstance(configurable, dict):
                for key in ("user_id", "thread_id"):
                    value = str(configurable.get(key, "")).strip()
                    if value:
                        runtime_context[key] = value
    except Exception as e:
        logger.debug(f"[tool monitor]提取runtime上下文失败: {e}")

    token = set_tool_runtime_context(runtime_context)
    try:
        result = await handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] in {
            "fill_context_for_report",
            "get_usage_report_data",
        }:
            request.runtime.context["report"] = True

        if isinstance(result, ToolMessage):
            result.content = truncate_tool_result(result.content)

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e
    finally:
        reset_tool_runtime_context(token)


@before_model
def log_before_model(
    state: AgentState,  # 整个Agent智能体中的状态记录
    runtime: Runtime,  # 记录了整个执行过程中的上下文信息
):  # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    try:
        last_message = state["messages"][-1]
        if os.getenv("TS_LOG_MESSAGE_CONTENT", "").lower() in {"1", "true", "yes"}:
            preview = _safe_preview_content(getattr(last_message, "content", ""))
            logger.debug(
                "[log_before_model]%s | %s", type(last_message).__name__, preview
            )
    except Exception as e:
        logger.debug(f"[log_before_model]最后一条消息预览失败: {e}")

    return None


@dynamic_prompt  # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):  # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    route = request.runtime.context.get("route", "")
    if is_report:  # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    if route == "after_sales":
        return load_after_sales_prompts()

    if route == "purchase":
        return load_purchase_prompts()

    return load_system_prompts()
