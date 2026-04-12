import asyncio
import inspect
import threading
from typing import Callable
from utils.prompt_loader import (
    load_after_sales_prompts,
    load_report_prompts,
    load_system_prompts,
)
from tools.tools import reset_tool_runtime_context, set_tool_runtime_context
from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call,
    before_model,
    dynamic_prompt,
    ModelRequest,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.prompt_loader import load_summary_prompts
from utils.logger_handler import logger
from model.factory import chat_model


context_summarize = SummarizationMiddleware(
    model=chat_model,
    trigger=("messages", 20),  # 触发时机：当消息数超过20时进行总结
    keep=("messages", 10),  # 增大保留窗口，减少最近关键信息被折叠的概率
    summary_prompt=load_summary_prompts(),
)


def _run_awaitable_sync(awaitable):
    """在同步上下文安全执行 awaitable，兼容同步 invoke 流程。"""
    try:
        asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        has_running_loop = False

    if not has_running_loop:
        return asyncio.run(awaitable)

    holder = {"value": None, "error": None}

    def _runner():
        try:
            holder["value"] = asyncio.run(awaitable)
        except Exception as e:
            holder["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        raise TimeoutError("工具调用超时")
    if holder["error"] is not None:
        raise holder["error"]
    return holder["value"]


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

after_sales_human_review = HumanInTheLoopMiddleware(
    interrupt_on={
        "create_purchase_order": {
            "allowed_decisions": ["approve", "reject"],
            "description": (
                "订单创建需要人工确认。\n"
                "请核对参数（product_model/quantity/consignee/phone/address）是否准确。"
            ),
        },
        "create_after_sales_ticket": {
            "allowed_decisions": ["approve", "reject"],
            "description": (
                "售后工单创建需要人工确认。\n"
                "请核对工单参数（summary/symptoms/phone）是否准确。"
            ),
        },
        "create_manual_return_request": {
            "allowed_decisions": ["approve", "reject"],
            "description": (
                "人工退货申请创建需要人工确认。\n"
                "请核对参数（reason/product_model/phone/address）是否准确。"
            ),
        },
    },
    description_prefix="售后敏感操作需要人工确认",
)


@wrap_tool_call
def monitor_tool(
    # 请求的数据封装
    request: ToolCallRequest,
    # 执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:  # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

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
        result = handler(request)
        if inspect.isawaitable(result):
            result = _run_awaitable_sync(result)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

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
        preview = _safe_preview_content(getattr(last_message, "content", ""))
        logger.debug(
            f"[log_before_model]{type(last_message).__name__} | {preview}"
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

    return load_system_prompts()
