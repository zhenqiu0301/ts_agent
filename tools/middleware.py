from typing import Callable
from utils.prompt_loader import (
    load_after_sales_prompts,
    load_report_prompts,
    load_system_prompts,
)
from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call,
    before_model,
    dynamic_prompt,
    ModelRequest,
    SummarizationMiddleware,
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
    keep=("messages", 8),  # 增大保留窗口，减少最近关键信息被折叠的概率
    summary_prompt=load_summary_prompts(),
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

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
    state: AgentState,  # 整个Agent智能体中的状态记录
    runtime: Runtime,  # 记录了整个执行过程中的上下文信息
):  # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(
        f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}"
    )

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
