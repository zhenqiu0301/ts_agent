from __future__ import annotations

from langchain.agents import create_agent

from model.factory import get_chat_model
from tools.mcp_tools import get_price_compare_mcp_tools
from tools.middleware import (
    after_sales_human_review,
    get_context_summarize,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from tools.tools import (
    create_after_sales_ticket,
    create_manual_return_request,
    create_purchase_order,
    get_user_context,
    rag_summarize,
    web_search,
)
from utils.prompt_loader import load_after_sales_prompts, load_system_prompts

purchase_tools = [
    rag_summarize,
    web_search,
    get_user_context,
    create_purchase_order,
]

after_sales_tools = [
    rag_summarize,
    get_user_context,
    # fill_context_for_report,
    # fetch_external_data,
    create_after_sales_ticket,
    create_manual_return_request,
]


async def _merge_tools(internal_tools):
    """内部工具 + MCP外部工具拼接，同名优先保留内部工具。"""
    mcp_tools = await get_price_compare_mcp_tools(refresh=False)
    merged = list(internal_tools)
    internal_names = {str(getattr(t, "name", "")).strip() for t in internal_tools}
    for tool in mcp_tools:
        name = str(getattr(tool, "name", "")).strip()
        if name and name in internal_names:
            continue
        merged.append(tool)
    return merged


class PurchaseAgent:
    def __init__(self, agent):
        self.agent = agent

    @classmethod
    async def create(cls, checkpointer, model=None):
        tools = await _merge_tools(purchase_tools)
        # 保留 report_prompt_switch：当上下文标记 report=True 时仍可自动切换到报告提示词
        agent = create_agent(
            model=model or get_chat_model(),
            system_prompt=load_system_prompts(),
            tools=tools,
            middleware=[
                after_sales_human_review,
                monitor_tool,
                log_before_model,
                report_prompt_switch,
                get_context_summarize(),
            ],
            checkpointer=checkpointer,
        )
        return cls(agent)


class AfterSalesAgent:
    def __init__(self, agent):
        self.agent = agent

    @classmethod
    async def create(cls, checkpointer, model=None):
        tools = list(after_sales_tools)
        agent = create_agent(
            model=model or get_chat_model(),
            system_prompt=load_after_sales_prompts(),
            tools=tools,
            middleware=[
                after_sales_human_review,
                monitor_tool,
                log_before_model,
                report_prompt_switch,
                get_context_summarize(),
            ],
            checkpointer=checkpointer,
        )
        return cls(agent)
