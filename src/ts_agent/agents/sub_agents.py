from __future__ import annotations

from langchain.agents import create_agent

from ts_agent.model.factory import get_chat_model
from ts_agent.tools.mcp_tools import get_price_comparison_tool
from ts_agent.tools.middleware import (
    after_sales_human_review,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from ts_agent.tools.tools import (
    create_after_sales_ticket,
    create_manual_return_request,
    create_purchase_order,
    get_usage_report_data,
    get_user_context,
    rag_summarize,
    web_search,
)
from ts_agent.utils.prompt_loader import load_after_sales_prompts, load_purchase_prompts

purchase_tools = [
    rag_summarize,
    web_search,
    get_user_context,
    create_purchase_order,
]

after_sales_tools = [
    rag_summarize,
    get_user_context,
    get_usage_report_data,
    create_after_sales_ticket,
    create_manual_return_request,
]


class PurchaseAgent:
    def __init__(self, agent):
        self.agent = agent

    @classmethod
    async def create(cls, checkpointer, model=None):
        tools = [*purchase_tools, get_price_comparison_tool()]
        # 保留 report_prompt_switch：当上下文标记 report=True 时仍可自动切换到报告提示词
        # 会话压缩统一由主图 summarize 节点负责，子 agent 不再内置摘要中间件
        agent = create_agent(
            model=model or get_chat_model(),
            system_prompt=load_purchase_prompts(),
            tools=tools,
            middleware=[
                after_sales_human_review,
                monitor_tool,
                log_before_model,
                report_prompt_switch,
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
            ],
            checkpointer=checkpointer,
        )
        return cls(agent)
