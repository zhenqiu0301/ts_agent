from __future__ import annotations

from langchain.agents import create_agent

from ts_agent.model.factory import get_chat_model
from ts_agent.tools.mcp_tools import get_price_comparison_tool
from ts_agent.tools.middleware import (
    limit_tool_rounds,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from ts_agent.tools.tools import (
    create_after_sales_ticket,
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
]

after_sales_tools = [
    rag_summarize,
    get_user_context,
    get_usage_report_data,
    create_after_sales_ticket,
]


class PurchaseAgent:
    def __init__(self, agent):
        self.agent = agent

    @classmethod
    async def create(cls, checkpointer, model=None):
        tools = [*purchase_tools, get_price_comparison_tool()]
        # 选购侧已无敏感写操作，不挂人工审批中间件
        # 会话压缩统一由主图 summarize 节点负责，子 agent 不再内置摘要中间件
        agent = create_agent(
            model=model or get_chat_model(),
            system_prompt=load_purchase_prompts(),
            tools=tools,
            middleware=[
                monitor_tool,
                limit_tool_rounds,
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
        # 会话压缩统一由主图 summarize 节点负责，子 agent 不再内置摘要中间件
        agent = create_agent(
            model=model or get_chat_model(),
            system_prompt=load_after_sales_prompts(),
            tools=tools,
            middleware=[
                monitor_tool,
                limit_tool_rounds,
                log_before_model,
                report_prompt_switch,
            ],
            checkpointer=checkpointer,
        )
        return cls(agent)
