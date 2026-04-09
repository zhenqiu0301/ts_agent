from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from model.factory import chat_model
from tools.middleware import (
    context_summarize,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from tools.tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from utils.prompt_loader import load_after_sales_prompts, load_system_prompts
from .extra_tools import create_after_sales_ticket


PURCHASE_TOOLS = [
    rag_summarize,
    get_user_location,
    get_weather,
    get_current_month,
]

AFTER_SALES_TOOLS = [
    rag_summarize,
    get_user_id,
    get_current_month,
    fill_context_for_report,
    fetch_external_data,
    create_after_sales_ticket,
]


def _load_purchase_prompt() -> str:
    """选购提示词独立文件不存在时，回退到主提示词。"""
    try:
        with open("prompts/purchase_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return load_system_prompts()


class PurchaseAgent:
    def __init__(self):
        # 保留 report_prompt_switch：当上下文标记 report=True 时仍可自动切换到报告提示词
        self.agent = create_agent(
            model=chat_model,
            system_prompt=_load_purchase_prompt(),
            tools=PURCHASE_TOOLS,
            middleware=[
                monitor_tool,
                log_before_model,
                report_prompt_switch,
                context_summarize,
            ],
            checkpointer=InMemorySaver(),
        )


class AfterSalesAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_after_sales_prompts(),
            tools=AFTER_SALES_TOOLS,
            middleware=[
                monitor_tool,
                log_before_model,
                report_prompt_switch,
                context_summarize,
            ],
            checkpointer=InMemorySaver(),
        )
