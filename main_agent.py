import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from tools.tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
)
from tools.middleware import (
    monitor_tool,
    log_before_model,
    report_prompt_switch,
    context_summarize,
)

tools = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
]
middleware = [
    monitor_tool,
    log_before_model,
    report_prompt_switch,
    context_summarize,
]


class MainAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=tools,
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )

    def execute_stream(self, query: str, thread_id: str):
        input_dict = {"messages": [HumanMessage(content=query)]}

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        last_emitted = ""
        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            context={"report": False},
            config={"configurable": {"thread_id": thread_id}},
        ):
            latest_message = chunk["messages"][-1]
            role = getattr(latest_message, "type", None) or getattr(
                latest_message, "role", None
            )
            if role not in ("ai", "assistant"):
                continue

            content = getattr(latest_message, "content", "")
            if not isinstance(content, str):
                continue

            text = content.strip()
            if not text:
                continue

            # stream_mode="values" 常返回“完整累计文本”，这里只输出新增增量。
            if text.startswith(last_emitted):
                delta = text[len(last_emitted) :]
            else:
                delta = text

            if delta:
                yield delta
                last_emitted = text


if __name__ == "__main__":
    agent = MainAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
