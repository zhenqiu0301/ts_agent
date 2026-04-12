import time
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from agents.main_graph_agent import MainGraphAgent


AGENT_RUNTIME_VERSION = "sync-wrap-tool-v1"

# 标题
st.title("ts智能客服")
st.divider()

if (
    "agent" not in st.session_state
    or st.session_state.get("agent_runtime_version") != AGENT_RUNTIME_VERSION
):
    st.session_state["agent"] = MainGraphAgent()
    st.session_state["agent_runtime_version"] = AGENT_RUNTIME_VERSION

if "message" not in st.session_state:
    st.session_state["message"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"ts-{datetime.now().strftime('%Y%m%d%H%M%S')}"

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "wts"

if "pending_bootstrap_summary" not in st.session_state:
    st.session_state["pending_bootstrap_summary"] = st.session_state[
        "agent"
    ].load_user_memory_summary(st.session_state["user_id"])

if "finalize_notice" in st.session_state:
    notice = st.session_state.pop("finalize_notice")
    if notice["type"] == "success":
        st.success(notice["text"])
    elif notice["type"] == "info":
        st.info(notice["text"])
    else:
        st.warning(notice["text"])

with st.sidebar:
    st.subheader("会话控制")
    st.caption(f"当前 thread_id: {st.session_state['thread_id']}")

    if st.button("结束会话并整理记忆", use_container_width=True):
        agent = st.session_state["agent"]
        if hasattr(agent, "finalize_thread"):
            recent_window = st.session_state.get("message", [])[
                -agent.MAX_RECENT_MESSAGES :
            ]
            recent_messages: list[BaseMessage] = []
            for item in recent_window:
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                if role == "user":
                    recent_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    recent_messages.append(AIMessage(content=content))
            with st.spinner("正在整理长期记忆..."):
                changed = agent.finalize_thread(
                    st.session_state["thread_id"],
                    st.session_state["user_id"],
                    recent_messages=recent_messages,
                )
            if changed:
                st.session_state["finalize_notice"] = {
                    "type": "success",
                    "text": "已完成长期记忆整理。",
                }
            else:
                st.session_state["finalize_notice"] = {
                    "type": "info",
                    "text": "没有可整理的长期记忆增量。",
                }
        else:
            st.session_state["finalize_notice"] = {
                "type": "warning",
                "text": "当前 Agent 不支持 finalize_thread。",
            }

        # finalize 结束后重建 agent，确保加载新会话摘要前状态被重置。
        st.session_state["agent"] = MainGraphAgent()
        agent = st.session_state["agent"]

        st.session_state["thread_id"] = f"ts-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state["message"] = []
        st.session_state["pending_bootstrap_summary"] = agent.load_user_memory_summary(
            st.session_state["user_id"]
        )
        st.rerun()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        agent = st.session_state["agent"]
        res_stream = agent.execute_stream(
            prompt,
            st.session_state["thread_id"],
            st.session_state["user_id"],
            st.session_state.get("pending_bootstrap_summary"),
        )

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(
            capture(res_stream, response_messages)
        )
        full_response = "".join(response_messages).strip()
        st.session_state["pending_bootstrap_summary"] = None
        st.session_state["message"].append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()

if __name__ == "__main__":
    pass
