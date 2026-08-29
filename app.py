from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ts_agent.agents.main_graph_agent import MainGraphAgent
from ts_agent.utils.logger_handler import enable_file_logging, logger

AGENT_RUNTIME_VERSION = "native-async-v1"
enable_file_logging()

# 标题
st.title("ts智能客服")
st.divider()

if (
    "agent" not in st.session_state
    or st.session_state.get("agent_runtime_version") != AGENT_RUNTIME_VERSION
):
    st.session_state["agent"] = None
    st.session_state["agent_runtime_version"] = AGENT_RUNTIME_VERSION

if "message" not in st.session_state:
    st.session_state["message"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"ts-{datetime.now().strftime('%Y%m%d%H%M%S')}"

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "wts"

if "pending_bootstrap_summary" not in st.session_state:
    st.session_state["pending_bootstrap_summary"] = None
if "bootstrap_summary_loaded" not in st.session_state:
    st.session_state["bootstrap_summary_loaded"] = False

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

    with st.expander("记忆管理"):
        if st.button("查看我的记忆", use_container_width=True):
            async def load_memories():
                agent = await MainGraphAgent.create()
                try:
                    st.session_state["memory_snapshot"] = await agent.list_user_memories(
                        st.session_state["user_id"]
                    )
                    yield "记忆已加载。"
                except Exception as e:
                    # 捕获异常让生成器正常收尾，确保 finally 在存活的事件循环上关闭 agent
                    logger.error(f"[memory]加载记忆失败: {e}", exc_info=True)
                    yield "记忆加载失败，请稍后重试。"
                finally:
                    await agent.close()

            st.write_stream(load_memories())

        snapshot = st.session_state.get("memory_snapshot")
        if snapshot:
            st.caption("结构化用户画像")
            st.json(snapshot.get("profile", []), expanded=False)
            st.caption("历史业务事件")
            st.json(snapshot.get("episodes", []), expanded=False)

        if st.button("清除我的长期记忆", use_container_width=True):
            async def clear_memories():
                agent = await MainGraphAgent.create()
                try:
                    await agent.clear_user_memories(st.session_state["user_id"])
                    st.session_state["memory_snapshot"] = None
                    st.session_state["bootstrap_summary_loaded"] = False
                    yield "长期记忆已清除。"
                except Exception as e:
                    logger.error(f"[memory]清除记忆失败: {e}", exc_info=True)
                    yield "记忆清除失败，请稍后重试。"
                finally:
                    await agent.close()

            st.write_stream(clear_memories())

    if st.button("结束会话并整理记忆", use_container_width=True):
        async def finalize_session():
            if not st.session_state.get("message"):
                yield "当前会话还没有可整理的内容。"
                return
            agent = await MainGraphAgent.create()
            try:
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
                changed = await agent.finalize_thread(
                    st.session_state["thread_id"],
                    st.session_state["user_id"],
                    recent_messages=recent_messages,
                )
                st.session_state["finalize_notice"] = {
                    "type": "success" if changed else "info",
                    "text": (
                        "已完成长期记忆整理。"
                        if changed
                        else "没有可整理的长期记忆增量。"
                    ),
                }
            except Exception as e:
                logger.error(f"[memory]整理长期记忆失败: {e}", exc_info=True)
                yield "长期记忆整理失败，请稍后重试。"
                return
            finally:
                await agent.close()
            st.session_state["thread_id"] = (
                f"ts-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            st.session_state["message"] = []
            st.session_state["pending_bootstrap_summary"] = None
            st.session_state["bootstrap_summary_loaded"] = False
            yield "会话已结束。"

        with st.spinner("正在整理长期记忆..."):
            st.write_stream(finalize_session())
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
        progress = st.status("正在识别用户意图...", expanded=False)

        def show_event(event):
            if event.get("type") == "tools":
                progress.update(label=f"正在调用工具：{', '.join(event['names'])}")
                return
            node_labels = {
                "pending_gate": "正在检查待处理业务动作",
                "analyze": "已完成意图识别",
                "purchase": "正在处理选购需求",
                "after_sales": "正在处理售后需求",
                "summarize": "正在更新会话记忆",
            }
            names = event.get("names", [])
            if names:
                progress.update(label=node_labels.get(names[-1], "正在生成回答"))

        async def capture(cache_list):
            agent = await MainGraphAgent.create()
            try:
                if not st.session_state["bootstrap_summary_loaded"]:
                    st.session_state["pending_bootstrap_summary"] = (
                        await agent.load_user_memory_summary(
                            st.session_state["user_id"], prompt
                        )
                    )
                    st.session_state["bootstrap_summary_loaded"] = True
                async for chunk in agent.execute_stream(
                    prompt,
                    st.session_state["thread_id"],
                    st.session_state["user_id"],
                    st.session_state.get("pending_bootstrap_summary"),
                    event_callback=show_event,
                ):
                    cache_list.append(chunk)
                    yield chunk
            except Exception as e:
                # 捕获后生成器正常收尾，确保 finally 在存活的事件循环上关闭 agent
                logger.error(f"[chat]处理消息失败: {e}", exc_info=True)
                message = "抱歉，刚才处理时出现异常，请稍后重试。"
                cache_list.append(message)
                yield message
            finally:
                await agent.close()

        st.chat_message("assistant").write_stream(capture(response_messages))
        progress.update(label="回答已完成", state="complete")
        full_response = "".join(response_messages).strip()
        st.session_state["pending_bootstrap_summary"] = None
        st.session_state["message"].append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()

if __name__ == "__main__":
    pass
