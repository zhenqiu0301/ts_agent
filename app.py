"""Streamlit 客户端：通过 ts_agent 服务端 API 完成对话与记忆管理。"""

import json
import os
import urllib.parse
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st

API_BASE_URL = os.getenv("TS_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _new_thread_id() -> str:
    # 时间戳 + 随机后缀，避免同一秒内多个客户端会话撞 id
    return f"ts-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"

NODE_LABELS = {
    "pending_gate": "正在检查待处理业务动作",
    "analyze": "已完成意图识别",
    "purchase": "正在处理选购需求",
    "after_sales": "正在处理售后需求",
    "summarize": "正在更新会话记忆",
}


def _parse_event_data(raw_lines: list[str]) -> dict[str, Any]:
    payload = "\n".join(raw_lines)
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else {"raw": payload}
    except json.JSONDecodeError:
        return {"raw": payload}


def iter_sse_events(response: httpx.Response):
    """把 SSE 响应流解析为 (event_type, data) 序列。"""
    event_type = ""
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line.startswith(":"):
            continue
        if not line:
            if event_type:
                yield event_type, _parse_event_data(data_lines)
            event_type, data_lines = "", []
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if event_type:
        yield event_type, _parse_event_data(data_lines)


def stream_chat(prompt, thread_id, user_id, on_event, cache_list):
    """调用 /api/chat 并把 SSE 事件转成可渲染的文本流。"""
    payload = {"thread_id": thread_id, "user_id": user_id, "message": prompt}
    accumulated: list[str] = []
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE_URL}/api/chat",
            json=payload,
            timeout=httpx.Timeout(10.0, read=300.0),
        ) as response:
            if response.status_code != 200:
                message = f"服务请求失败（HTTP {response.status_code}），请稍后重试。"
                cache_list.append(message)
                yield message
                return
            for kind, data in iter_sse_events(response):
                if kind == "chunk":
                    delta = str(data.get("delta", ""))
                    if delta:
                        accumulated.append(delta)
                        cache_list.append(delta)
                        yield delta
                elif kind == "tool":
                    on_event("tool", data.get("names", []))
                elif kind == "node":
                    on_event("node", data.get("names", []))
                elif kind == "done":
                    full = str(data.get("response", ""))
                    # 兜底：服务端未产出任何 chunk 时直接展示完整回答
                    if not accumulated and full:
                        cache_list.append(full)
                        yield full
                elif kind == "error":
                    message = str(
                        data.get("message", "服务端处理失败，请稍后重试。")
                    )
                    cache_list.append(message)
                    yield message
    except httpx.HTTPError:
        message = "与服务端的连接中断，请确认服务端已启动后重试。"
        cache_list.append(message)
        yield message


# 标题
st.title("ts智能客服")
st.divider()

try:
    health = httpx.get(f"{API_BASE_URL}/api/health", timeout=3)
    server_ready = health.status_code == 200
except httpx.HTTPError:
    server_ready = False
if not server_ready:
    st.warning(
        f"无法连接服务端（{API_BASE_URL}）。请先启动服务端："
        "`uv run uvicorn ts_agent.server.app:app --port 8000` 或直接运行 `./start.sh`。"
    )
    st.stop()

if "message" not in st.session_state:
    st.session_state["message"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = _new_thread_id()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "wts"

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
    input_user_id = st.text_input(
        "用户 ID",
        value=st.session_state["user_id"],
        help="长期记忆按用户 ID 隔离；修改后会自动开启新会话。",
    )
    st.caption(
        f"当前用户: {st.session_state['user_id']} | thread_id: {st.session_state['thread_id']}"
    )

    normalized_user = input_user_id.strip()
    if not normalized_user:
        st.caption("用户 ID 不能为空，将继续使用当前用户。")
    elif normalized_user != st.session_state["user_id"]:
        st.session_state["user_id"] = normalized_user
        st.session_state["thread_id"] = _new_thread_id()
        st.session_state["message"] = []
        st.session_state["memory_snapshot"] = None
        st.rerun()

    with st.expander("记忆管理"):
        if st.button("查看我的记忆", use_container_width=True):
            quoted_user = urllib.parse.quote(st.session_state["user_id"], safe="")
            try:
                response = httpx.get(
                    f"{API_BASE_URL}/api/memory/{quoted_user}", timeout=30
                )
                response.raise_for_status()
                st.session_state["memory_snapshot"] = response.json()
                st.toast("记忆已加载。")
            except (httpx.HTTPError, ValueError):
                st.toast("记忆加载失败，请稍后重试。", icon="⚠️")

        snapshot = st.session_state.get("memory_snapshot")
        if snapshot:
            st.caption("结构化用户画像")
            st.json(snapshot.get("profile", []), expanded=False)
            st.caption("历史业务事件")
            st.json(snapshot.get("episodes", []), expanded=False)

        if st.button("清除我的长期记忆", use_container_width=True):
            quoted_user = urllib.parse.quote(st.session_state["user_id"], safe="")
            try:
                response = httpx.delete(
                    f"{API_BASE_URL}/api/memory/{quoted_user}", timeout=30
                )
                response.raise_for_status()
                st.session_state["memory_snapshot"] = None
                st.toast("长期记忆已清除。")
            except httpx.HTTPError:
                st.toast("记忆清除失败，请稍后重试。", icon="⚠️")

    if st.button("结束会话并整理记忆", use_container_width=True):
        if not st.session_state.get("message"):
            st.info("当前会话还没有可整理的内容。")
        else:
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/api/session/finalize",
                    json={
                        "thread_id": st.session_state["thread_id"],
                        "user_id": st.session_state["user_id"],
                        "messages": st.session_state["message"],
                    },
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state["finalize_notice"] = {
                    "type": "success" if response.json().get("changed") else "info",
                    "text": (
                        "已完成长期记忆整理。"
                        if response.json().get("changed")
                        else "没有可整理的长期记忆增量。"
                    ),
                }
                st.session_state["thread_id"] = _new_thread_id()
                st.session_state["message"] = []
            except (httpx.HTTPError, ValueError):
                st.session_state["finalize_notice"] = {
                    "type": "warning",
                    "text": "长期记忆整理失败，请稍后重试。",
                }
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

        def show_event(kind, names):
            if kind == "tool":
                progress.update(label=f"正在调用工具：{', '.join(names)}")
                return
            if names:
                progress.update(label=NODE_LABELS.get(names[-1], "正在生成回答"))

        st.chat_message("assistant").write_stream(
            stream_chat(
                prompt,
                st.session_state["thread_id"],
                st.session_state["user_id"],
                show_event,
                response_messages,
            )
        )
        progress.update(label="回答已完成", state="complete")
        full_response = "".join(response_messages).strip()
        st.session_state["message"].append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()

if __name__ == "__main__":
    pass
