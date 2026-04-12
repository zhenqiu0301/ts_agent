from __future__ import annotations

import asyncio
import threading
import os
import sys
import time
import shutil
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.logger_handler import logger

DEFAULT_PRICE_COMPARE_MCP_TOOLS = (
    "jd.goods.query",
    "pdd.goods.search",
)


def _flatten_exception_messages(exc: BaseException) -> list[str]:
    """展开异常组，提取叶子异常文本。"""
    children = getattr(exc, "exceptions", None)
    if children:
        messages: list[str] = []
        for sub in children:
            messages.extend(_flatten_exception_messages(sub))
        return messages
    msg = str(exc).strip()
    return [msg] if msg else [exc.__class__.__name__]


def _format_compact_error(exc: BaseException) -> str:
    messages = _flatten_exception_messages(exc)
    seen: set[str] = set()
    uniq = []
    for m in messages:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return " | ".join(uniq[:3]) if uniq else exc.__class__.__name__


class MCPToolLister:
    """仅负责连接 MCP server 并拉取工具列表。"""

    def __init__(self, config_path: str = "config/mcp.yml", timeout_seconds: float = 15.0):
        base_dir = Path(__file__).resolve().parent.parent
        p = Path(config_path)
        self.config_path = str(p if p.is_absolute() else (base_dir / p))
        self.timeout_seconds = timeout_seconds
        self._client: MultiServerMCPClient | None = None
        self._tool_names: list[str] = []
        self._tool_objects: list[Any] = []
        self._lock = threading.Lock()
        self._max_retries = int(os.getenv("MCP_MAX_RETRIES", "3") or "3")
        self._failure_count = 0
        self._disabled = False
        self._disable_cooldown_sec = float(
            os.getenv("MCP_DISABLE_COOLDOWN_SEC", "90") or "90"
        )
        self._disabled_until = 0.0

    def _is_temporarily_disabled(self) -> bool:
        if self._disabled:
            return True
        if self._disabled_until <= 0:
            return False
        if time.time() < self._disabled_until:
            return True
        self._disabled_until = 0.0
        self._failure_count = 0
        logger.info("[mcp_tools] MCP临时禁用冷却结束，自动恢复重连。")
        return False

    def _mark_success(self) -> None:
        self._failure_count = 0
        self._disabled_until = 0.0
        self._disabled = False

    def _mark_failure(self, reason: str) -> None:
        self._failure_count += 1
        logger.warning(
            f"[mcp_tools] MCP重连失败({self._failure_count}/{self._max_retries}): {reason}"
        )
        if self._failure_count >= self._max_retries:
            self._client = None
            self._tool_names = []
            self._tool_objects = []
            if self._disable_cooldown_sec > 0:
                self._disabled_until = time.time() + self._disable_cooldown_sec
                self._failure_count = 0
                logger.warning(
                    "[mcp_tools] MCP重连超过上限，已临时禁用 "
                    f"{int(self._disable_cooldown_sec)}s，冷却后会自动重试。"
                )
            else:
                self._disabled = True
                logger.warning("[mcp_tools] MCP重连超过上限，已禁用本进程内外部MCP工具。")

    def _load_servers(self) -> dict[str, dict[str, Any]]:
        if os.getenv("MCP_DISABLE_EXTERNAL", "").strip().lower() in {"1", "true", "yes"}:
            logger.warning("[mcp_tools] 已通过 MCP_DISABLE_EXTERNAL 禁用外部 MCP。")
            return {}

        path = Path(self.config_path)
        if not path.exists():
            logger.warning(f"[mcp_tools] 配置文件不存在: {self.config_path}")
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            mcp_tools = data.get("mcp_tools", {})
            if not isinstance(mcp_tools, dict):
                return {}

            normalized: dict[str, dict[str, Any]] = {}
            repo_root = Path(__file__).resolve().parent.parent
            for name, conf in mcp_tools.items():
                if not isinstance(name, str) or not isinstance(conf, dict):
                    continue
                item = dict(conf)
                transport = str(item.get("transport", "streamable_http")).strip()
                # langchain-mcp-adapters 当前版本中，stdio 不支持 request_timeout 参数
                if transport == "stdio":
                    item.pop("request_timeout", None)
                    command = item.get("command")
                    if isinstance(command, str) and command.strip() == "python":
                        item["command"] = sys.executable
                    # DNS 预检查：域名不可解析时直接跳过，避免反复起子进程刷错误日志。
                    env = item.get("env", {})
                    if isinstance(env, dict):
                        if "PATH" not in env:
                            env["PATH"] = os.getenv("PATH", "")
                        if "HOME" not in env:
                            env["HOME"] = os.getenv("HOME", "")
                        item["env"] = env
                        raw_url = str(env.get("ENV_URL", "")).strip()
                        host = urlparse(raw_url).hostname if raw_url else ""
                        if host:
                            try:
                                socket.getaddrinfo(host, None)
                            except OSError:
                                logger.warning(
                                    f"[mcp_tools] server '{name}' ENV_URL 域名解析失败(将继续尝试): {host}"
                                )
                    # stdio 下相对路径按项目根目录解析，避免 cwd 变化导致启动失败
                    args = item.get("args", [])
                    if isinstance(args, list) and args:
                        resolved_args: list[Any] = []
                        for arg in args:
                            if isinstance(arg, str):
                                # 常见场景：streamlit/IDE 启动时 PATH 不完整，node 找不到。
                                if arg.strip() == "node":
                                    node_path = shutil.which("node")
                                    if not node_path:
                                        node_path = os.getenv("NODE_BINARY", "").strip() or None
                                    if not node_path:
                                        for candidate in ("/opt/homebrew/bin/node", "/usr/local/bin/node"):
                                            if Path(candidate).exists():
                                                node_path = candidate
                                                break
                                    if node_path:
                                        resolved_args.append(node_path)
                                        continue
                                p = Path(arg)
                                if not p.is_absolute():
                                    cand = repo_root / p
                                    if cand.exists():
                                        resolved_args.append(str(cand))
                                        continue
                            resolved_args.append(arg)
                        item["args"] = resolved_args
                normalized[name] = item
            return normalized
        except Exception as e:
            logger.warning(f"[mcp_tools] 读取配置失败: {str(e)}")
            return {}

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if not has_running_loop:
            try:
                return asyncio.run(coro)
            except Exception as e:
                logger.warning(f"[mcp_tools] 异步执行失败: {str(e)}")
                return None

        holder: dict[str, Any] = {"value": None, "error": None}

        def runner():
            try:
                holder["value"] = asyncio.run(coro)
            except Exception as e:
                holder["error"] = e

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=self.timeout_seconds + 1)

        if t.is_alive():
            logger.warning("[mcp_tools] 拉取工具列表超时")
            return None
        if holder["error"] is not None:
            logger.warning(f"[mcp_tools] 拉取工具列表失败: {holder['error']}")
            return None
        return holder["value"]

    async def _refresh_async(self) -> list[str]:
        if self._is_temporarily_disabled():
            return []
        servers = self._load_servers()
        if not servers:
            self._mark_failure("无可用MCP server配置")
            return []
        try:
            if self._client is None:
                self._client = MultiServerMCPClient(servers)
            tools = await asyncio.wait_for(self._client.get_tools(), timeout=self.timeout_seconds)
        except Exception as e:
            self._mark_failure(_format_compact_error(e))
            # 常见断链异常下重建 client，避免后续重试一直复用坏连接。
            self._client = None
            return []
        self._mark_success()
        self._tool_objects = list(tools)
        names = []
        for tool in tools:
            name = str(getattr(tool, "name", "")).strip()
            if name:
                names.append(name)
        self._tool_names = sorted(set(names))
        if self._tool_names:
            logger.info("[mcp_tools] 已拉取外部 MCP 工具列表:")
            for name in self._tool_names:
                logger.info(f"[mcp_tools] - {name}")
        else:
            logger.info("[mcp_tools] 外部 MCP 工具列表为空。")
        return self._tool_names

    def list_tools(self, refresh: bool = False) -> list[str]:
        with self._lock:
            if self._is_temporarily_disabled():
                return []
            if refresh or not self._tool_names:
                names = self._run_async(self._refresh_async())
                return names if isinstance(names, list) else []
            return list(self._tool_names)

    def get_tool_objects(self, refresh: bool = False) -> list[Any]:
        with self._lock:
            if self._is_temporarily_disabled():
                return []
            if refresh or not self._tool_objects:
                self._run_async(self._refresh_async())
            return list(self._tool_objects)


_lister = MCPToolLister()


def list_mcp_tools(refresh: bool = False) -> list[str]:
    """项目唯一外部 MCP 能力：拉取工具列表。"""
    return _lister.list_tools(refresh=refresh)


def get_mcp_tool_objects(refresh: bool = False) -> list[Any]:
    """返回可直接传给 LangChain Agent 的 MCP 工具对象列表。"""
    return _lister.get_tool_objects(refresh=refresh)


def _keep_price_compare_tools_by_whitelist(mcp_tools: list[Any]) -> list[Any]:
    """外部 MCP 工具仅保留价格比较白名单。"""
    env_whitelist = os.getenv("MCP_PRICE_COMPARE_TOOL_WHITELIST", "").strip()
    if env_whitelist:
        wanted_names = {n.strip() for n in env_whitelist.split(",") if n.strip()}
    else:
        wanted_names = set(DEFAULT_PRICE_COMPARE_MCP_TOOLS)

    selected = []
    for tool in mcp_tools:
        name = str(getattr(tool, "name", "")).strip()
        if name in wanted_names:
            selected.append(tool)
    selected.sort(key=lambda t: str(getattr(t, "name", "")))
    return selected


def _run_awaitable_safely(awaitable):
    """在同步上下文安全执行 awaitable，避免事件循环冲突。"""
    try:
        asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        has_running_loop = False

    if not has_running_loop:
        return asyncio.run(awaitable)

    holder: dict[str, Any] = {"value": None, "error": None}

    def _runner():
        try:
            holder["value"] = asyncio.run(awaitable)
        except Exception as e:
            holder["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        raise TimeoutError("MCP工具调用超时")
    if holder["error"] is not None:
        raise holder["error"]
    return holder["value"]


def _syncify_mcp_tool(tool: Any):
    """将仅支持 ainvoke 的 MCP 工具包装为可同步调用。"""
    name = str(getattr(tool, "name", "")).strip()
    if not name:
        return tool

    async_callable = getattr(tool, "ainvoke", None)
    if not callable(async_callable):
        return tool

    def _call_sync(**kwargs):
        try:
            return _run_awaitable_safely(tool.ainvoke(kwargs))
        except Exception as e:
            msg = _format_compact_error(e)
            return f"[MCP:{name}] 调用失败：{msg}"

    return StructuredTool.from_function(
        func=_call_sync,
        name=name,
        description=str(getattr(tool, "description", "") or f"MCP tool: {name}"),
        args_schema=getattr(tool, "args_schema", None),
        infer_schema=getattr(tool, "args_schema", None) is None,
    )


def get_sync_price_compare_mcp_tools(refresh: bool = False) -> list[Any]:
    """获取已白名单过滤、且同步可调用的价格比较 MCP 工具列表。"""
    raw = get_mcp_tool_objects(refresh=refresh)
    filtered = _keep_price_compare_tools_by_whitelist(raw)
    return [_syncify_mcp_tool(t) for t in filtered]


def smoke_test_selected_tools(max_retries: int = 6) -> dict[str, str]:
    """
    仅用于本地联调：验证已保留的两个价格工具是否可调用。
    返回 {tool_name: status}，status 为 OK / ERR:...
    """
    target_names = ("jd.goods.query", "pdd.goods.search")
    payloads: dict[str, dict[str, Any]] = {
        # 关键词搜索更符合当前选购场景的低门槛调用方式
        "jd.goods.query": {"keyword": "扫地机器人"},
        "pdd.goods.search": {"keyword": "扫地机器人"},
    }

    async def _call_one(tool: Any, payload: dict[str, Any]) -> str:
        out = await tool.ainvoke(payload)
        return str(out)[:300]

    results: dict[str, str] = {}
    for tool_name in target_names:
        last_err = "tools not loaded"
        for _ in range(max_retries):
            tools = get_mcp_tool_objects(refresh=True)
            mapping = {str(getattr(t, "name", "")).strip(): t for t in tools}
            tool = mapping.get(tool_name)
            if tool is None:
                last_err = "tool missing in fetched list"
                time.sleep(1)
                continue
            try:
                out = asyncio.run(_call_one(tool, payloads[tool_name]))
                results[tool_name] = f"OK: {out}"
                break
            except Exception as e:
                last_err = str(e)[:300]
                time.sleep(1)
        if tool_name not in results:
            results[tool_name] = f"ERR: {last_err}"
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        res = smoke_test_selected_tools()
        for k, v in res.items():
            print(f"{k} => {v}")
        sys.stdout.flush()
        os._exit(0)

    tools = list_mcp_tools(refresh=True)
    print(f"count={len(tools)}")
    for name in tools:
        print(name)
    # 某些 MCP server 关闭时会遗留后台线程，直接退出避免调试脚本出现解释器关停噪音。
    sys.stdout.flush()
    os._exit(0)
