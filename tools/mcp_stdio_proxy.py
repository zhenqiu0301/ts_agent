from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time


def _looks_like_protocol_json(line: str) -> bool:
    s = line.strip()
    if not s.startswith("{") and not s.startswith("["):
        return False
    try:
        obj = json.loads(s)
    except Exception:
        return False
    return isinstance(obj, (dict, list))


_LOG_DROPPED = os.getenv("MCP_PROXY_LOG_DROPPED", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
_DROP_LOG_MAX = int(os.getenv("MCP_PROXY_DROP_LOG_MAX", "20") or "20")
_DROP_LOG_WINDOW_SEC = float(os.getenv("MCP_PROXY_DROP_LOG_WINDOW_SEC", "10") or "10")
_drop_state = {"count": 0, "window_start": time.time(), "suppressed": 0}
_drop_lock = threading.Lock()


def _log_dropped_line(line: str) -> None:
    """按时间窗口限流打印被丢弃的非 JSON-RPC 行。默认关闭。"""
    if not _LOG_DROPPED:
        return
    now = time.time()
    with _drop_lock:
        elapsed = now - _drop_state["window_start"]
        if elapsed >= _DROP_LOG_WINDOW_SEC:
            if _drop_state["suppressed"] > 0:
                sys.stderr.write(
                    f"[mcp-proxy] suppressed {_drop_state['suppressed']} dropped lines in last "
                    f"{_DROP_LOG_WINDOW_SEC:.0f}s\n"
                )
                sys.stderr.flush()
            _drop_state["window_start"] = now
            _drop_state["count"] = 0
            _drop_state["suppressed"] = 0

        if _drop_state["count"] < _DROP_LOG_MAX:
            sys.stderr.write(f"[mcp-proxy] drop non-jsonrpc: {line}")
            sys.stderr.flush()
            _drop_state["count"] += 1
        else:
            _drop_state["suppressed"] += 1


def _forward_stdout_filtering(child: subprocess.Popen[bytes]) -> None:
    if child.stdout is None:
        return
    for raw in iter(child.stdout.readline, b""):
        line = raw.decode("utf-8", errors="ignore")
        if _looks_like_protocol_json(line):
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
        else:
            _log_dropped_line(line)


def _forward_stderr(child: subprocess.Popen[bytes]) -> None:
    if child.stderr is None:
        return
    for raw in iter(child.stderr.readline, b""):
        try:
            sys.stderr.buffer.write(raw)
            sys.stderr.buffer.flush()
        except Exception:
            pass


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: mcp_stdio_proxy.py <command> [args...]\n")
        return 2

    cmd = sys.argv[1:]
    try:
        child = subprocess.Popen(
            cmd,
            stdin=sys.stdin.buffer,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except Exception as exc:
        sys.stderr.write(f"[mcp-proxy] failed to start child: {exc}\n")
        return 1

    t_out = threading.Thread(target=_forward_stdout_filtering, args=(child,), daemon=False)
    t_err = threading.Thread(target=_forward_stderr, args=(child,), daemon=False)
    t_out.start()
    t_err.start()

    code = child.wait()
    t_out.join(timeout=1)
    t_err.join(timeout=1)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
