#!/usr/bin/env bash
# 启动 API 服务端（FastAPI + 共享 Agent）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

API_HOST="${TS_AGENT_API_HOST:-127.0.0.1}"
API_PORT="${TS_AGENT_API_PORT:-8000}"

if command -v uv >/dev/null 2>&1; then
  cd "$ROOT_DIR"
  exec uv run uvicorn ts_agent.server.app:app --host "$API_HOST" --port "$API_PORT"
elif [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  # 无 uv 时直接使用项目虚拟环境（需已执行过 uv sync / pip install）
  cd "$ROOT_DIR"
  exec "$ROOT_DIR/.venv/bin/python" -m uvicorn ts_agent.server.app:app \
    --host "$API_HOST" --port "$API_PORT"
else
  echo "未找到 uv，且项目 .venv 不存在。请先安装 uv 并执行 uv sync。" >&2
  exit 1
fi
