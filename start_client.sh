#!/usr/bin/env bash
# 启动 Streamlit 聊天客户端（连接 ts_agent API 服务端）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 未显式指定 API 地址时，按服务端脚本的同一组环境变量拼出默认地址
export TS_AGENT_API_URL="${TS_AGENT_API_URL:-http://${TS_AGENT_API_HOST:-127.0.0.1}:${TS_AGENT_API_PORT:-8000}}"

if command -v uv >/dev/null 2>&1; then
  cd "$ROOT_DIR"
  exec uv run streamlit run "$ROOT_DIR/app.py"
elif [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  # 无 uv 时直接使用项目虚拟环境（需已执行过 uv sync / pip install）
  cd "$ROOT_DIR"
  exec "$ROOT_DIR/.venv/bin/python" -m streamlit run "$ROOT_DIR/app.py"
else
  echo "未找到 uv，且项目 .venv 不存在。请先安装 uv 并执行 uv sync。" >&2
  exit 1
fi
