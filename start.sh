#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "未找到虚拟环境 Python: $PYTHON_BIN"
  echo "请先创建虚拟环境并安装依赖。"
  exit 1
fi

exec "$PYTHON_BIN" -m streamlit run "$ROOT_DIR/app.py"
