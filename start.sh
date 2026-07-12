#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv。请先安装 uv 并执行 uv sync。"
  exit 1
fi

cd "$ROOT_DIR"
exec uv run streamlit run "$ROOT_DIR/app.py"
