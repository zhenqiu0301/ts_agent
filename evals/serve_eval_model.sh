#!/usr/bin/env bash
# 评测模型服务部署脚本：为 evals/run_eval.py 准备一个 OpenAI 兼容端点。
#
# 用法：
#   evals/serve_eval_model.sh up              # 后台启动并等待就绪
#   evals/serve_eval_model.sh down            # 停止本脚本启动的服务
#   evals/serve_eval_model.sh status          # 查看端点与进程状态
#   evals/serve_eval_model.sh eval [参数...]  # 一键：启动 → 跑评测 → 自动关闭
#
# 典型例子：
#   TS_EVAL_MODEL=Qwen/Qwen3.5-2B evals/serve_eval_model.sh eval --category routing
#   EVAL_BACKEND=ollama TS_EVAL_MODEL=qwen3.5:2b evals/serve_eval_model.sh up
#   TS_EVAL_BASE_URL=http://gpu-server:8000/v1 TS_EVAL_MODEL=Qwen3.5-2B \
#     evals/serve_eval_model.sh eval --judge      # 远端已有服务：附加模式，只连不启
#
# 常用环境变量：
#   TS_EVAL_MODEL           模型名（HF 仓库名 / 本地权重路径 / Ollama 模型名），up 与 eval 必填
#   EVAL_BACKEND            vllm（默认，需 Linux + NVIDIA GPU）| ollama（macOS 可用）
#   VLLM_BIN                vllm 可执行文件路径（默认 vllm，可指向 conda 环境二进制）
#   VLLM_HOST / VLLM_PORT   本地 vLLM 地址（默认 127.0.0.1:8100，避开 API 服务端的 8000）
#   VLLM_TOOL_CALL_PARSER   vLLM tool-call 解析器（默认 hermes；Qwen3.5 系列用 qwen3_coder）
#   VLLM_SERVED_MODEL_NAME  覆盖 --served-model-name（用本地权重路径时建议设置成短名）
#   VLLM_MAX_MODEL_LEN      上下文长度（默认 8192，按 16GB 内存机器与客服场景商定）
#   VLLM_GPU_MEM_UTIL       内存上限占比（默认 0.6，约 9.6GB / 16GB）
#   VLLM_MAX_NUM_SEQS       并发序列数（默认 4，评测/单人使用足够）
#   VLLM_MAX_NUM_BATCHED_TOKENS  prefill 分块（默认 4096，控制内存峰值）
#   VLLM_DISABLE_MM         1（默认）= --language-model-only 关闭图片/视频输入；0 保留多模态
#   VLLM_EXTRA_ARGS         追加给 vllm serve 的额外参数
#   TS_EVAL_BASE_URL        指向已有服务时进入“附加模式”，up/eval 都不会启停它
#   EVAL_HEALTH_TIMEOUT     就绪等待秒数（默认 1800，首次下载权重较慢）
#   EVAL_POLL_INTERVAL      健康检查轮询间隔秒数（默认 5）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BACKEND="${EVAL_BACKEND:-vllm}"
TS_EVAL_MODEL="${TS_EVAL_MODEL:-}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
VLLM_DISABLE_MM="${VLLM_DISABLE_MM:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
HEALTH_TIMEOUT="${EVAL_HEALTH_TIMEOUT:-1800}"
POLL_INTERVAL="${EVAL_POLL_INTERVAL:-5}"

PID_FILE="$SCRIPT_DIR/.serve_eval_model.pid"
LOG_FILE="$SCRIPT_DIR/.serve_eval_model.log"
STARTED_HERE=0
EFFECTIVE_MODEL="$TS_EVAL_MODEL"

case "$BACKEND" in
  vllm) DEFAULT_BASE_URL="http://$VLLM_HOST:$VLLM_PORT/v1" ;;
  ollama) DEFAULT_BASE_URL="http://127.0.0.1:11434/v1" ;;
  *) echo "EVAL_BACKEND 只支持 vllm 或 ollama，当前：$BACKEND" >&2; exit 1 ;;
esac
BASE_URL="${TS_EVAL_BASE_URL:-$DEFAULT_BASE_URL}"

usage() {
  sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

probe() {
  curl -fsS --max-time 3 "$BASE_URL/models" 2>/dev/null
}

is_healthy() {
  probe >/dev/null 2>&1
}

require_model() {
  if [ -z "$TS_EVAL_MODEL" ]; then
    echo "需要通过 TS_EVAL_MODEL 指定模型，例如：TS_EVAL_MODEL=Qwen/Qwen3.5-2B" >&2
    exit 1
  fi
}

wait_healthy() {
  local waited=0
  printf '等待端点就绪 %s（超时 %ss）…\n' "$BASE_URL" "$HEALTH_TIMEOUT"
  while ! is_healthy; do
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      echo "等待超时。最近日志（${LOG_FILE}）：" >&2
      tail -n 30 "$LOG_FILE" >&2 || true
      exit 1
    fi
    sleep "$POLL_INTERVAL"
    waited=$((waited + POLL_INTERVAL))
    if [ $((waited % 60)) -eq 0 ]; then
      echo "  已等待 ${waited}s…"
    fi
  done
}

start_vllm() {
  if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
    echo "未找到 vllm 命令：$VLLM_BIN" >&2
    echo "可安装官方版（需 Linux + NVIDIA GPU）：pip install vllm" >&2
    echo "或用 VLLM_BIN 指向已有的 vllm 可执行文件（如 conda 环境）。" >&2
    exit 1
  fi
  if [ "$(uname -s)" != "Linux" ]; then
    echo "提示：当前不是 Linux，vLLM 需要 GPU/Metal 后端（如 vllm-metal 插件）；" >&2
    echo "若无后端支持，建议 EVAL_BACKEND=ollama 或用 TS_EVAL_BASE_URL 连远端 GPU 服务器。" >&2
  fi
  local served_name="${VLLM_SERVED_MODEL_NAME:-$TS_EVAL_MODEL}"
  echo "启动 vLLM：bin=$VLLM_BIN model=$TS_EVAL_MODEL served=$served_name addr=$VLLM_HOST:$VLLM_PORT"
  echo "参数：max_model_len=$VLLM_MAX_MODEL_LEN mem_util=$VLLM_GPU_MEM_UTIL seqs=$VLLM_MAX_NUM_SEQS batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS disable_mm=$VLLM_DISABLE_MM parser=$VLLM_TOOL_CALL_PARSER"
  echo "日志：$LOG_FILE"
  : > "$LOG_FILE"
  local args=()
  args+=(--host "$VLLM_HOST" --port "$VLLM_PORT")
  args+=(--served-model-name "$served_name")
  args+=(--enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_CALL_PARSER")
  args+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
  args+=(--gpu-memory-utilization "$VLLM_GPU_MEM_UTIL")
  args+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
  args+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
  if [ "$VLLM_DISABLE_MM" = "1" ]; then
    args+=(--language-model-only)
  fi
  if [ -n "$VLLM_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206  # VLLM_EXTRA_ARGS 有意按空白拆分为多个参数
    args+=($VLLM_EXTRA_ARGS)
  fi
  # shellcheck disable=SC2086
  nohup "$VLLM_BIN" serve "$TS_EVAL_MODEL" "${args[@]}" >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  EFFECTIVE_MODEL="$served_name"
}

start_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "未找到 ollama。请先安装：https://ollama.com" >&2
    exit 1
  fi
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "启动 ollama serve…"
    : > "$LOG_FILE"
    nohup ollama serve >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    local waited=0
    while ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
      if [ "$waited" -ge 20 ]; then
        echo "ollama serve 启动超时，日志：$LOG_FILE" >&2
        exit 1
      fi
      sleep 1
      waited=$((waited + 1))
    done
  fi
  if ! ollama list | awk '{print $1}' | grep -Fxq "$TS_EVAL_MODEL" \
    && ! ollama list | awk '{print $1}' | grep -Fxq "${TS_EVAL_MODEL}:latest"; then
    echo "拉取模型：ollama pull $TS_EVAL_MODEL"
    ollama pull "$TS_EVAL_MODEL"
  fi
  EFFECTIVE_MODEL="$TS_EVAL_MODEL"
}

start_backend() {
  if is_healthy; then
    STARTED_HERE=0
    echo "端点已就绪，进入附加模式（不会启动/关闭服务）：$BASE_URL"
    return 0
  fi
  STARTED_HERE=1
  case "$BACKEND" in
    vllm) start_vllm ;;
    ollama) start_ollama ;;
  esac
  wait_healthy
}

stop_backend() {
  if [ ! -f "$PID_FILE" ]; then
    if is_healthy; then
      echo "端点在运行但不是本脚本启动的（附加模式），如需停止请自行处理：$BASE_URL"
    else
      echo "没有由本脚本启动的评测服务。"
    fi
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "停止进程 $pid …"
  kill "$pid" 2>/dev/null || true
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge 20 ]; then
      kill -9 "$pid" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  rm -f "$PID_FILE"
  echo "已停止。完整日志见：$LOG_FILE"
}

cmd_status() {
  if is_healthy; then
    echo "端点可用：$BASE_URL"
    probe | python3 -c 'import json,sys; [print("  -", m["id"]) for m in json.load(sys.stdin).get("data", [])]' 2>/dev/null || true
  else
    echo "端点不可用：$BASE_URL"
  fi
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "本脚本托管进程：${pid}（日志 ${LOG_FILE}）"
    else
      echo "存在残留 PID 文件（进程已退出），可执行 down 清理"
    fi
  fi
}

run_eval_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run python -m evals.run_eval "$@"
  elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    "$REPO_ROOT/.venv/bin/python" -m evals.run_eval "$@"
  else
    echo "未找到 uv，且项目 .venv 不存在。请先安装 uv 并执行 uv sync。" >&2
    exit 1
  fi
}

cmd_eval() {
  require_model
  start_backend
  cleanup() {
    if [ "$STARTED_HERE" -eq 1 ]; then
      echo ""
      stop_backend
    fi
  }
  trap cleanup EXIT
  echo ""
  echo "运行评测：模型=$EFFECTIVE_MODEL 端点=$BASE_URL 参数=$*"
  (
    cd "$REPO_ROOT"
    TS_EVAL_MODEL="$EFFECTIVE_MODEL" TS_EVAL_BASE_URL="$BASE_URL" \
      TS_EVAL_API_KEY="${TS_EVAL_API_KEY:-empty}" run_eval_cmd "$@"
  )
}

ACTION="${1:-}"
if [ -z "$ACTION" ]; then
  usage
  exit 1
fi
shift

case "$ACTION" in
  up)
    require_model
    start_backend
    cmd_status
    echo ""
    echo "跑评测：TS_EVAL_MODEL=$EFFECTIVE_MODEL TS_EVAL_BASE_URL=$BASE_URL \\"
    echo "  uv run python -m evals.run_eval --category routing"
    ;;
  down)
    stop_backend
    ;;
  status)
    cmd_status
    ;;
  eval)
    cmd_eval "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "未知命令：$ACTION" >&2
    usage
    exit 1
    ;;
esac
