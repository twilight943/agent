#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${AGENTREC_LLM_MODEL_PATH:-/data/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${AGENTREC_LLM_MODEL:-$MODEL_PATH}"
HOST="${AGENTREC_LLM_HOST:-127.0.0.1}"
PORT="${AGENTREC_LLM_PORT:-8000}"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --language-model-only \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.55
