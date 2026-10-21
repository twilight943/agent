#!/usr/bin/env bash
set -euo pipefail
ENVIRONMENT="${1:-server}"
export AGENTREC_ENV="$ENVIRONMENT"
uvicorn src.api.main:app --host "${AGENTREC_HOST:-0.0.0.0}" --port "${AGENTREC_PORT:-8080}"
