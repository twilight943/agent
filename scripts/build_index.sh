#!/usr/bin/env bash
set -euo pipefail
ENVIRONMENT="${1:-server}"
python -m src.embedding.build_faiss --env "$ENVIRONMENT"
