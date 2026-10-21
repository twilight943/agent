#!/usr/bin/env bash
set -euo pipefail
ENVIRONMENT="${1:-server}"
python -m src.data.preprocess --env "$ENVIRONMENT"
