#!/usr/bin/env bash
set -euo pipefail
ENVIRONMENT="${1:-server}"
python -m src.recsys.train_ranker --env "$ENVIRONMENT"
