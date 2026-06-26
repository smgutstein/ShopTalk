#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m server.evals.eval_llm_decision "$@"