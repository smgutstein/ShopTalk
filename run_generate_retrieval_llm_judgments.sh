#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m server.evals.retrieval_llm_eval generate "$@"
