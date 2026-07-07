#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m server.evals.score_retrieval_llm_judgments "$@"