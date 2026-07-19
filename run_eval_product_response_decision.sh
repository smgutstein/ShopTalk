#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m server.evals.product_response_decision.eval_product_response_decision "$@"