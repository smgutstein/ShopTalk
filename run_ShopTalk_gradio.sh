#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this script,
# so relative package imports work consistently.
cd "$(dirname "$0")"

python -m server.gradio_app "$@"