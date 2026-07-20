#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this script so relative paths in the
# selected configuration file resolve from the repository root.
cd "$(dirname "$0")"

exec python -m server.gradio_app "$@"
