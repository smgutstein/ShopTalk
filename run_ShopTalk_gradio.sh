#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this script,
# so relative package imports work consistently.
cd "$(dirname "$0")"

# Default Gradio binding:
#   - outside Docker: bind only to host localhost
#   - inside Docker: bind to all container interfaces so Docker port mapping works
if [[ -f /.dockerenv ]]; then
    DEFAULT_SERVER_NAME="0.0.0.0"
else
    DEFAULT_SERVER_NAME="127.0.0.1"
fi

DEFAULT_SERVER_PORT="${GRADIO_PORT:-7860}"

# Allow environment overrides.
SERVER_NAME="${SERVER_NAME:-${DEFAULT_SERVER_NAME}}"
SERVER_PORT="${SERVER_PORT:-${DEFAULT_SERVER_PORT}}"

python -m server.gradio_app \
    --server_name "${SERVER_NAME}" \
    --server_port "${SERVER_PORT}" \
    "$@"