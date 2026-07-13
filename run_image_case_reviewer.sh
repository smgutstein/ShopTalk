#!/usr/bin/env bash
set -euo pipefail

# Always run from the repository root so ``server`` is importable as a package
# and all repository-relative paths behave consistently.
cd "$(dirname "$0")"

# Match run_ShopTalk_gradio.sh:
#   * on the host, bind only to localhost;
#   * inside Docker, bind to every container interface so the Compose port
#     mapping can forward browser traffic from the host.
#
# This affects only network reachability. Files are written to the same
# repository paths in either environment because Docker bind-mounts the project
# at /workspace.
if [[ -f /.dockerenv ]]; then
    DEFAULT_SERVER_NAME="0.0.0.0"
else
    DEFAULT_SERVER_NAME="127.0.0.1"
fi

# The Python config loader gives this environment variable precedence over the
# INI's host default. An explicit environment override remains possible for an
# unusual local setup.
export IMAGE_REVIEWER_SERVER_NAME="${IMAGE_REVIEWER_SERVER_NAME:-${DEFAULT_SERVER_NAME}}"

python -m server.evals.retrieval_llm_response.run_image_case_reviewer "$@"
