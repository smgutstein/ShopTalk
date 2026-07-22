#!/usr/bin/env bash
set -euo pipefail

# Always run from the repository root so relative paths in the selected INI file
# resolve consistently, regardless of the caller's current working directory.
cd "$(dirname "$0")"

# The Python workflow writes both an immutable generated file and a linked,
# editable copy in reviewed/. Any arguments supplied here (normally a positional CONFIG path)
# are passed through unchanged.
python -m server.evals.retrieval_llm_response.retrieval_llm_eval generate "$@"
