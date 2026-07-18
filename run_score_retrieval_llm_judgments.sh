#!/usr/bin/env bash
set -euo pipefail

# Always run from the repository root so relative paths in the selected INI file
# resolve consistently, regardless of the caller's current working directory.
cd "$(dirname "$0")"

# With judgments=auto, the Python workflow finds the newest reviewed file whose
# prefix matches the selected evaluation config and writes its report to results/.
# Any arguments supplied here (normally positional CONFIG and optionally
# --allow-unjudged) are passed through unchanged.
python -m server.evals.retrieval_llm_eval score "$@"
