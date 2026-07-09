"""Compatibility wrapper for the relocated search-decision evaluator.

The real implementation now lives under ``server.evals.search_decision`` so the
``evals`` package can group source, cases, and generated results by eval type.
This wrapper preserves older commands and shell scripts that still execute:

    python -m server.evals.eval_search_decision
"""

from __future__ import annotations

from .search_decision.eval_search_decision import main


if __name__ == "__main__":
    raise SystemExit(main())
