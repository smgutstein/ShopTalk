"""Compatibility wrapper for the relocated retrieval/LLM response evaluator.

The real implementation now lives under ``server.evals.retrieval_llm_response``.
This wrapper preserves older commands and shell scripts that still execute:

    python -m server.evals.retrieval_llm_eval generate
    python -m server.evals.retrieval_llm_eval score
"""

from __future__ import annotations

from .retrieval_llm_response.retrieval_llm_eval import (  # noqa: F401
    DEFAULT_EVAL_CONFIG_PATH,
    generate_main,
    main,
    score_main,
)


if __name__ == "__main__":
    raise SystemExit(main())
