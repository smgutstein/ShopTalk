"""Compatibility wrapper for the relocated product-response-decision evaluator.

The real implementation now lives under
``server.evals.product_response_decision``. This wrapper preserves older
commands and shell scripts that still execute:

    python -m server.evals.eval_llm_decision
"""

from __future__ import annotations

from .product_response_decision.eval_product_response_decision import main


if __name__ == "__main__":
    raise SystemExit(main())
