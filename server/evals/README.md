# ShopTalk evaluation layout

The eval package is organized by the part of the ShopTalk pipeline being tested.
This keeps source files, hand-authored cases, generated results, and reviewed
artifacts from accumulating in one flat directory.

## `search_decision/`

Evaluates the pre-retrieval decision: whether the conversation policy should run
a product search at all.

- `eval_search_decision.py` is the evaluator implementation.
- `search_decision_eval.ini` defines the complete default run.
- `cases/` contains hand-authored JSONL cases.
- `results/` contains generated reports and can usually be deleted/regenerated.

## `product_response_decision/`

Evaluates the post-retrieval response-mode decision: recommend a product, ask the
user to clarify/dive deeper, or say no suitable item was found.

- `eval_product_response_decision.py` is the evaluator implementation.
- `product_response_decision_eval.ini` defines the complete default run.
- `cases/` contains hand-authored JSONL cases.
- `results/` contains generated reports and can usually be deleted/regenerated.

## `retrieval_llm_response/`

Runs full retrieval plus final LLM response generation, producing human-editable
judgment files and metrics reports.

- `retrieval_llm_eval.py` is the unified generate/score implementation.
- Each modality has its own required `retrieval_llm_eval_*_45.ini` config.
- `cases/` contains hand-authored JSONL cases.
- `generated/` contains immutable numbered judgment files produced by eval runs.
- `reviewed/` receives matching editable copies for human judgment.
- `results/` contains metrics reports whose names identify the reviewed input.
- `[score] judgments = auto` selects the newest reviewed copy matching the
  configured evaluation prefix, so filenames do not need to be copied into the INI.

## Entry points

The root-level shell scripts provide convenient entry points for the two
policy evaluations:

- `run_eval_search_decision.sh` runs
  `server.evals.search_decision.eval_search_decision`.
- `run_eval_product_response_decision.sh` runs
  `server.evals.product_response_decision.eval_product_response_decision`.

The root-level retrieval/response scripts call
`server.evals.retrieval_llm_response.retrieval_llm_eval` directly. Both require
a modality-specific INI path as the first positional argument.
