# ShopTalk evaluation layout

The eval package is organized by the part of the ShopTalk pipeline being tested.
This keeps source files, hand-authored cases, generated results, and reviewed
artifacts from accumulating in one flat directory.

## `search_decision/`

Evaluates the pre-retrieval decision: whether the conversation policy should run
a product search at all.

- `eval_search_decision.py` is the evaluator implementation.
- `cases/` contains hand-authored JSONL cases.
- `results/` contains generated reports and can usually be deleted/regenerated.

## `product_response_decision/`

Evaluates the post-retrieval response-mode decision: recommend a product, ask the
user to clarify/dive deeper, or say no suitable item was found.

- `eval_product_response_decision.py` is the evaluator implementation.
- `cases/` contains hand-authored JSONL cases.
- `results/` contains generated reports and can usually be deleted/regenerated.

## `retrieval_llm_response/`

Runs full retrieval plus final LLM response generation, producing human-editable
judgment files and metrics reports.

- `retrieval_llm_eval.py` is the unified generate/score implementation.
- `retrieval_llm_eval.ini` is the normal config for this eval.
- `cases/` contains hand-authored JSONL cases.
- `results/` contains generated judgment/metrics files.
- `reviewed/` is reserved for manually reviewed judgment files, which are more
  valuable than unreviewed generated outputs.

## Compatibility wrappers

The old module paths remain as thin wrappers:

- `server.evals.eval_search_decision`
- `server.evals.eval_llm_decision`
- `server.evals.retrieval_llm_eval`

Those wrappers keep existing shell scripts and old notes working while the real
implementations live in the structured subdirectories.
