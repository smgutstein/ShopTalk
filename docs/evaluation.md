# Evaluation Workflows

ShopTalk evaluates the recommendation pipeline at three levels:

1. whether the current conversation turn requires a product search;
2. whether retrieved candidates support a recommendation or require another response;
3. whether the complete retrieval-and-response pipeline retrieves an appropriate product and produces a correct, grounded response.

The evaluation scripts require an OpenAI API key. Each evaluator uses an INI file that records the complete run configuration.

### Search-Decision Evaluation

```bash
./run_eval_search_decision.sh
```

This evaluates whether the LLM policy makes the correct pre-retrieval decision: search now, ask for clarification, or continue without another search.

The complete default run is defined in:

```text
server/evals/search_decision/search_decision_eval.ini
```

To reproduce a different run, copy and edit that file, then select it explicitly:

```bash
./run_eval_search_decision.sh --config path/to/search_decision_eval.ini
```

The evaluator accepts no per-setting command-line overrides. Results are written to the directory configured in the INI file; the default is:

```text
server/evals/search_decision/results/
```

### Product-Response-Decision Evaluation

```bash
./run_eval_product_response_decision.sh
```

This evaluates the post-retrieval LLM decision: recommend one of the retrieved products, ask for more information, or report that the request cannot be fulfilled by the available candidates.

The complete run definition is stored in:

```text
server/evals/product_response_decision/product_response_decision_eval.ini
```

Use a different complete run configuration with:

```bash
./run_eval_product_response_decision.sh --config path/to/product_response_decision_eval.ini
```

Results are written to numbered text files under:

```text
server/evals/product_response_decision/results/
```

### Retrieval and LLM Response Evaluation

The end-to-end evaluator measures retrieval quality together with the correctness and grounding of the final LLM response. Separate configurations are provided for text-only, image-only, and combined text-and-image requests.

Generate raw judgments for a modality with:

```bash
./run_generate_retrieval_llm_judgments.sh \
  server/evals/retrieval_llm_response/retrieval_llm_eval_text_only_45.ini
```

```bash
./run_generate_retrieval_llm_judgments.sh \
  server/evals/retrieval_llm_response/retrieval_llm_eval_image_only_45.ini
```

```bash
./run_generate_retrieval_llm_judgments.sh \
  server/evals/retrieval_llm_response/retrieval_llm_eval_text_plus_image_45.ini
```

Each generated file records the user request, retrieval behavior, retrieved candidates, selected product, final response, and evidence available to the LLM.

### Human Review and Scoring Workflow

End-to-end evaluation artifacts move through three directories:

```text
server/evals/retrieval_llm_response/generated/   Raw retrieval and model outputs
server/evals/retrieval_llm_response/reviewed/    Human-reviewed judgments
server/evals/retrieval_llm_response/results/     Scored text and JSON reports
```

After reviewing a generated judgment file, score it with:

```bash
./run_score_retrieval_llm_judgments.sh path/to/reviewed_judgments.json
```

The scorer reports retrieval success, target rank, reciprocal rank, product-decision correctness, response quality, grounding, unsupported or contradicted claims, and behavior on cases where no suitable catalog product exists.

Return to the [main README](../README.md).
