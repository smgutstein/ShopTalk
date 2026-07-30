# Repository Structure

The following tree identifies the main runtime, preprocessing, evaluation, test, configuration, and smoke-test files:

```text
artifacts/
  smoke_test_vector_db/
    faiss/
      embeddings.faiss                       Prebuilt smoke-test FAISS index
      product_ids.json                       Product IDs aligned with the smoke-test index

EDA/product_blurbs/
  smoke_blurb_dict.json                      Product metadata for the smoke-test catalog

smoke_test_images/                           Product images for the smoke-test catalog
smoke_images.csv                             Smoke-test image ID/path mappings

EDA/                                         Offline product-data preparation
  DescriptionGenerator.py                    Generate product descriptions / blurbs
  preprocessor.py                            Product metadata preprocessing helpers
  config.ini                                 EDA/preprocessing configuration

generate_vector_db.py                        Build ImageBind/FAISS vector artifacts

server/
  gradio_app.py                              Main Gradio application

  recommender_core/                          Primary runtime recommendation package
    config.py                                Runtime config dataclasses and INI loading
    conversation_policy.py                   LLM control and recommendation policy
    diagnostics.py                           Diagnostics data structures/helpers
    llm_prompts.py                           Prompt construction helpers
    parsing.py                               Query/embedding-mode parsing helpers
    product_candidate.py                     Product candidate representation
    product_images.py                        Product image-path loading helpers
    product_vector_store.py                  FAISS/product metadata wrapper
    query_embedder.py                        ImageBind query embedding wrapper
    recommender_factory.py                   Runtime construction functions
    reply_types.py                           Structured reply/result models
    shop_talk_recommender.py                 Main recommender orchestration class
    vector_db.py                             FAISS artifact loading
    vector_query.py                          Query-vector combination/search helpers

  evals/                                     Evaluation suites and supporting artifacts
    search_decision/
      eval_search_decision.py                Pre-retrieval search-decision evaluation
      cases/                                 Search-decision test cases

    product_response_decision/
      eval_product_response_decision.py      Post-retrieval product-response evaluation
      cases/                                 Product-response test cases

    retrieval_llm_response/
      retrieval_llm_eval.py                  Generate judgments and score reviewed files
      retrieval_llm_eval_*.ini               Modality-specific evaluation configurations
      cases/                                 Text, image, and text-plus-image cases
      generated/                             Raw generated judgment files
      reviewed/                              Human-reviewed judgment files
      results/                               Scored evaluation outputs
      query_images/                          Evaluation images and source provenance
      image_case_reviewer/                   Image-search and case-review application
      run_image_case_reviewer.py             Reviewer application entry point

    retrieval_llm_eval.py                    Compatibility wrapper
    generate_retrieval_llm_judgments.py      Compatibility wrapper
    score_retrieval_llm_judgments.py         Compatibility wrapper

  recommender.py                             Compatibility import wrapper
  gradio_images.py                           Compatibility import wrapper
  shoptalk_paths.py                          Compatibility import wrapper
  static/images/                             Runtime product images (generated/downloaded)

tests/                                       Unit and lightweight behavior tests

Dockerfiles/                                 Docker development environment
  Dockerfile                                 ShopTalk development image
  docker-compose.yaml                        Compose service definition
  docker-entrypoint.sh                       Container user/setup entrypoint
  requirements-docker.txt                    Docker copy of Python dependencies
  shoptalk_shell.sh                          Docker helper script
  .env-public                                Non-secret Docker environment defaults

images/                                      Architecture diagrams and editable sources

run_ShopTalk_gradio.sh                       Launch the Gradio app
run_eval_search_decision.sh                  Run the search-decision evaluation
run_eval_product_response_decision.sh        Run the product-response evaluation
run_generate_retrieval_llm_judgments.sh      Generate retrieval/response judgments
run_score_retrieval_llm_judgments.sh         Score reviewed retrieval/response judgments
run_image_case_reviewer.sh                   Launch the image-case reviewer

shoptalk_config.ini                          App/evaluation LLM model settings
shoptalk_smoke_config.ini                    Configuration for the committed smoke test
.env.example                                 Template for the local OpenAI API key
requirements.txt                             Python dependencies
setup.py                                     Package installation metadata
```

Return to the [main README](../README.md).
