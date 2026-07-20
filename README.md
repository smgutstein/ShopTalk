# ShopTalk - A Multimodal RAG-Style Product Recommendation Chatbot

## Overview

This repository implements a RAG-style product recommendation chatbot. It retrieves candidate products from a FAISS vector database built from multimodal product artifacts. Then, it uses an LLM to generate grounded, personality-aware recommendations.

## Core Questions

1. When using a joint text-image representaional space for a RAG-style product recommendation chatbot, does adding images to a request help produce accurate suggestions for a user?
2. Does a shared multimodal text/image representation space produce better product retrieval than separate unimodal representation spaces?
3. Can a weighted hybrid approach produce retrieval results that better match a user's preferences and constraints?


## Current Project Status

### Implemented

- Product metadata preprocessing and product blurb generation under `EDA/`.
- Multimodal vector database generation in `generate_vector_db.py`.
- ImageBind-based product embeddings using a shared text/image representation space.
- FAISS artifact generation for serving-time vector search.
- Optional NumPy artifact generation for inspection and debugging.
- Gradio application with:
  - pure text query input, pure image query input, or both
  - product recommendation display,
  - top retrieved product gallery,
  - diagnostics panel.
- Configurable LLM model and temperature via `shoptalk_config.ini`.
- Runtime launcher scripts for the Gradio app and eval modules.
- Unit tests for preprocessing, vector DB helpers, recommender helpers, Gradio helper behavior, image path handling, diagnostics, and vector-store behavior.
- Two targeted LLM evaluation modules:
  - search-decision evaluation: should the assistant search now, ask a clarifying question, or continue without search?
  - product-decision evaluation: given retrieved products, should the assistant recommend a product or ask for more information?
- Numbered human-readable eval result files, so repeated eval runs do not overwrite previous results.
- Completed an initial 45-case comparison of text-only and text-plus-image requests using the current shared ImageBind vector store.

### Still Planned / In Progress

- Add a smaller smoke-test dataset or documented fixture path so reviewers can run a minimal demo without reconstructing the full product artifact set.
- Answer 2nd & 3rd core questions
  - Compare the current shared ImageBind representation approach against separate text and image representation spaces.

## Architecture Overview

At a high level, ShopTalk has two components.

### Offline / Artifact Generation

![ShopTalk offline artifact generation](./images/shoptalk_offline_artifacts.svg)

The current vector generation path embeds product text and product images with ImageBind, combines the normalized text and image embeddings, normalizes the resulting product vector, and writes vector-search artifacts.

The FAISS backend writes:

```text
artifacts/vector_db/faiss/embeddings.faiss
artifacts/vector_db/faiss/product_ids.json
```

The NumPy backend writes:

```text
artifacts/vector_db/numpy/embeddings.npy
artifacts/vector_db/numpy/product_ids.json
```

The FAISS backend is the serving backend. The NumPy backend is mainly for debugging, inspection, and reproducibility checks.

### Runtime / Conversation Flow

![ShopTalk runtime conversational search flow](./images/shoptalk_runtime_flow.svg)

The LLM layer is not supposed to invent products. It works from retrieved product candidates and decides whether to recommend a product or gather more information.

## Repository Layout

Important files and directories:

```text
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
  requirements-docker.txt                    Docker-specific Python dependencies
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
.env.example                                 Template for the local OpenAI API key
requirements.txt                             Python dependencies
setup.py                                     Package installation metadata
```

## Generated / Downloaded Files Not Tracked by Git

The repository intentionally does not include the full image archive, generated product blurb JSON, image metadata CSV, or vector database artifacts.

Expected local paths:

| Path | Purpose | Source |
|---|---|---|
| `EDA/product_blurbs/combined_blurb_dict.json` | Product metadata and generated product blurbs keyed by product ID | Generated by the EDA pipeline |
| `images.csv` | Mapping from Amazon image IDs to relative image paths | Amazon Berkeley Objects metadata |
| `server/static/images/` | Product image files used by ImageBind and the UI | Extracted from Amazon Berkeley Objects image archive |
| `artifacts/vector_db/` | Generated FAISS/NumPy vector artifacts | Created by `generate_vector_db.py` |
| `.env` | Shared local OpenAI API key for both Docker and conda/local runs | Created locally from `.env.example` |

These are local artifacts, not source files.

## Environment Setup

There are two supported ways to create the ShopTalk development environment:

1. **Docker**, which is the preferred path for most development and review because it packages the Torch/ImageBind/FAISS stack in a repeatable container.
2. **Conda**, which is useful when you want direct local control of the Python environment.

The project has a few heavy dependencies. ImageBind and FAISS are the two most likely sources of environment friction, which is why the Docker path is listed first.

### Recommended: Docker Development Environment

A Docker-based development environment is provided under `Dockerfiles/`. This is the preferred path if you want to avoid repeatedly rebuilding the local Torch/ImageBind/FAISS environment by hand. It is still a development container, not a production deployment.

From the repository root:

```bash
cd Dockerfiles
./shoptalk_shell.sh
```

With no argument, `shoptalk_shell.sh` builds and starts the Compose service if needed, then opens an interactive shell inside the container.

Common Docker helper commands:

```bash
.Dockerfiles/shoptalk_shell.sh shell      # start if needed, then open an interactive shell
.Dockerfiles/shoptalk_shell.sh start      # build/start the container in the background
.Dockerfiles/shoptalk_shell.sh status     # show Compose service status
.Dockerfiles/shoptalk_shell.sh logs       # follow container logs
.Dockerfiles/shoptalk_shell.sh stop       # stop the container but keep it available
.Dockerfiles/shoptalk_shell.sh down       # stop and remove the Compose container/network
.Dockerfiles/shoptalk_shell.sh restart    # recreate the container
```

<!--
OpenAI-backed app and eval features require one shared local API-key file at the repository root. From the repository root, create it with:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```text
OPENAI_API_KEY="your_openai_api_key"
```

Both the Docker workflow and the conda/local workflow use this same root-level `.env` file for API keys. Docker also uses `Dockerfiles/.env-public` for non-secret Compose/container defaults; you normally should not need to edit it.

After entering the container shell, launch the Gradio app from the mounted repository root:

```bash
./run_ShopTalk_gradio.sh shoptalk_config.ini
```

The Gradio bind address and port are read from the `[server]` section of
`shoptalk_config.ini`. The checked-in configuration uses `0.0.0.0:7860` so
Docker port mapping can reach the app.

Docker does not remove the need for the generated/local artifacts described below. The image archive, `images.csv`, generated product blurbs, and FAISS vector DB still need to exist at the expected mounted paths before the full app can run.
-->

### Alternative: Conda Environment

Use the conda path if you specifically want a local non-container environment.

#### 1. Create and activate a conda environment

From the parent directory that will contain both `ImageBind` and this repository:

```bash
conda create --name imagebind python=3.10 -y
conda activate imagebind
```

#### 2. Install ImageBind

```bash
git clone https://github.com/facebookresearch/ImageBind
cd ImageBind
pip install .
```

#### 3. Install ShopTalk

Move to the ShopTalk repository root:

```bash
cd ../ShopTalk
pip install -e .
```

`pip install -e .` uses `setup.py` and `requirements.txt`.

Current warning: `requirements.txt` is oriented toward the current development environment and includes `faiss-gpu-cu12`. If your machine does not support that FAISS package, install the appropriate FAISS package for your environment instead. For CPU-only testing, `faiss-cpu` is usually the simpler choice.

Keep `requirements.txt` and `Dockerfiles/requirements-docker.txt` synchronized as the app, eval, and test dependencies change.

## Data Setup

ShopTalk expects Amazon Berkeley Objects-style images and metadata.

### 1. Download images and metadata

Download [`abo-images-small.tar`](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar) from the Amazon Berkeley Objects dataset.

Extract it and move the contents of:

```text
images/small/
```

into:

```text
server/static/images/
```

After extraction, the project should contain directories like:

```text
server/static/images/00/
server/static/images/01/
...
server/static/images/ff/
```

Also place `images.csv` in the project root. If you have `images.csv.gz`, unzip it:

```bash
gunzip images.csv.gz
```

The project root should then contain:

```text
images.csv
```

### 2. Generate product blurbs

The vector DB generator expects:

```text
EDA/product_blurbs/combined_blurb_dict.json
```

A typical generation command is:

```bash
python EDA/DescriptionGenerator.py
```

Check `EDA/config.ini` and the EDA scripts for the exact local input/output paths expected by your dataset layout.

## Generate Vector DB Artifacts

From the project root, generate the default FAISS artifacts:

```bash
python generate_vector_db.py \
  --vector_backend faiss \
  --vector_db_output_dir artifacts/vector_db
```

Useful options:

```bash
python generate_vector_db.py --help
```

Common flags:

| Flag | Meaning |
|---|---|
| `--product_blurbs` | Path to generated product blurb JSON. Default: `EDA/product_blurbs/combined_blurb_dict.json` |
| `--image_root` | Root directory containing product images. Default: `server/static/images` |
| `--images_csv` | Path to image ID to image path CSV. Default: `images.csv` |
| `--vector_backend` | Artifact backend to write: `faiss` or `numpy` |
| `--vector_db_output_dir` | Output directory for generated vector artifacts. Default: `artifacts/vector_db` |
| `--batch_size` | Batch size for ImageBind embedding. Default: `128` |
| `--cpu` | Force CPU inference |
| `--skip_missing_images` | Skip products whose referenced image file is missing |
| `--debug` | Enable debug logging |

For inspection/debugging, generate NumPy artifacts instead:

```bash
python generate_vector_db.py \
  --vector_backend numpy \
  --vector_db_output_dir artifacts/vector_db
```

Serving currently expects the FAISS backend.

## LLM Setup

### 1. Configure the OpenAI API key

Use of OpenAI's LLMs require one shared local API-key file at the repository root. From the repository root, create it with:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```text
OPENAI_API_KEY="your_openai_api_key"
```

### 2. Configure the LLM model

The application, retrieval, data-path, server, and evaluation settings live in
`shoptalk_config.ini`. For example:

```ini
[llm]
model_name = gpt-4o
temperature = 0.1

[app]
personality = -1

[retrieval]
vector_db_output_dir = artifacts/vector_db
vector_backend = faiss
top_k = 10

[data]
product_blurbs = EDA/product_blurbs/combined_blurb_dict.json
images_csv = images.csv

[server]
server_name = auto
server_port = 7860

[evals]
model_name = gpt-4o
temperature = 0.0
```

`server_name = auto` binds to `0.0.0.0` inside Docker and `127.0.0.1`
outside Docker. Set an explicit address to override auto-detection.


## Run the Gradio App

After the image files, `images.csv`, product blurbs, vector DB artifacts, and `.env` file are in place, run:

```bash
./run_ShopTalk_gradio.sh shoptalk_config.ini
```

Equivalent module command:

```bash
python -m server.gradio_app shoptalk_config.ini
```

Common runtime options:

```bash
python -m server.gradio_app --help
```

Note: The launcher detects when it is running inside Docker and binds Gradio to `0.0.0.0` automatically so Docker port mapping works. On a normal local host, it defaults to `127.0.0.1`. You can still override the bind address manually with `--server_name`, but that should not be necessary for the standard Docker workflow.

Examples:

```bash

# Enable debug output
./run_ShopTalk_gradio.sh shoptalk_config.ini --debug

# Force CPU usage
./run_ShopTalk_gradio.sh shoptalk_config.ini --cpu
```

The Gradio UI supports text-only queries, image-only queries, and combined text-plus-image queries.

## Run Tests

In order to verify technical correctness of code, there is a series of lightweight unit and behavior tests in the `tests` dir. They are intended to catch errors in helpers, configuration, parsing, diagnostics, vector-store loading, image-path handling, and UI helper logic without requiring a full ImageBind embedding run.

```bash
python -m pytest -q
```


## Run Evaluation Scripts

There are several decisions that go into any product recommendation. The chatbot must decide:
1. Whether to query the RAG database or depend solely on existing conversational context
2. Whether to make a recommendation based upon what has been found within the vector DB & conversational context, query the user for more information, or decide that the user's request cannot be fulfilled.
3. What product to recommend

The eval scripts require an OpenAI API key and use the eval model settings from `shoptalk_config.ini` unless overridden.

### Search-decision eval

```bash
./run_eval_search_decision.sh
```

This evaluates whether the LLM policy makes the right pre-retrieval decision: search now, ask for clarification, or continue without search.

Useful options:

```bash
./run_eval_search_decision.sh --help
./run_eval_search_decision.sh --category boundary
./run_eval_search_decision.sh --show-passes
./run_eval_search_decision.sh --model gpt-4o --temperature 0.0
```

Results are written to numbered text files under:

```text
server/evals/eval_results/
```

### Product-decision eval

```bash
./run_eval_product_response_decision.sh
```

This evaluates the post-retrieval LLM decision: recommend a retrieved product or ask for more information.

Useful options:

```bash
./run_eval_product_response_decision.sh --help
./run_eval_product_response_decision.sh --limit 10
./run_eval_product_response_decision.sh --model gpt-4o --temperature 0.0
```

Results are also written to numbered text files under:

```text
server/evals/search_decision/results/
server/evals/product_response_decision/results/
```

## Notes for Reviewers

This repository is best read as a portfolio ML systems project, not as a polished product.

The strongest parts are:

- end-to-end shape of the system,
- multimodal vector search pipeline,
- separation of recommender runtime components,
- Gradio demo with text and image input,
- diagnostics visibility,
- targeted LLM behavior evals.

The weakest or least finished parts are:

- the full artifact setup is still heavy,
- there is not yet a tiny reproducible demo dataset checked into the repo,
- retrieval quality evaluation is still thin,
- the current multimodal approach has not yet been compared against separate text/image representation spaces,
- `requirements.txt`, `Dockerfiles/requirements-docker.txt`, and environment-file documentation need to be kept synchronized as the app/eval stack changes,
- the README and portfolio presentation have lagged behind the code until this revision.

## Suggested Next Development Steps

1. Use this file as the new `README.md` after verifying local commands.
2. Clean up `requirements.txt` so app, eval, and test dependencies are explicitly declared.
3. Add a short demo section with screenshots or a GIF of the Gradio UI.
4. Add a small smoke-test artifact path or scripted mini-demo so reviewers can run something quickly.
5. Implement the representation comparison:
   - current shared ImageBind embedding,
   - separate text embedding search,
   - separate image embedding search,
   - score fusion or reranking between text/image channels.
6. Add retrieval metrics for a small hand-labeled query set.
7. Keep architecture cleanup focused. The project needs clarity more than production-scale abstraction.

## Legacy Notes

Older versions of this project used a Flask server entry point. The current app entry point is the Gradio app in:

```text
server/gradio_app.py
```

