# ShopTalk

ShopTalk is a portfolio project for conversational, multimodal product search. It combines product metadata, product images, ImageBind embeddings, FAISS vector search, and an LLM-guided conversation layer to recommend products from a local product catalog.

The core idea is simple: a shopping assistant should be able to handle requests that are more natural than keyword search. A user can ask for something like "a red men's shirt", upload an example image, or combine text and image input. ShopTalk embeds product text and product imagery, retrieves similar catalog items, and then uses an LLM policy layer to decide whether to ask for more detail or recommend one of the retrieved products.

This is not presented as a production shopping system. It is intended to show an end-to-end ML application architecture: preprocessing, multimodal representation, vector retrieval, conversational decision logic, a user interface, diagnostics, and targeted evaluation scripts.

## Current Project Status

### Implemented

- Product metadata preprocessing and product blurb generation under `EDA/`.
- Multimodal vector database generation in `generate_vector_db.py`.
- ImageBind-based product embeddings using a shared text/image representation space.
- FAISS artifact generation for serving-time vector search.
- Optional NumPy artifact generation for inspection and debugging.
- Gradio application with:
  - text query input,
  - optional image upload,
  - product recommendation display,
  - top retrieved product gallery,
  - diagnostics panel.
- Refactored recommender core under `server/recommender_core/`.
- Configurable LLM model and temperature via `shoptalk_config.ini`.
- Runtime launcher scripts for the Gradio app and eval modules.
- Unit tests for preprocessing, vector DB helpers, recommender helpers, Gradio helper behavior, image path handling, diagnostics, and vector-store behavior.
- Two targeted LLM evaluation modules:
  - search-decision evaluation: should the assistant search now, ask a clarifying question, or continue without search?
  - product-decision evaluation: given retrieved products, should the assistant recommend a product or ask for more information?
- Numbered human-readable eval result files, so repeated eval runs do not overwrite previous results.

### Still Planned / In Progress

- Replace the old project README with this updated README after review.
- Improve the portfolio presentation so the repository clearly tells the story of the project without requiring conversational context.
- Compare the current shared ImageBind representation approach against separate text and image representation spaces.
- Add more direct examples and screenshots of the Gradio app in use.
- Add a smaller smoke-test dataset or documented fixture path so reviewers can run a minimal demo without reconstructing the full product artifact set.
- Strengthen end-to-end testing around the full Gradio/recommender path. Current tests deliberately avoid heavy ImageBind inference.
- Continue modest architectural cleanup where it clarifies the demo, but avoid over-engineering this as if it were a production service.
- Improve retrieval/recommendation metrics. The current evals mostly target LLM control decisions, not retrieval quality.

## Architecture Overview

At a high level, ShopTalk has two phases.

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
EDA/
  DescriptionGenerator.py                  Generate product descriptions / blurbs
  preprocessor.py                          Product metadata preprocessing helpers
  config.ini                               EDA/preprocessing configuration

generate_vector_db.py                      Build ImageBind/FAISS vector artifacts

server/
  gradio_app.py                            Main Gradio application
  gradio_images.py                         Image display helpers for Gradio
  shoptalk_paths.py                        Project-relative path constants
  templates/template.html                  Legacy Flask template kept in repo

server/recommender_core/
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
  reply_types.py                           Reply/result dataclasses
  shop_talk_recommender.py                 Main recommender orchestration class
  vector_db.py                             FAISS artifact loading
  vector_query.py                          Query-vector combination/search helpers

server/evals/
  eval_search_decision.py                  Eval for pre-retrieval search decisions
  eval_llm_decision.py                     Eval for post-retrieval product decisions
  eval_cases_search_decision.jsonl         Search-decision test cases
  eval_cases_llm_decision.jsonl            Product-decision test cases

tests/                                     Unit tests and lightweight behavior tests

run_ShopTalk_gradio.sh                     Root-level launcher for Gradio app
run_eval_search_decision.sh                Root-level launcher for search-decision eval
run_eval_llm_decision.sh                   Root-level launcher for product-decision eval

shoptalk_config.ini                        App/eval LLM model settings
requirements.txt                           Python dependencies
setup.py                                   Package install metadata
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
| `.env` | Local OpenAI API key | Created locally |

These are local artifacts, not source files.

## Installation

The project has a few heavy dependencies. ImageBind and FAISS are the two most likely sources of environment friction.

### 1. Create and activate a conda environment

From the parent directory that will contain both `ImageBind` and this repository:

```bash
conda create --name imagebind python=3.10 -y
conda activate imagebind
```

### 2. Install ImageBind

```bash
git clone https://github.com/facebookresearch/ImageBind
cd ImageBind
pip install .
```

### 3. Install ShopTalk

Move to the ShopTalk repository root:

```bash
cd ../ShopTalk
pip install -e .
```

`pip install -e .` uses `setup.py` and `requirements.txt`.

Current warning: `requirements.txt` is oriented toward the current development environment and includes `faiss-gpu-cu12`. If your machine does not support that FAISS package, install the appropriate FAISS package for your environment instead. For CPU-only testing, `faiss-cpu` is usually the simpler choice.

The current code also relies on packages that are not clearly captured by the existing `requirements.txt` in this repo snapshot, especially for the Gradio UI and tests. If they are not already present in your environment, install them explicitly:

```bash
pip install gradio pytest langchain-classic
```

That dependency mismatch should eventually be fixed in `requirements.txt`, but the command above documents what is needed to run the current app and test suite.

### 4. Configure the OpenAI API key

Create a `.env` file in the project root:

```text
OPENAI_API_KEY="your_openai_api_key"
```

### 5. Configure the LLM model

The default model settings live in `shoptalk_config.ini`:

```ini
[llm]
model_name = gpt-4o
temperature = 0.1

[evals]
model_name = gpt-4o
temperature = 0.0
```

You can override the app model at runtime with `--model` and `--temperature`.

## Data Setup

ShopTalk expects Amazon Berkeley Objects-style images and metadata.

### 1. Download images and metadata

Download `abo-images-small.tar` from the Amazon Berkeley Objects dataset.

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

## Run the Gradio App

After the image files, `images.csv`, product blurbs, vector DB artifacts, and `.env` file are in place, run:

```bash
./run_ShopTalk_gradio.sh
```

Equivalent module command:

```bash
python -m server.gradio_app
```

Common runtime options:

```bash
python -m server.gradio_app --help
```

Examples:

```bash
# Force CPU
./run_ShopTalk_gradio.sh --cpu

# Enable debug output
./run_ShopTalk_gradio.sh --debug

# Use a specific LLM model
./run_ShopTalk_gradio.sh --model gpt-4o --temperature 0.1

# Choose a specific assistant personality by index
./run_ShopTalk_gradio.sh --personality 0

# Create a public Gradio share link
./run_ShopTalk_gradio.sh --share
```

The Gradio UI supports text-only queries, image-only queries, and combined text-plus-image queries.

## Run Tests

From the project root:

```bash
python -m pytest -q
```

The tests are mostly lightweight unit and behavior tests. They are intended to catch regressions in helpers, configuration, parsing, diagnostics, vector-store loading, image-path handling, and UI helper logic without requiring a full ImageBind embedding run.

## Run Evaluation Scripts

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
./run_eval_llm_decision.sh
```

This evaluates the post-retrieval LLM decision: recommend a retrieved product or ask for more information.

Useful options:

```bash
./run_eval_llm_decision.sh --help
./run_eval_llm_decision.sh --limit 10
./run_eval_llm_decision.sh --model gpt-4o --temperature 0.0
```

Results are also written to numbered text files under:

```text
server/evals/eval_results/
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
- dependency declarations need cleanup because the current source uses Gradio and `langchain_classic` while `requirements.txt` does not clearly document every runtime/test dependency,
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

`server/templates/template.html` is still present, but the active demo path is Gradio.
