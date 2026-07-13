# ShopTalk Image-Case Reviewer

This utility helps me build repeatable text-plus-image evaluation datasets for
ShopTalk.

It reads one evaluation case at a time, searches the web for candidate images,
shows those candidates in a Gradio gallery, downloads the image I select, and
updates a separate working JSONL file with the selected image path.

The original source JSONL is validated but never modified.

---

## Where the code belongs

The reviewer package should be located at:

```text
server/evals/retrieval_llm_response/image_case_reviewer/
```

The entry point should be:

```text
server/evals/retrieval_llm_response/run_image_case_reviewer.py
```

The launcher should be at the repository root:

```text
run_image_case_reviewer.sh
```

The expected structure is:

```text
server/evals/retrieval_llm_response/
├── run_image_case_reviewer.py
└── image_case_reviewer/
    ├── __init__.py
    ├── app.py
    ├── config.py
    ├── downloader.py
    ├── image_case_reviewer.ini
    ├── models.py
    ├── repository.py
    ├── searcher.py
    └── utils.py
```

---

## Required Python packages

The reviewer uses:

```text
ddgs
Pillow
requests
```

Install them in the active ShopTalk environment:

```bash
python -m pip install ddgs Pillow requests
```

They should also be listed in:

```text
requirements.txt
Dockerfiles/requirements-docker.txt
```

When Docker dependencies change, rebuild the Docker image before running the
reviewer inside the container.

---

## Input file

The default configuration expects the source cases at:

```text
server/evals/retrieval_llm_response/cases/
eval_cases_retrieval_llm_text_plus_image_45.jsonl
```

Create the directory if necessary:

```bash
mkdir -p server/evals/retrieval_llm_response/cases
```

The source file is treated as read-only.

On the first run, the reviewer validates the source file and creates a separate
working copy:

```text
server/evals/retrieval_llm_response/cases/
eval_cases_retrieval_llm_text_plus_image_45_with_images.jsonl
```

Subsequent saves update the working copy, not the source.

---

## Configuration

The default INI file is:

```text
server/evals/retrieval_llm_response/image_case_reviewer/
image_case_reviewer.ini
```

The important settings are:

```ini
[paths]
source_cases = server/evals/retrieval_llm_response/cases/eval_cases_retrieval_llm_text_plus_image_45.jsonl
working_cases = server/evals/retrieval_llm_response/cases/eval_cases_retrieval_llm_text_plus_image_45_with_images.jsonl
image_dir = server/evals/retrieval_llm_response/query_images
provenance_file = server/evals/retrieval_llm_response/query_images/image_sources.json

[search]
max_results = 12
region = us-en
safesearch = moderate

[download]
max_download_mb = 25
request_timeout_seconds = 30
jpeg_quality = 95

[server]
server_name = 127.0.0.1
server_port = 7861
share = false
```

### Search region

Useful region values include:

```text
us-en   United States, English
uk-en   United Kingdom, English
ca-en   Canada, English
au-en   Australia, English
wt-wt   Worldwide, with no specific regional preference
```

For broad product-image searches, `wt-wt` may provide a larger candidate pool.
For results more representative of a U.S. user, use `us-en`.

---

## Running on the host

From the ShopTalk repository root:

```bash
chmod +x run_image_case_reviewer.sh
./run_image_case_reviewer.sh
```

Then open:

```text
http://127.0.0.1:7861
```

The normal ShopTalk Gradio application uses port 7860, so the reviewer uses
7861 by default.

---

## Running in Docker

The Docker Compose configuration must expose reviewer port 7861:

```yaml
ports:
  - "127.0.0.1:7861:7861"
```

After changing Docker requirements or Compose configuration, rebuild or restart
the development environment as appropriate.

Open a shell in the ShopTalk container, then run:

```bash
cd /workspace
./run_image_case_reviewer.sh
```

Open the same address from the host browser:

```text
http://127.0.0.1:7861
```

The launcher automatically handles the only host-versus-Docker difference:

- On the host, Gradio binds to `127.0.0.1`.
- Inside Docker, Gradio binds to `0.0.0.0` so the Compose port mapping can reach it.

This affects network access only.

Because the repository is bind-mounted at `/workspace`, images and JSON files
written inside Docker appear directly in the host repository. No Docker-specific
file paths are needed.

---

## Using the reviewer

For each case:

1. Read the original user query.
2. Edit the web-search phrase when the original query is too conversational or
   vague for image search.
3. Click **Search**.
4. Compare the candidate images in the gallery.
5. Click a candidate to preview it.
6. Click the save button to download it and assign it to the current case.
7. Move to the next case.

Changing the web-search phrase does not modify the original evaluation query.

A more effective search phrase may be shorter and more product-oriented. For
example:

```text
Original query:
I need something lightweight that I can carry on a long day hike.

Possible image-search phrase:
lightweight hiking backpack product photo
```

---

## What is written when I save an image

The selected image is saved under:

```text
server/evals/retrieval_llm_response/query_images/
```

The filename is based on the case ID, for example:

```text
server/evals/retrieval_llm_response/query_images/text_image_001.jpg
```

The matching record in the working JSONL receives a repository-relative path:

```json
"image_path": "server/evals/retrieval_llm_response/query_images/text_image_001.jpg"
```

The strict evaluation-case JSONL is not expanded with search metadata.

Instead, provenance is stored separately in:

```text
server/evals/retrieval_llm_response/query_images/image_sources.json
```

That file records information such as:

- case ID,
- search phrase,
- selected image URL,
- source page,
- search-result title,
- saved local path.

This keeps the evaluation-case schema stable while preserving enough
information to understand and repeat the image-selection process.

---

## Replacing an image

Saving a different image for the same case replaces that case's current local
image and updates:

- the working JSONL `image_path`,
- the provenance entry for that case.

The original source JSONL remains unchanged.

---

## Resuming later

The reviewer preserves an existing working JSONL.

Therefore, I can stop the application and resume later without losing completed
selections.

On restart, the reviewer uses:

```text
eval_cases_retrieval_llm_text_plus_image_45_with_images.jsonl
```

when that working file already exists.

If I want to restart the review from the original source, I should first move or
delete the working JSONL and, if appropriate, the corresponding images and
provenance file.

---

## Running another experiment

To keep experiments separate, copy the default INI and change the working paths.

Example:

```text
server/evals/retrieval_llm_response/image_case_reviewer/
image_case_reviewer_worldwide.ini
```

Then run:

```bash
./run_image_case_reviewer.sh     --config server/evals/retrieval_llm_response/image_case_reviewer/image_case_reviewer_worldwide.ini
```

For repeatable experiments, each INI should use distinct values for:

```ini
working_cases =
image_dir =
provenance_file =
```

The command line intentionally accepts only `--config`. Experiment settings
remain together in one INI file rather than being split across many command-line
arguments.

---

## Expected limitations

- Web-search results may vary over time.
- Some candidate thumbnails or source images may fail because websites block
  hotlinking or remove files.
- `ddgs` is an external search library and may occasionally be rate-limited.
- A source-page URL does not establish image licensing or redistribution rights.
- The utility is intended to help construct evaluation data, not to guarantee
  legal reuse of arbitrary web images.

For a public portfolio repository, I should verify that any committed images are
appropriate to redistribute.

---

## Quick reference

Host:

```bash
./run_image_case_reviewer.sh
```

Docker:

```bash
cd /workspace
./run_image_case_reviewer.sh
```

Browser:

```text
http://127.0.0.1:7861
```

Source cases:

```text
server/evals/retrieval_llm_response/cases/
eval_cases_retrieval_llm_text_plus_image_45.jsonl
```

Working cases:

```text
server/evals/retrieval_llm_response/cases/
eval_cases_retrieval_llm_text_plus_image_45_with_images.jsonl
```

Images:

```text
server/evals/retrieval_llm_response/query_images/
```

Provenance:

```text
server/evals/retrieval_llm_response/query_images/image_sources.json
```
