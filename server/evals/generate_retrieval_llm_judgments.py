"""Generate hand-editable ShopTalk retrieval/LLM judgment files.

This module is the first half of a deliberately human-in-the-loop evaluation
workflow:

    fixed eval cases -> real ShopTalk run -> editable JSON judgment file

It does *not* try to decide whether ShopTalk did well. Instead, it captures the
inputs, retrieval diagnostics, selected product, and final LLM response, then
adds blank fields for a human reviewer to fill in. The companion scorer module
(`score_retrieval_llm_judgments.py`) reads the human-augmented JSON later and
computes metrics.

That split is intentional. Retrieval relevance and response quality are fuzzy
for shopping recommendations. For this portfolio project, the honest approach is
not to hide that subjectivity; it is to make the judgment step explicit,
repeatable, and easy to inspect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # Supports: python -m server.evals.generate_retrieval_llm_judgments
    from ..recommender_core.config import RecommenderConfig, load_shoptalk_config
    from ..recommender_core.diagnostics import diagnostics_to_dict
    from ..recommender_core.recommender_factory import build_recommender
    from ..recommender_core.shop_talk_recommender import ShopTalkRecommender
    from ..shoptalk_paths import (
        COMBINED_BLURBS_PATH,
        DEFAULT_CONFIG_PATH,
        DEFAULT_VECTOR_BACKEND,
        IMAGES_CSV,
        VECTOR_DB_OUTPUT_DIR,
    )
except ImportError:  # Supports running from inside server/: python -m evals...
    from recommender_core.config import RecommenderConfig, load_shoptalk_config
    from recommender_core.diagnostics import diagnostics_to_dict
    from recommender_core.recommender_factory import build_recommender
    from recommender_core.shop_talk_recommender import ShopTalkRecommender
    from shoptalk_paths import (
        COMBINED_BLURBS_PATH,
        DEFAULT_CONFIG_PATH,
        DEFAULT_VECTOR_BACKEND,
        IMAGES_CSV,
        VECTOR_DB_OUTPUT_DIR,
    )


DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases_retrieval_llm.jsonl")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("eval_results")
DEFAULT_OUTPUT_PREFIX = "retrieval_llm_judgments"


@dataclass(frozen=True)
class RetrievalLlmEvalCase:
    """One preset ShopTalk evaluation case.

    The case file is JSONL because preset cases are naturally line-oriented and
    easy to append to over time. The generated judgment output is normal pretty
    JSON because it contains nested retrieved products and human judgment fields.

    Fields:
        case_id: Stable unique ID used in reports and failure summaries.
        query_type: One of text_only, image_only, or text_plus_image.
        category: Coarse grouping such as office_furniture or missing_product.
        query: User-facing text. Null for image-only cases.
        image_path: Optional local query image. Required for image-only and
            text+image cases, null for text-only cases.
        target_product_id: Expected catalog product for positive cases. Null for
            negative/missing-product cases.
        target_title: Human-readable expected product title, copied into output
            so you do not need to cross-reference the product dictionary while
            judging.
        expected_available: True when the catalog is expected to contain a good
            match; False when the case is deliberately asking for something the
            catalog does not have.
        requires_image: Marker for future image-only/text+image case generation.
            Text-only cases can still set this to True to indicate their target
            product should have a usable image artifact.
        notes: Free-form explanation of what the case is meant to test.
    """

    case_id: str
    query_type: str
    category: str
    query: str | None
    image_path: str | None
    target_product_id: str | None
    target_title: str | None
    expected_available: bool
    requires_image: bool = False
    notes: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the judgment generator.

    The artifact/model flags intentionally mirror the Gradio app and existing
    eval scripts. That makes this module easier to run against the same vector DB,
    product blurb JSON, image map, model name, and temperature already used by the
    rest of the project.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run preset ShopTalk cases and write a hand-editable retrieval/LLM "
            "judgment JSON file."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Path to JSONL eval cases. Default: {DEFAULT_CASES_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to run while iterating.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit JSON output path. If omitted, a numbered path is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for numbered judgment files. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Filename prefix for numbered judgment files. Default: {DEFAULT_OUTPUT_PREFIX}",
    )

    parser.add_argument("-p", "--personality", type=int, default=0)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ShopTalk config file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Override eval model name from the config file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override eval temperature from the config file.",
    )
    parser.add_argument(
        "--vector_db_output_dir",
        type=str,
        default=str(VECTOR_DB_OUTPUT_DIR),
        help="Base directory containing generated vector DB artifacts.",
    )
    parser.add_argument(
        "--vector_backend",
        default=DEFAULT_VECTOR_BACKEND,
        choices=["faiss"],
        help="Vector backend to load.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of products to retrieve per case.",
    )
    parser.add_argument(
        "--product_blurbs",
        type=str,
        default=str(COMBINED_BLURBS_PATH),
        help="Path to the product blurbs JSON file.",
    )
    parser.add_argument(
        "--images_csv",
        type=str,
        default=str(IMAGES_CSV),
        help="Path to the image ID mapping CSV file.",
    )
    return parser.parse_args()


def load_jsonl_cases(path: Path, limit: int | None = None) -> list[RetrievalLlmEvalCase]:
    """Load preset evaluation cases from a JSONL file.

    Empty lines are ignored so the file stays easy to hand-edit. Malformed JSON
    and schema mismatches raise descriptive errors with the line number because a
    bad eval case should fail before the expensive recommender/model setup runs.
    """
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}")

    cases: list[RetrievalLlmEvalCase] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            try:
                cases.append(RetrievalLlmEvalCase(**payload))
            except TypeError as exc:
                raise ValueError(
                    f"Invalid case schema on line {line_number} of {path}: {exc}"
                ) from exc

            if limit is not None and len(cases) >= limit:
                break

    return cases


def require_case_inputs(case: RetrievalLlmEvalCase) -> None:
    """Validate modality-specific case inputs.

    This prevents accidentally evaluating something different from what the case
    ID claims. For example, an image-only case with leftover text is not really an
    image-only test; it is a text+image test with confusing metadata.
    """
    if case.query_type == "text_only":
        if not case.query:
            raise ValueError(f"{case.case_id}: text_only case requires query")
        if case.image_path is not None:
            raise ValueError(f"{case.case_id}: text_only case should not include image_path")
        return

    if case.query_type == "image_only":
        if case.query:
            raise ValueError(f"{case.case_id}: image_only case should not include query")
        if not case.image_path:
            raise ValueError(f"{case.case_id}: image_only case requires image_path")
        return

    if case.query_type == "text_plus_image":
        if not case.query:
            raise ValueError(f"{case.case_id}: text_plus_image case requires query")
        if not case.image_path:
            raise ValueError(f"{case.case_id}: text_plus_image case requires image_path")
        return

    raise ValueError(f"{case.case_id}: unsupported query_type={case.query_type!r}")


def validate_image_path_if_present(case: RetrievalLlmEvalCase) -> None:
    """Fail early when an image-based case points to a missing file."""
    if case.image_path is None:
        return

    path = Path(case.image_path)
    if not path.exists():
        raise FileNotFoundError(f"{case.case_id}: image_path does not exist: {path}")


def next_numbered_output_path(output_dir: Path, *, prefix: str, suffix: str) -> Path:
    """Return the next available non-overwriting numbered output path.

    Example:
        retrieval_llm_judgments_001.json
        retrieval_llm_judgments_002.json

    Numbered outputs are useful here because judgment files are hand-edited. A
    rerun should not silently destroy the review work from an earlier run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = output_dir / f"{prefix}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def build_recommender_from_args(args: argparse.Namespace) -> ShopTalkRecommender:
    """Build the same recommender runtime used by the Gradio app.

    The generator evaluates the real end-to-end ShopTalk turn flow rather than a
    shortcut around retrieval or LLM response generation. We still use eval model
    defaults from the config file so evaluation runs can be deterministic-ish
    compared with normal app usage.
    """
    file_config = load_shoptalk_config(args.config)
    model_name = args.model or file_config.eval_model_name
    temperature = (
        args.temperature
        if args.temperature is not None
        else file_config.eval_temperature
    )

    config = RecommenderConfig(
        personality_index=args.personality,
        debug=args.debug,
        force_cpu=args.cpu,
        model_name=model_name,
        temperature=temperature,
        vector_db_output_dir=Path(args.vector_db_output_dir),
        vector_backend=args.vector_backend,
        top_k=args.top_k,
        blurbs_path=Path(args.product_blurbs),
        images_csv_path=Path(args.images_csv),
    )
    return build_recommender(config)


def serialize_retrieved_products(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert retrieval diagnostics into editable candidate judgments.

    Diagnostics already provide a compact top_products list. This function adds a
    rank and blank human relevance fields. The rank is important because the
    scorer can later compute Hit@1/Hit@5 style metrics from the same file you
    reviewed by hand.
    """
    products = diagnostics.get("top_products") or []
    serialized: list[dict[str, Any]] = []

    for rank, product in enumerate(products, start=1):
        serialized.append(
            {
                "rank": rank,
                "product_id": product.get("product_id"),
                "title": product.get("item_name"),
                "score": product.get("score"),
                "product_type": product.get("product_type"),
                "image_paths": product.get("image_paths", []),
                # Human-edited field. Use:
                #   2 = strong match
                #   1 = acceptable / partial match
                #   0 = irrelevant
                #   null = not judged yet
                "retrieval_relevance": None,
                # Optional explanation for ambiguous calls. Keep this short; long
                # judgment notes make the file painful to scan.
                "retrieval_notes": "",
            }
        )

    return serialized


def build_human_eval_stub() -> dict[str, Any]:
    """Return blank per-case judgment fields for manual review.

    Product-level relevance belongs inside each retrieved product. These fields
    judge the whole case and the LLM response built on top of retrieval.
    """
    return {
        "target_product_retrieved": None,
        "retrieval_quality": None,
        "llm_chose_good_product": None,
        "llm_response_quality": None,
        "llm_response_grounded": None,
        "should_have_refused_or_said_no_match": None,
        "human_notes": "",
    }


def chosen_product_title(payload: dict[str, Any]) -> str | None:
    """Return the chosen product title from a ShopTalk reply payload, if any."""
    chosen_product = payload.get("chosen_product")
    if chosen_product is None:
        return None
    return getattr(chosen_product, "item_name", None)


def run_case(recommender: ShopTalkRecommender, case: RetrievalLlmEvalCase) -> dict[str, Any]:
    """Run one case through ShopTalk and build one judgment record.

    A conversation reset before every case is non-negotiable. Without it, earlier
    eval cases would become hidden context for later ones, and the results would
    depend on case ordering rather than only on the current user query/image.
    """
    require_case_inputs(case)
    validate_image_path_if_present(case)

    recommender.reset_conversation()
    payload = recommender.generate_reply(
        user_input=case.query,
        image_path=case.image_path,
    )

    diagnostics = diagnostics_to_dict(payload.get("diagnostics"))
    final_response = payload.get("final_response", "")

    return {
        "case_id": case.case_id,
        "query_type": case.query_type,
        "category": case.category,
        "query": case.query,
        "image_path": case.image_path,
        "target_product_id": case.target_product_id,
        "target_title": case.target_title,
        "expected_available": case.expected_available,
        "requires_image": case.requires_image,
        "case_notes": case.notes,
        "system_output": {
            "search_performed": diagnostics.get("search_performed"),
            "llm_search_query": diagnostics.get("llm_search_query"),
            "embedding_mode": diagnostics.get("embedding_mode"),
            "decision": diagnostics.get("decision"),
            "chosen_product_id": diagnostics.get("chosen_pid"),
            "chosen_product_title": chosen_product_title(payload),
            "initial_llm_response": diagnostics.get("initial_llm_response"),
            "final_response": final_response,
            "timings": diagnostics.get("timings", {}),
        },
        "retrieved_products": serialize_retrieved_products(diagnostics),
        "human_eval": build_human_eval_stub(),
    }


def build_output_payload(
    *,
    args: argparse.Namespace,
    cases_path: Path,
    judgment_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the top-level JSON document written for human review."""
    file_config = load_shoptalk_config(args.config)
    model_name = args.model or file_config.eval_model_name
    temperature = (
        args.temperature
        if args.temperature is not None
        else file_config.eval_temperature
    )

    return {
        "metadata": {
            "schema_version": "retrieval_llm_judgments_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "cases_path": str(cases_path),
            "top_k": args.top_k,
            "model_name": model_name,
            "temperature": temperature,
            "human_judgment_scale": {
                "2": "strong match / good response",
                "1": "acceptable / partial match",
                "0": "irrelevant / bad response",
                "null": "not yet judged",
            },
            "human_eval_fields": {
                "target_product_retrieved": "true/false/null; whether target_product_id appears in retrieved_products",
                "retrieval_quality": "2/1/0/null; overall quality of retrieved set",
                "llm_chose_good_product": "true/false/null; whether final product choice was reasonable",
                "llm_response_quality": "2/1/0/null; overall final answer quality",
                "llm_response_grounded": "true/false/null; whether response sticks to retrieved products/catalog evidence",
                "should_have_refused_or_said_no_match": "true/false/null; especially useful for expected_available=false cases",
            },
        },
        "cases": judgment_cases,
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write pretty JSON so the output can be edited by hand."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, ensure_ascii=False)
        outfile.write("\n")


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    cases = load_jsonl_cases(args.cases, limit=args.limit)
    recommender = build_recommender_from_args(args)

    judgment_cases: list[dict[str, Any]] = []
    for case in cases:
        judgment_cases.append(run_case(recommender, case))

    output_path = args.output or next_numbered_output_path(
        args.output_dir,
        prefix=args.output_prefix,
        suffix=".json",
    )
    payload = build_output_payload(
        args=args,
        cases_path=args.cases,
        judgment_cases=judgment_cases,
    )
    write_json(payload, output_path)

    print(f"Wrote retrieval/LLM judgment file to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
