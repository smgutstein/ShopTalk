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
import configparser
import json
import time
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


# These defaults keep the generator self-contained: case definitions, generated
# judgment files, and eval configuration all live beside the eval modules unless
# the INI file says otherwise.
DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases_retrieval_llm.jsonl")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("eval_results")
DEFAULT_OUTPUT_PREFIX = "retrieval_llm_judgments"
DEFAULT_EVAL_CONFIG_PATH = Path(__file__).with_name("retrieval_llm_eval.ini")


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
    """Parse the single eval-config argument.

    Earlier versions of this eval runner exposed every path and runtime setting
    as a command-line flag. That worked, but the command became noisy and easy to
    mistype. The normal interface is now intentionally narrow: point the script
    at one INI file, and keep the eval setup there.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run preset ShopTalk cases and write a hand-editable retrieval/LLM "
            "judgment JSON file."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help=f"Path to retrieval/LLM eval INI file. Default: {DEFAULT_EVAL_CONFIG_PATH}",
    )
    return parser.parse_args()


def _config_value(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: str,
) -> str:
    """Return a stripped INI value and reject undocumented blank values.

    The eval INI is meant to double as documentation. For that reason optional
    settings should use explicit sentinels such as ``none``, ``auto``, or
    ``config`` instead of blank strings. A blank value usually means someone
    copied an older config or accidentally deleted the documented default.
    """
    value = parser.get(section, option, fallback=fallback).strip()
    if value == "":
        raise ValueError(
            f"Blank value for [{section}] {option}. "
            "Use an explicit documented value such as 'none', 'auto', or 'config'."
        )
    return value


def _optional_limit(parser: configparser.ConfigParser) -> int | None:
    """Return [eval] limit, where ``none`` means run every case."""
    value = _config_value(parser, "eval", "limit", fallback="none")
    if value.lower() == "none":
        return None
    return int(value)


def _auto_path(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> Path | None:
    """Return a configured path, where ``auto`` means choose a numbered file."""
    value = _config_value(parser, section, option, fallback="auto")
    if value.lower() == "auto":
        return None
    return Path(value)


def _config_default_str(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> str | None:
    """Return a string override, where ``config`` means use shoptalk_config.ini."""
    value = _config_value(parser, section, option, fallback="config")
    if value.lower() == "config":
        return None
    return value


def _config_default_float(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> float | None:
    """Return a float override, where ``config`` means use shoptalk_config.ini."""
    value = _config_value(parser, section, option, fallback="config")
    if value.lower() == "config":
        return None
    return float(value)


def _non_negative_float(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: str,
) -> float:
    """Return a float setting that must be zero or greater.

    The generator uses these values for pacing API-bound eval runs. Negative
    sleep durations or retry delays are almost certainly config mistakes, so we
    reject them early with a clear message instead of failing inside ``sleep``.
    """
    value = float(_config_value(parser, section, option, fallback=fallback))
    if value < 0:
        raise ValueError(f"[{section}] {option} must be >= 0, got {value}")
    return value


def _non_negative_int(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: str,
) -> int:
    """Return an integer setting that must be zero or greater."""
    value = int(_config_value(parser, section, option, fallback=fallback))
    if value < 0:
        raise ValueError(f"[{section}] {option} must be >= 0, got {value}")
    return value


def load_eval_args(config_path: Path) -> argparse.Namespace:
    """Load the old runtime arguments from a small INI file.

    The rest of this module still expects an argparse-like object with fields
    such as ``cases``, ``top_k``, and ``vector_db_output_dir``. To keep this patch
    small, the INI loader creates that same shape instead of rewriting the rest
    of the evaluation code.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Return an argparse-like namespace so the rest of the module can use the
    # same attribute names it used when these settings were CLI flags.
    return argparse.Namespace(
        cases=Path(parser.get("eval", "cases", fallback=str(DEFAULT_CASES_PATH))),
        limit=_optional_limit(parser),
        output=_auto_path(parser, "eval", "output"),
        output_dir=Path(parser.get("eval", "output_dir", fallback=str(DEFAULT_RESULTS_DIR))),
        output_prefix=parser.get(
            "eval",
            "output_prefix",
            fallback=DEFAULT_OUTPUT_PREFIX,
        ),
        progress=parser.getboolean("eval", "progress", fallback=True),
        sleep_seconds_between_cases=_non_negative_float(
            parser,
            "eval",
            "sleep_seconds_between_cases",
            fallback="0.0",
        ),
        max_retries=_non_negative_int(
            parser,
            "eval",
            "max_retries",
            fallback="0",
        ),
        retry_sleep_seconds=_non_negative_float(
            parser,
            "eval",
            "retry_sleep_seconds",
            fallback="5.0",
        ),
        personality=parser.getint("runtime", "personality", fallback=0),
        debug=parser.getboolean("runtime", "debug", fallback=False),
        cpu=parser.getboolean("runtime", "cpu", fallback=False),
        config=Path(parser.get("runtime", "shoptalk_config", fallback=str(DEFAULT_CONFIG_PATH))),
        model=_config_default_str(parser, "runtime", "model"),
        temperature=_config_default_float(parser, "runtime", "temperature"),
        vector_db_output_dir=parser.get(
            "artifacts",
            "vector_db_output_dir",
            fallback=str(VECTOR_DB_OUTPUT_DIR),
        ),
        vector_backend=parser.get(
            "artifacts",
            "vector_backend",
            fallback=DEFAULT_VECTOR_BACKEND,
        ),
        top_k=parser.getint("artifacts", "top_k", fallback=10),
        product_blurbs=parser.get(
            "artifacts",
            "product_blurbs",
            fallback=str(COMBINED_BLURBS_PATH),
        ),
        images_csv=parser.get("artifacts", "images_csv", fallback=str(IMAGES_CSV)),
    )


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

            # Parse each line independently so one malformed case reports its
            # exact line number instead of producing a vague batch-load failure.
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

    # The full ShopTalk turn is intentionally used here. Unlike the targeted
    # policy evals, this generator captures retrieval diagnostics and the final
    # answer that a human will later judge.
    recommender.reset_conversation()
    payload = recommender.generate_reply(
        user_input=case.query,
        image_path=case.image_path,
    )

    # Diagnostics are converted to plain dictionaries before writing JSON so the
    # hand-editable output has no dependency on internal Python dataclasses.
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


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    """Return True when an exception appears to be an API rate-limit failure.

    This avoids taking a direct dependency on OpenAI exception classes inside the
    eval runner. ShopTalk may change LLM providers later, but rate-limit errors
    usually still include either ``429`` or ``rate limit`` in their message.
    Non-rate-limit exceptions are re-raised immediately so real bugs do not get
    hidden behind retry behavior.
    """
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "rate_limit" in message


def run_case_with_retries(
    recommender: ShopTalkRecommender,
    case: RetrievalLlmEvalCase,
    *,
    max_retries: int,
    retry_sleep_seconds: float,
    show_progress: bool,
) -> dict[str, Any]:
    """Run one case, retrying short-lived rate-limit failures.

    The eval generator runs many LLM-backed ShopTalk turns in sequence. Without
    a retry loop, a temporary 429 error can kill the entire batch and waste the
    successfully completed cases. We only retry likely rate-limit errors; other
    exceptions still fail fast.
    """
    for attempt in range(max_retries + 1):
        # ``attempt`` is zero-based; max_retries therefore means "extra tries
        # after the first failure," not total attempts.
        try:
            return run_case(recommender, case)
        except Exception as exc:
            if not _looks_like_rate_limit_error(exc) or attempt >= max_retries:
                raise

            retry_number = attempt + 1
            if show_progress:
                print(
                    f"Rate limit while running {case.case_id}; "
                    f"retry {retry_number}/{max_retries} after "
                    f"{retry_sleep_seconds:.1f}s.",
                    flush=True,
                )
            time.sleep(retry_sleep_seconds)

    # The loop always returns or raises, but this keeps type checkers happy.
    raise RuntimeError(f"Unexpected retry-loop exit for {case.case_id}")


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
        # Keep run metadata with the judgments so a reviewed file remains useful
        # even after model defaults or eval configuration have changed.
        "metadata": {
            "schema_version": "retrieval_llm_judgments_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "cases_path": str(cases_path),
            "top_k": args.top_k,
            "model_name": model_name,
            "temperature": temperature,
            "sleep_seconds_between_cases": args.sleep_seconds_between_cases,
            "max_retries": args.max_retries,
            "retry_sleep_seconds": args.retry_sleep_seconds,
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
    cli_args = parse_args()
    args = load_eval_args(cli_args.config)
    cases = load_jsonl_cases(args.cases, limit=args.limit)
    output_path = args.output or next_numbered_output_path(
        args.output_dir,
        prefix=args.output_prefix,
        suffix=".json",
    )

    if args.progress:
        print(f"Loaded {len(cases)} retrieval/LLM eval case(s) from {args.cases}", flush=True)
        print(f"Judgment output will be written to {output_path}", flush=True)
        print(
            "Run pacing: "
            f"sleep_seconds_between_cases={args.sleep_seconds_between_cases}, "
            f"max_retries={args.max_retries}, "
            f"retry_sleep_seconds={args.retry_sleep_seconds}",
            flush=True,
        )
        print("Building ShopTalk recommender...", flush=True)

    recommender = build_recommender_from_args(args)

    if args.progress:
        print("Starting eval cases...", flush=True)

    judgment_cases: list[dict[str, Any]] = []
    total_cases = len(cases)
    # Cases are processed sequentially to preserve predictable console output and
    # to avoid hammering the LLM API during a small hand-reviewed eval run.
    for index, case in enumerate(cases, start=1):
        remaining_after_this = total_cases - index
        if args.progress:
            print(
                f"[{index}/{total_cases}] Running {case.case_id} "
                f"({case.query_type}, {case.category}); ",
                flush=True,
            )

        judgment_record = run_case_with_retries(
            recommender,
            case,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            show_progress=args.progress,
        )
        judgment_cases.append(judgment_record)

        if args.progress:
            retrieved_count = len(judgment_record.get("retrieved_products") or [])
            print(
                f"[{index}/{total_cases}] Finished {case.case_id}; "
                f"captured {retrieved_count} retrieved product(s).",
                flush=True,
            )

        if index < total_cases and args.sleep_seconds_between_cases > 0:
            if args.progress:
                print(
                    f"Sleeping {args.sleep_seconds_between_cases:.1f}s before next case...",
                    flush=True,
                )
            time.sleep(args.sleep_seconds_between_cases)

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
