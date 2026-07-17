"""Unified ShopTalk retrieval/LLM evaluation workflow.

This module supports both halves of the human-in-the-loop retrieval/LLM eval:

    generate -> fixed eval cases -> real ShopTalk run -> editable judgment JSON
    score    -> reviewed judgment JSON -> retrieval + LLM response metrics

The two steps share one INI file but are selected with explicit subcommands so
the normal commands stay short and self-documenting.
"""

from __future__ import annotations

import argparse
import configparser
import json
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:  # Supports: python -m server.evals.retrieval_llm_response.retrieval_llm_eval
    from ...recommender_core.config import RecommenderConfig, load_shoptalk_config
    from ...recommender_core.diagnostics import diagnostics_to_dict
    from ...recommender_core.recommender_factory import build_recommender
    from ...recommender_core.shop_talk_recommender import ShopTalkRecommender
    from ...shoptalk_paths import (
        COMBINED_BLURBS_PATH,
        DEFAULT_CONFIG_PATH,
        DEFAULT_VECTOR_BACKEND,
        IMAGES_CSV,
        VECTOR_DB_OUTPUT_DIR,
    )
except ImportError:  # Supports running from inside server/: python -m evals.retrieval_llm_response...
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

# Defaults keep cases, generated judgment files, score reports, and eval
# configuration beside the eval modules unless the INI file says otherwise.
DEFAULT_CASES_PATH = Path(__file__).with_name("cases") / "eval_cases_retrieval_llm.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_GENERATED_DIR = Path(__file__).with_name("generated")
DEFAULT_REVIEWED_DIR = Path(__file__).with_name("reviewed")
DEFAULT_SCORED_DIR = Path(__file__).with_name("scored")
DEFAULT_OUTPUT_PREFIX = "retrieval_llm_judgments"
DEFAULT_SCORE_OUTPUT_PREFIX = "retrieval_llm_metrics"
DEFAULT_EVAL_CONFIG_PATH = Path(__file__).with_name("retrieval_llm_eval.ini")
EXPECTED_SCHEMA_VERSION = "retrieval_llm_judgments_v4"

REQUIRED_CASE_NOTE_PREFIXES = (
    "valid-match case.",
    "difficult/ambiguous match case.",
    "no appropriate product case.",
)


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
        notes: Explanation of what the case is meant to test. It must begin
            with one of the exact prefixes in ``REQUIRED_CASE_NOTE_PREFIXES``.
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
        if  Path(__file__).with_name(config_path.name).exists():
            config_path = Path(__file__).with_name(config_path.name)
        else:
            raise FileNotFoundError(f"Eval config file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Return an argparse-like namespace so the rest of the module can use the
    # same attribute names it used when these settings were CLI flags.
    return argparse.Namespace(
        cases=Path(parser.get("eval", "cases", fallback=str(DEFAULT_CASES_PATH))),
        limit=_optional_limit(parser),
        output=_auto_path(parser, "eval", "output"),
        output_dir=Path(parser.get("eval", "output_dir", fallback=str(DEFAULT_GENERATED_DIR))),
        reviewed_output_dir=Path(
            parser.get("eval", "reviewed_output_dir", fallback=str(DEFAULT_REVIEWED_DIR))
        ),
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
                case = RetrievalLlmEvalCase(**payload)
            except TypeError as exc:
                raise ValueError(
                    f"Invalid case schema on line {line_number} of {path}: {exc}"
                ) from exc

            validate_case_notes(case)
            cases.append(case)

            if limit is not None and len(cases) >= limit:
                break

    return cases


def validate_case_notes(case: RetrievalLlmEvalCase) -> None:
    """Require an exact case-classification prefix in every case note.

    The classification stays in the existing free-form ``notes`` field rather
    than adding another schema field. Requiring one of three exact prefixes
    keeps the generated ``case_notes`` readable and consistently searchable.
    """
    if case.notes.startswith(REQUIRED_CASE_NOTE_PREFIXES):
        return

    expected = ", ".join(repr(prefix) for prefix in REQUIRED_CASE_NOTE_PREFIXES)
    raise ValueError(
        f"{case.case_id}: notes must begin with one of these exact prefixes: "
        f"{expected}"
    )


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


def next_linked_judgment_paths(
    generated_dir: Path,
    reviewed_dir: Path,
    *,
    prefix: str,
) -> tuple[Path, Path]:
    """Return matching unused generated/reviewed paths for one eval run."""
    generated_dir.mkdir(parents=True, exist_ok=True)
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        run_name = f"{prefix}_{index:03d}"
        generated = generated_dir / f"{run_name}.json"
        reviewed = reviewed_dir / f"{run_name}_reviewed.json"
        if not generated.exists() and not reviewed.exists():
            return generated, reviewed
        index += 1


def reviewed_path_for_generated(generated_path: Path, reviewed_dir: Path) -> Path:
    """Derive the editable reviewed-copy path from a generated filename."""
    return reviewed_dir / f"{generated_path.stem}_reviewed{generated_path.suffix}"


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


def load_product_blurb_source(path: Path) -> dict[str, Any]:
    """Load the product-blurb artifact used by the ShopTalk run.

    The generated judgment file now carries inline evidence for each retrieved
    product. That evidence should come from the same source configured for the
    recommender, not from a richer external catalog. Otherwise the grounding
    review would answer the wrong question: whether the response is true in some
    broader sense rather than whether it was supported by what ShopTalk gave the
    LLM.
    """
    if not path.exists():
        raise FileNotFoundError(f"Product blurb file not found: {path}")

    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)

    # The current combined-blurb artifact is expected to be a product_id-keyed
    # JSON object. Failing early is preferable to silently emitting null evidence
    # for every product and making the human reviewer guess what went wrong.
    if not isinstance(payload, dict):
        raise ValueError(f"Product blurb file must contain a JSON object: {path}")
    return payload


def product_llm_evidence(
    product_blurbs: dict[str, Any],
    product_id: Any,
) -> str | None:
    """Return the exact LLM-facing product text for one product, if present.

    The preferred source is ``product_blurbs[product_id]["llm_str"]`` because
    that is the compact product description intended for LLM prompting. Storing
    it beside each retrieved product makes the review file self-contained: when
    the final response claims a material, style, dimension, or use case, the
    reviewer can immediately check whether that claim is grounded.

    A direct string value is accepted as a defensive fallback in case a future
    artifact maps ``product_id -> llm_string`` directly. Other shapes return
    ``None`` rather than inventing evidence from unrelated fields.
    """
    if not product_id:
        return None

    source_record = product_blurbs.get(str(product_id))
    if isinstance(source_record, str):
        return source_record
    if isinstance(source_record, dict):
        value = source_record.get("llm_str")
        if isinstance(value, str):
            return value
    return None


def serialize_target_product(
    case: RetrievalLlmEvalCase,
    product_blurbs: dict[str, Any],
) -> dict[str, Any] | None:
    """Return target-product metadata plus the LLM-facing evidence string.

    The target product is part of the eval-case design, not a product chosen by
    the running system. Including its ``llm_evidence`` lets the reviewer check
    whether the case's intended target actually satisfies the user request. That
    is separate from judging whether retrieval found the target or an equivalent
    item.
    """
    if not case.target_product_id:
        return None

    return {
        "product_id": case.target_product_id,
        "title": case.target_title,
        "llm_evidence": product_llm_evidence(product_blurbs, case.target_product_id),
    }


def serialize_retrieved_products(
    diagnostics: dict[str, Any],
    product_blurbs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert retrieval diagnostics into editable candidate judgments.

    Diagnostics already provide a compact ``top_products`` list. This function
    adds rank, blank human relevance fields, and ``llm_evidence`` copied from
    the configured product-blurb artifact. Keeping evidence inline is deliberate:
    it makes each retrieved-product judgment auditable without a paired reference
    file, while still limiting the extra text to products that were actually
    returned for a case.
    """
    products = diagnostics.get("top_products") or []
    serialized: list[dict[str, Any]] = []

    for rank, product in enumerate(products, start=1):
        product_id = product.get("product_id")
        serialized.append(
            {
                "rank": rank,
                "product_id": product_id,
                "title": product.get("item_name"),
                "score": product.get("score"),
                "product_type": product.get("product_type"),
                "image_paths": product.get("image_paths", []),
                # This is the exact product text reviewers should use when
                # checking whether the final LLM response made unsupported
                # product claims. It is intentionally named for its role in the
                # eval output rather than for the source artifact key
                # (product_blurbs[product_id]["llm_str"]).
                "llm_evidence": product_llm_evidence(product_blurbs, product_id),
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
        "target_or_equivalent_retrieved": None,
        "target_product_appropriate": None,
        "retrieval_quality": None,
        "llm_product_decision_correct": None,
        "llm_response_quality": None,
        "llm_response_grounded": None,
        "unsupported_claims": [],
        "contradicted_claims": [],
        "should_have_refused_or_said_no_match": None,
        "human_notes": "",
    }


def chosen_product_title(payload: dict[str, Any]) -> str | None:
    """Return the chosen product title from a ShopTalk reply payload, if any."""
    chosen_product = payload.get("chosen_product")
    if chosen_product is None:
        return None
    return getattr(chosen_product, "item_name", None)


def run_case(
    recommender: ShopTalkRecommender,
    case: RetrievalLlmEvalCase,
    product_blurbs: dict[str, Any],
) -> dict[str, Any]:
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
        "target_product": serialize_target_product(case, product_blurbs),
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
        "retrieved_products": serialize_retrieved_products(diagnostics, product_blurbs),
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
    product_blurbs: dict[str, Any],
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
            return run_case(recommender, case, product_blurbs)
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
            "schema_version": EXPECTED_SCHEMA_VERSION,
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
                "target_or_equivalent_retrieved": (
                    "true/false/null; whether retrieved_products include the exact target "
                    "or a human-judged equivalent satisfying the user's specified attributes"
                ),
                "target_product_appropriate": (
                    "true/false/null; optional review of whether the case target itself "
                    "is an appropriate answer to the query, based on target_product.llm_evidence"
                ),
                "retrieval_quality": "2/1/0/null; overall quality of retrieved set",
                "llm_product_decision_correct": (
                    "true/false/null; whether the LLM made the correct product-selection "
                    "decision, including correctly choosing no product for no-match cases"
                ),
                "llm_response_quality": "2/1/0/null; overall final answer quality",
                "llm_response_grounded": "true/false/null; whether response sticks to retrieved products/catalog evidence",
                "unsupported_claims": "list[str]; specific response claims not supported by retrieved_products[*].llm_evidence",
                "contradicted_claims": "list[str]; specific response claims contradicted by retrieved_products[*].llm_evidence",
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


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def latest_reviewed_judgment_path(reviewed_dir: Path, *, prefix: str) -> Path:
    """Return the highest-numbered reviewed judgment file for ``prefix``.

    Generated and reviewed files share the same run number. Selecting the newest
    reviewed copy removes the need to paste a newly generated filename into the
    INI before every scoring run.
    """
    if not reviewed_dir.exists():
        raise FileNotFoundError(f"Reviewed judgment directory not found: {reviewed_dir}")

    matches: list[tuple[int, Path]] = []
    marker = f"{prefix}_"
    suffix = "_reviewed.json"
    for candidate in reviewed_dir.iterdir():
        name = candidate.name
        if not candidate.is_file() or not name.startswith(marker) or not name.endswith(suffix):
            continue
        number_text = name[len(marker) : -len(suffix)]
        if number_text.isdigit():
            matches.append((int(number_text), candidate))

    if not matches:
        raise FileNotFoundError(
            f"No reviewed judgment files matching {prefix}_NNN_reviewed.json "
            f"were found in {reviewed_dir}"
        )
    return max(matches, key=lambda item: item[0])[1]


def configured_judgment_path(parser: configparser.ConfigParser) -> Path:
    """Return an explicit reviewed file or automatically select the newest one."""
    value = _config_value(parser, "score", "judgments", fallback="auto")
    if value.lower() != "auto":
        return Path(value)

    reviewed_dir = Path(
        parser.get("eval", "reviewed_output_dir", fallback=str(DEFAULT_REVIEWED_DIR))
    )
    prefix = parser.get("eval", "output_prefix", fallback=DEFAULT_OUTPUT_PREFIX)
    return latest_reviewed_judgment_path(reviewed_dir, prefix=prefix)


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


def load_score_args(config_path: Path) -> argparse.Namespace:
    """Load scorer settings from the same INI file used by the generator.

    The scorer still works with the same internal argument names it used before;
    this function only moves those values out of the command line and into the
    config file.
    """
    if not config_path.exists():
        if  Path(__file__).with_name(config_path.name).exists():
            config_path = Path(__file__).with_name(config_path.name)
        else:
            raise FileNotFoundError(f"Eval config file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Keep the old argparse-like shape internally while moving user-facing
    # configuration into the shared eval INI file.
    return argparse.Namespace(
        judgments=configured_judgment_path(parser),
        output=_auto_path(parser, "score", "output"),
        output_dir=Path(parser.get("score", "output_dir", fallback=str(DEFAULT_SCORED_DIR))),
        output_prefix=parser.get(
            "score",
            "output_prefix",
            fallback=DEFAULT_SCORE_OUTPUT_PREFIX,
        ),
        allow_unjudged=parser.getboolean("score", "allow_unjudged", fallback=False),
    )


def load_judgments(path: Path) -> dict[str, Any]:
    """Load and minimally validate a judgment JSON file.

    The eval format is still evolving, so this scorer intentionally accepts only
    the current schema. That keeps the code simple while the project is still in
    development and avoids pretending old generated artifacts are first-class
    compatibility targets.
    """
    if not path.exists():
        raise FileNotFoundError(f"Judgment file not found: {path}")

    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)

    # Schema validation is deliberately minimal but important: the scorer should
    # fail loudly if pointed at an unrelated JSON file or a future incompatible
    # judgment format.
    schema_version = payload.get("metadata", {}).get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version!r}; "
            f"expected {EXPECTED_SCHEMA_VERSION!r}"
        )

    if not isinstance(payload.get("cases"), list):
        raise ValueError("Judgment file must contain a top-level 'cases' list")

    return payload


def next_numbered_output_path(output_dir: Path, *, prefix: str, suffix: str) -> Path:
    """Return the next available numbered output path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = output_dir / f"{prefix}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def default_scored_output_path(output_dir: Path, judgment_path: Path) -> Path:
    """Return the default score report path for one judgment file.

    The report name is derived from the reviewed judgment file so it remains
    obvious which JSON produced the metrics report. The first report uses:

        <judgment-file-stem>_scored.txt

    Later reports for the same judgment file get numbered suffixes rather than
    overwriting the earlier report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{judgment_path.stem}_scored"
    first_candidate = output_dir / f"{base_name}_001.txt"
    if not first_candidate.exists():
        return first_candidate

    index = 2
    while True:
        candidate = output_dir / f"{base_name}_{index:03d}.txt"
        if not candidate.exists():
            return candidate
        index += 1


def is_positive_case(case: dict[str, Any]) -> bool:
    """Return True when the catalog is expected to contain a good match."""
    return bool(case.get("expected_available"))


def is_missing_product_case(case: dict[str, Any]) -> bool:
    """Return True when the case deliberately asks for an unavailable product."""
    return not is_positive_case(case)


def judged_number(value: Any) -> float | None:
    """Return numeric judgment values as floats, preserving unjudged nulls.

    Human-edited JSON may contain numbers as ints/floats. This helper avoids
    spreading isinstance checks throughout metric code and treats booleans as not
    numeric because bool is a subclass of int in Python.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def judged_bool(value: Any) -> bool | None:
    """Return boolean judgment values, preserving unjudged nulls."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def product_relevance(product: dict[str, Any]) -> float | None:
    """Return the human product relevance score for one retrieved product."""
    return judged_number(product.get("retrieval_relevance"))


def top_k_products(case: dict[str, Any], k: int) -> list[dict[str, Any]]:
    """Return retrieved products up to rank k, sorted defensively by rank."""
    products = case.get("retrieved_products") or []
    return sorted(products, key=lambda product: product.get("rank", 10**9))[:k]


def first_strict_relevant_rank(case: dict[str, Any], *, threshold: float = 2.0) -> int | None:
    """Return the rank of the first product with relevance >= threshold."""
    for product in top_k_products(case, k=10**9):
        relevance = product_relevance(product)
        if relevance is not None and relevance >= threshold:
            return int(product.get("rank", 10**9))
    return None


def hit_at_k(case: dict[str, Any], *, k: int, threshold: float) -> bool | None:
    """Return whether any top-k product meets the relevance threshold.

    None means the case does not have enough product-level judgments to score the
    metric honestly. The report separates unjudged counts from metric values.
    """
    products = top_k_products(case, k)
    if not products:
        return False

    saw_judged_product = False
    for product in products:
        relevance = product_relevance(product)
        if relevance is None:
            continue
        saw_judged_product = True
        if relevance >= threshold:
            return True

    return False if saw_judged_product else None


def mean_relevance_at_k(case: dict[str, Any], *, k: int) -> float | None:
    """Return mean human relevance of judged top-k products for one case."""
    judged_values = [
        relevance
        for product in top_k_products(case, k)
        if (relevance := product_relevance(product)) is not None
    ]
    return mean(judged_values) if judged_values else None


def exact_target_product_retrieved_from_products(case: dict[str, Any]) -> bool | None:
    """Infer whether the exact target product ID appears in retrieved products.

    Exact target retrieval is a diagnostic for case construction and retrieval
    behavior. It is intentionally separate from the primary shopping-success
    field, ``target_or_equivalent_retrieved``, because natural user queries may
    be satisfied by products other than the product that inspired the case.
    """
    target_product_id = case.get("target_product_id")
    if not target_product_id:
        return None

    for product in case.get("retrieved_products") or []:
        if product.get("product_id") == target_product_id:
            return True
    return False


def target_or_equivalent_retrieved(case: dict[str, Any]) -> bool | None:
    """Return the human judgment for retrieval of a target or equivalent item."""
    return judged_bool(human_eval(case).get("target_or_equivalent_retrieved"))


def human_eval(case: dict[str, Any]) -> dict[str, Any]:
    """Return the per-case human_eval object, defaulting to an empty dict."""
    return case.get("human_eval") or {}


def case_has_chosen_product(case: dict[str, Any]) -> bool:
    """Return True when the LLM actually selected a catalog product.

    This helper is used for false-recommendation metrics on missing-product cases.
    The explicit human judgment field, ``llm_product_decision_correct``, is now
    scored separately because a good LLM decision can be either choosing a good
    product or correctly choosing no product.
    """
    system_output = case.get("system_output") or {}
    return system_output.get("chosen_product_id") is not None


def average_bool(values: list[bool | None]) -> float | None:
    """Average judged booleans as rates, ignoring nulls."""
    judged = [value for value in values if value is not None]
    if not judged:
        return None
    return sum(1 for value in judged if value) / len(judged)


def average_number(values: list[float | None]) -> float | None:
    """Average judged numeric values, ignoring nulls."""
    judged = [value for value in values if value is not None]
    return mean(judged) if judged else None


def compute_retrieval_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute retrieval-side metrics from human product/case judgments.

    The first version scores the current single ShopTalk pipeline. When you later
    add multiple retrieval modes, the judgment schema can be extended with per-mode
    rankings and this function can compute the same metrics by mode.
    """
    positive_cases = [case for case in cases if is_positive_case(case)]

    target_or_equivalent_values = [
        target_or_equivalent_retrieved(case) for case in positive_cases
    ]
    exact_target_retrieved_values = [
        exact_target_product_retrieved_from_products(case) for case in positive_cases
    ]

    # Strict metrics count only strong matches. The lenient Hit@5 metric also
    # credits acceptable partial matches, which is useful for shopping queries
    # where a near miss may still be worth showing.
    hit_at_1_strict = [hit_at_k(case, k=1, threshold=2.0) for case in positive_cases]
    hit_at_5_strict = [hit_at_k(case, k=5, threshold=2.0) for case in positive_cases]
    hit_at_5_lenient = [hit_at_k(case, k=5, threshold=1.0) for case in positive_cases]

    reciprocal_ranks: list[float | None] = []
    for case in positive_cases:
        # MRR rewards placing the first strong match high in the retrieved list.
        first_rank = first_strict_relevant_rank(case, threshold=2.0)
        reciprocal_ranks.append(None if first_rank is None else 1.0 / first_rank)

    return {
        # Missing-product cases are excluded from ranking metrics because there
        # is no expected positive target to retrieve. They still contribute to
        # overall retrieval_quality below.
        "positive_cases": len(positive_cases),
        "target_or_equivalent_retrieved_rate": average_bool(target_or_equivalent_values),
        "exact_target_retrieved_rate_diagnostic": average_bool(exact_target_retrieved_values),
        "strict_hit_at_1": average_bool(hit_at_1_strict),
        "strict_hit_at_5": average_bool(hit_at_5_strict),
        "lenient_hit_at_5": average_bool(hit_at_5_lenient),
        "mrr_strict": average_number(reciprocal_ranks),
        "mean_top1_relevance": average_number(
            [mean_relevance_at_k(case, k=1) for case in positive_cases]
        ),
        "mean_relevance_at_5": average_number(
            [mean_relevance_at_k(case, k=5) for case in positive_cases]
        ),
        "mean_retrieval_quality": average_number(
            [judged_number(human_eval(case).get("retrieval_quality")) for case in cases]
        ),
    }


def compute_llm_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute LLM-response metrics from per-case human judgments."""
    positive_cases = [case for case in cases if is_positive_case(case)]
    missing_cases = [case for case in cases if is_missing_product_case(case)]

    return {
        # Missing-product cases are excluded from ranking metrics because there
        # is no expected positive target to retrieve. They still contribute to
        # overall retrieval_quality below.
        "positive_cases": len(positive_cases),
        "missing_product_cases": len(missing_cases),
        "product_decision_correct_rate_positive": average_bool(
            [
                judged_bool(human_eval(case).get("llm_product_decision_correct"))
                for case in positive_cases
            ]
        ),
        "mean_response_quality_all": average_number(
            [judged_number(human_eval(case).get("llm_response_quality")) for case in cases]
        ),
        "grounded_response_rate_all": average_bool(
            [judged_bool(human_eval(case).get("llm_response_grounded")) for case in cases]
        ),
        # False recommendations are measured from observed system behavior: did the
        # LLM choose any product when the catalog should have had no match?
        "false_recommendation_rate_missing": average_bool(
            [case_has_chosen_product(case) for case in missing_cases]
        ),
        "correct_no_match_rate_missing": average_bool(
            [judged_bool(human_eval(case).get("should_have_refused_or_said_no_match")) for case in missing_cases]
        ),
    }


def find_unjudged_fields(cases: list[dict[str, Any]]) -> list[str]:
    """Return human-readable warnings for null judgment fields.

    Nulls are allowed while a judgment file is still being edited, but they should
    be visible. Silent null handling would make a partial review look cleaner than
    it actually is.
    """
    warnings: list[str] = []

    required_case_fields = [
        "retrieval_quality",
        "llm_product_decision_correct",
        "llm_response_quality",
        "llm_response_grounded",
        "should_have_refused_or_said_no_match",
    ]

    for case in cases:
        case_id = case.get("case_id", "<unknown>")
        per_case_eval = human_eval(case)

        if target_or_equivalent_retrieved(case) is None:
            warnings.append(
                f"{case_id}: human_eval.target_or_equivalent_retrieved is null"
            )

        for field in required_case_fields:
            if per_case_eval.get(field) is None:
                warnings.append(f"{case_id}: human_eval.{field} is null")

        for product in case.get("retrieved_products") or []:
            product_id = product.get("product_id", "<unknown>")
            if product.get("retrieval_relevance") is None:
                warnings.append(
                    f"{case_id}: retrieved product {product_id} has null retrieval_relevance"
                )

    return warnings


def summarize_failures(cases: list[dict[str, Any]]) -> list[str]:
    """Collect concise failure notes for the report.

    These are not formal metrics. They are meant to speed up inspection by showing
    where retrieval, LLM choice, grounding, or no-match behavior looked weak.
    """
    failures: list[str] = []

    for case in cases:
        case_id = case.get("case_id", "<unknown>")
        eval_obj = human_eval(case)

        target_or_equivalent = target_or_equivalent_retrieved(case)
        if is_positive_case(case) and target_or_equivalent is False:
            failures.append(f"{case_id}: no target-or-equivalent product was retrieved")

        retrieval_quality = judged_number(eval_obj.get("retrieval_quality"))
        if retrieval_quality == 0:
            failures.append(f"{case_id}: retrieval set judged bad")

        product_decision_correct = judged_bool(
            eval_obj.get("llm_product_decision_correct")
        )
        if product_decision_correct is False:
            failures.append(f"{case_id}: LLM product decision judged incorrect")

        grounded = judged_bool(eval_obj.get("llm_response_grounded"))
        if grounded is False:
            failures.append(f"{case_id}: final LLM response was not grounded")

        no_match_expected = judged_bool(eval_obj.get("should_have_refused_or_said_no_match"))
        if (
            is_missing_product_case(case)
            and no_match_expected is True
            and case_has_chosen_product(case)
        ):
            failures.append(
                f"{case_id}: missing-product case appears to have received a false recommendation"
            )

    return failures


def format_metric(value: Any) -> str:
    """Format metric values for a readable fixed-width report."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_metrics_report(
    *,
    payload: dict[str, Any],
    judgment_path: Path,
    output_path: Path,
    warnings: list[str],
    retrieval_metrics: dict[str, Any],
    llm_metrics: dict[str, Any],
    failures: list[str],
) -> None:
    """Write a human-readable metrics report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = payload.get("metadata", {})
    cases = payload.get("cases", [])

    with output_path.open("w", encoding="utf-8") as outfile:
        # redirect_stdout keeps the report formatting simple while still writing
        # directly to a durable numbered text file instead of the console.
        with redirect_stdout(outfile):
            print("ShopTalk Retrieval + LLM Eval Metrics")
            print("====================================")
            print()
            print("Run metadata")
            print("------------")
            print(f"judgments:    {judgment_path}")
            print(f"cases_path:   {metadata.get('cases_path', '<unknown>')}")
            print(f"created_at:   {metadata.get('created_at', '<unknown>')}")
            print(f"model:        {metadata.get('model_name', '<unknown>')}")
            print(f"temperature:  {metadata.get('temperature', '<unknown>')}")
            print(f"total cases:  {len(cases)}")
            print()

            print("Judgment completeness")
            print("---------------------")
            print(f"unjudged fields/products: {len(warnings)}")
            if warnings:
                print("First 25 warnings:")
                for warning in warnings[:25]:
                    print(f"  - {warning}")
                if len(warnings) > 25:
                    print(f"  ... {len(warnings) - 25} more")
            print()

            print("Retrieval metrics")
            print("-----------------")
            for key, value in retrieval_metrics.items():
                print(f"{key:32} {format_metric(value)}")
            print()

            print("LLM response metrics")
            print("--------------------")
            for key, value in llm_metrics.items():
                print(f"{key:32} {format_metric(value)}")
            print()

            print("Failure summary")
            print("---------------")
            if failures:
                for failure in failures:
                    print(f"- {failure}")
            else:
                print("No judged failures detected.")

def generate_main(config_path: Path = DEFAULT_EVAL_CONFIG_PATH) -> int:
    """Run preset ShopTalk cases and write a hand-editable judgment JSON file.

    The generated file is intentionally self-contained for human review. Each
    retrieved product includes ``llm_evidence`` copied from
    ``product_blurbs[product_id]["llm_str"]``. That keeps grounding checks close
    to the response being judged without adding a separate evidence file or run
    manifest while the eval format is still changing.
    """
  
    args = load_eval_args(config_path)
    cases = load_jsonl_cases(args.cases, limit=args.limit)
    if args.output is None:
        output_path, reviewed_output_path = next_linked_judgment_paths(
            args.output_dir,
            args.reviewed_output_dir,
            prefix=args.output_prefix,
        )
    else:
        output_path = args.output
        reviewed_output_path = reviewed_path_for_generated(
            output_path,
            args.reviewed_output_dir,
        )
        if reviewed_output_path.exists():
            raise FileExistsError(
                f"Reviewed judgment copy already exists: {reviewed_output_path}"
            )

    if args.progress:
        print(f"Loaded {len(cases)} retrieval/LLM eval case(s) from {args.cases}", flush=True)
        print(f"Generated judgment output will be written to {output_path}", flush=True)
        print(f"Editable reviewed copy will be written to {reviewed_output_path}", flush=True)
        print(
            "Run pacing: "
            f"sleep_seconds_between_cases={args.sleep_seconds_between_cases}, "
            f"max_retries={args.max_retries}, "
            f"retry_sleep_seconds={args.retry_sleep_seconds}",
            flush=True,
        )
        print("Loading product blurb evidence...", flush=True)

    product_blurbs = load_product_blurb_source(Path(args.product_blurbs))

    if args.progress:
        print("Building ShopTalk recommender...", flush=True)

    recommender = build_recommender_from_args(args)

    if args.progress:
        print("Starting eval cases...", flush=True)

    judgment_cases: list[dict[str, Any]] = []
    total_cases = len(cases)
    # Cases are processed sequentially to preserve predictable console output and
    # to avoid hammering the LLM API during a small hand-reviewed eval run.
    for index, case in enumerate(cases, start=1):
        if args.progress:
            print(
                f"[{index}/{total_cases}] Running {case.case_id} "
                f"({case.query_type}, {case.category}); ",
                flush=True,
            )

        judgment_record = run_case_with_retries(
            recommender,
            case,
            product_blurbs,
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
    write_json(payload, reviewed_output_path)

    print(f"Wrote generated retrieval/LLM judgment file to {output_path}")
    print(f"Wrote editable reviewed copy to {reviewed_output_path}")
    return 0


def score_main(
    config_path: Path = DEFAULT_EVAL_CONFIG_PATH,
    *,
    allow_unjudged: bool = False,
) -> int:
    """Score a reviewed judgment JSON file and write a metrics report."""
    args = load_score_args(config_path)
    if allow_unjudged:
        args.allow_unjudged = True

    payload = load_judgments(args.judgments)
    cases = payload["cases"]

    warnings = find_unjudged_fields(cases)
    # By default, refuse to score partial judgment files. This prevents a half-
    # reviewed file from producing deceptively clean metrics.
    if warnings and not args.allow_unjudged:
        print(
            "Judgment file still has unjudged fields. "
            "Set [score] allow_unjudged=true in the INI or use "
            "--allow-unjudged to compute partial metrics anyway."
        )
        for warning in warnings[:25]:
            print(f"  - {warning}")
        if len(warnings) > 25:
            print(f"  ... {len(warnings) - 25} more")
        return 2

    retrieval_metrics = compute_retrieval_metrics(cases)
    llm_metrics = compute_llm_metrics(cases)
    failures = summarize_failures(cases)

    output_path = args.output or default_scored_output_path(
        args.output_dir,
        args.judgments,
    )
    write_metrics_report(
        payload=payload,
        judgment_path=args.judgments,
        output_path=output_path,
        warnings=warnings,
        retrieval_metrics=retrieval_metrics,
        llm_metrics=llm_metrics,
        failures=failures,
    )

    print(f"Scored judgment file: {args.judgments}")
    print(f"Wrote retrieval/LLM metrics report to {output_path}")
    return 0


def _add_config_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add positional and backward-compatible optional config arguments.

    The preferred interface is a single positional INI path::

        retrieval_llm_eval.py generate path/to/eval.ini

    ``-c/--config`` remains available so existing commands do not break. Both
    forms are optional because the historical default INI is still supported.
    Resolution and conflict checking happen after parsing in
    :func:`_resolve_config_path`.
    """
    subparser.add_argument(
        "config_path",
        nargs="?",
        type=Path,
        help=(
            "Path to the retrieval/LLM evaluation INI file. "
            f"Default: {DEFAULT_EVAL_CONFIG_PATH}"
        ),
    )
    subparser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        type=Path,
        help="Backward-compatible alternative to the positional INI path.",
    )


def _resolve_config_path(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> Path:
    """Resolve the selected INI path and reject contradictory arguments.

    Accepting both spellings is useful during migration, but silently choosing
    one when both differ would make runs hard to reproduce. The parser therefore
    reports an error for conflicting paths while allowing the same path to be
    supplied redundantly.
    """
    positional = args.config_path
    optional = args.config_option

    if positional is not None and optional is not None and positional != optional:
        parser.error(
            "configuration file specified twice with different paths; "
            "use either positional CONFIG or -c/--config"
        )

    return positional or optional or DEFAULT_EVAL_CONFIG_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the unified retrieval/LLM evaluation command line."""
    parser = argparse.ArgumentParser(
        description="Generate or score ShopTalk retrieval/LLM judgment files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run preset cases and write a hand-editable judgment JSON file.",
    )
    _add_config_arguments(generate_parser)

    score_parser = subparsers.add_parser(
        "score",
        help="Score a hand-edited retrieval/LLM judgment JSON file.",
    )
    _add_config_arguments(score_parser)
    score_parser.add_argument(
        "--allow-unjudged",
        action="store_true",
        help=(
            "Compute partial metrics even when judgment fields are still null. "
            "This overrides [score] allow_unjudged=false in the INI."
        ),
    )

    args = parser.parse_args(argv)
    args.config = _resolve_config_path(parser, args)
    return args


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    if args.command == "generate":
        return generate_main(args.config)
    if args.command == "score":
        return score_main(args.config, allow_unjudged=args.allow_unjudged)
    raise ValueError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
