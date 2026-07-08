"""Score hand-edited ShopTalk retrieval/LLM judgment files.

This module is the second half of the evaluation workflow created by
`generate_retrieval_llm_judgments.py`:

    hand-edited judgment JSON -> retrieval metrics + LLM response metrics

It assumes a human has filled in the judgment fields in the generated JSON. The
scorer intentionally keeps the metrics simple and readable. The goal is to make
ShopTalk's behavior comparable across runs and later retrieval modes, not to
pretend this small hand-labeled eval set is a formal benchmark.
"""

from __future__ import annotations

import argparse
import configparser
import json
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean
from typing import Any


# The scorer writes reports beside the eval modules by default, matching the
# generator's output convention and avoiding accidental overwrites via numbering.
DEFAULT_RESULTS_DIR = Path(__file__).with_name("eval_results")
DEFAULT_OUTPUT_PREFIX = "retrieval_llm_metrics"
DEFAULT_EVAL_CONFIG_PATH = Path(__file__).with_name("retrieval_llm_eval.ini")
EXPECTED_SCHEMA_VERSION = "retrieval_llm_judgments_v1"


def _config_value(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: str,
) -> str:
    """Return a stripped INI value and reject undocumented blank values.

    The eval INI is intended to document normal scorer behavior. Optional values
    therefore use explicit sentinels such as ``auto`` and ``required`` instead of
    blank strings.
    """
    value = parser.get(section, option, fallback=fallback).strip()
    if value == "":
        raise ValueError(
            f"Blank value for [{section}] {option}. "
            "Use an explicit documented value such as 'auto' or 'required'."
        )
    return value


def _required_path(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> Path:
    """Return a required path, rejecting the documentation sentinel."""
    value = _config_value(parser, section, option, fallback="required")
    if value.lower() == "required":
        raise ValueError(
            f"[{section}] {option} is still set to 'required'. "
            "Set it to the reviewed judgment JSON file before scoring."
        )
    return Path(value)


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
        judgments=_required_path(parser, "score", "judgments"),
        output=_auto_path(parser, "score", "output"),
        output_dir=Path(parser.get("score", "output_dir", fallback=str(DEFAULT_RESULTS_DIR))),
        output_prefix=parser.get(
            "score",
            "output_prefix",
            fallback=DEFAULT_OUTPUT_PREFIX,
        ),
        allow_unjudged=parser.getboolean("score", "allow_unjudged", fallback=False),
    )


def load_judgments(path: Path) -> dict[str, Any]:
    """Load and minimally validate a judgment JSON file."""
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


def target_product_retrieved_from_products(case: dict[str, Any]) -> bool | None:
    """Infer target retrieval from retrieved product IDs when possible.

    The generated file also has a manual target_product_retrieved field. This
    helper is used as a fallback when that field is still null for positive cases.
    """
    target_product_id = case.get("target_product_id")
    if not target_product_id:
        return None

    for product in case.get("retrieved_products") or []:
        if product.get("product_id") == target_product_id:
            return True
    return False


def human_eval(case: dict[str, Any]) -> dict[str, Any]:
    """Return the per-case human_eval object, defaulting to an empty dict."""
    return case.get("human_eval") or {}


def case_has_chosen_product(case: dict[str, Any]) -> bool:
    """Return True when the LLM actually selected a catalog product.

    ``llm_chose_good_product`` is only meaningful when there is a selected
    product to judge. For dive-deeper or wrong-track/no-match responses, a null
    value is not an incomplete judgment; it means the field is not applicable.
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

    target_retrieved_values: list[bool | None] = []
    for case in positive_cases:
        # Prefer the human field when present, but infer from product IDs as a
        # convenience when the target product is explicit and the reviewer has
        # not manually filled in the case-level flag.
        manual_value = judged_bool(human_eval(case).get("target_product_retrieved"))
        target_retrieved_values.append(
            manual_value
            if manual_value is not None
            else target_product_retrieved_from_products(case)
        )

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
        "target_retrieved_rate": average_bool(target_retrieved_values),
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
        "good_choice_rate_positive": average_bool(
            [judged_bool(human_eval(case).get("llm_chose_good_product")) for case in positive_cases]
        ),
        "mean_response_quality_all": average_number(
            [judged_number(human_eval(case).get("llm_response_quality")) for case in cases]
        ),
        "grounded_response_rate_all": average_bool(
            [judged_bool(human_eval(case).get("llm_response_grounded")) for case in cases]
        ),
        # For missing-product cases, ``llm_chose_good_product`` is normally
        # null because there is no product recommendation to judge. A false
        # recommendation is therefore an observable system behavior: did the
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
        "target_product_retrieved",
        "retrieval_quality",
        "llm_response_quality",
        "llm_response_grounded",
        "should_have_refused_or_said_no_match",
    ]

    for case in cases:
        case_id = case.get("case_id", "<unknown>")
        per_case_eval = human_eval(case)
        for field in required_case_fields:
            if per_case_eval.get(field) is None:
                warnings.append(f"{case_id}: human_eval.{field} is null")

        # ``llm_chose_good_product`` is required only when the LLM chose a
        # product. If the LLM correctly produced a no-match/wrong-track or
        # dive-deeper response, null means not applicable rather than unjudged.
        if case_has_chosen_product(case) and per_case_eval.get("llm_chose_good_product") is None:
            warnings.append(f"{case_id}: human_eval.llm_chose_good_product is null")

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

        target_retrieved = judged_bool(eval_obj.get("target_product_retrieved"))
        if is_positive_case(case) and target_retrieved is False:
            failures.append(f"{case_id}: target product was not retrieved")

        retrieval_quality = judged_number(eval_obj.get("retrieval_quality"))
        if retrieval_quality == 0:
            failures.append(f"{case_id}: retrieval set judged bad")

        llm_chose_good_product = judged_bool(eval_obj.get("llm_chose_good_product"))
        if llm_chose_good_product is False:
            failures.append(f"{case_id}: LLM did not choose a good product")

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


def main() -> int:
    """CLI entry point."""
    cli_args = parse_args()
    args = load_score_args(cli_args.config)
    if cli_args.allow_unjudged:
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

    output_path = args.output or next_numbered_output_path(
        args.output_dir,
        prefix=args.output_prefix,
        suffix=".txt",
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

def parse_args() -> argparse.Namespace:
    """Parse the single eval-config argument."""
    parser = argparse.ArgumentParser(
        description="Score a hand-edited ShopTalk retrieval/LLM judgment JSON file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help=f"Path to retrieval/LLM eval INI file. Default: {DEFAULT_EVAL_CONFIG_PATH}",
    )
    parser.add_argument(
        "--allow-unjudged",
        action="store_true",
        help=(
            "Compute partial metrics even when judgment fields are still null. "
            "This overrides [score] allow_unjudged=false in the INI."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
