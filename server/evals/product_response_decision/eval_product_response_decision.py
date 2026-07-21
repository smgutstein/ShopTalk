"""Evaluate the LLM product-decision layer without running the full app.

This evaluator intentionally targets the post-retrieval decision made by
``ConversationPolicy.decide_next_action``:

    conversation history + displayed/retrieved products -> RecommendationAction

It does not evaluate vector retrieval, ImageBind embeddings, Gradio behavior, or
whether the system should run a new search before retrieval.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
from contextlib import redirect_stdout
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Any, Iterable

try:  # Supports: python -m server.evals.product_response_decision.eval_product_response_decision
    from ...recommender_core.conversation_policy import ConversationPolicy
    from ...recommender_core.product_candidate import ProductCandidate
    from ...recommender_core.utils import load_openai_api_key
except ImportError:  # Supports running from inside server/: python -m evals.product_response_decision.eval_product_response_decision
    from recommender_core.conversation_policy import ConversationPolicy
    from recommender_core.product_candidate import ProductCandidate
    from recommender_core.utils import load_openai_api_key

from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage


# Keep the default paths next to this module so the eval can be run from the
# repository root without supplying a long list of command-line arguments.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("product_response_decision_eval.ini")


@dataclass(frozen=True)
class EvalCase:
    """One normalized post-retrieval decision test case.

    The raw JSONL file is intentionally simple and hand-editable. This dataclass
    is the stricter in-memory representation used after the JSON has been parsed
    into LangChain messages and ProductCandidate objects.

    The evaluator supplies ``found_products`` and ``source_knowledge`` directly
    because this test is not supposed to call FAISS/ImageBind retrieval. It only
    asks whether the policy reacts correctly to the candidate products it was
    handed.
    """
    case_id: str
    category: str
    conversation_history: list[Any]
    found_products: dict[str, ProductCandidate]
    source_knowledge: str
    expected_action: str
    expected_product_id: str | None
    reason: str


@dataclass(frozen=True)
class EvalResult:
    """Result of running one EvalCase through ConversationPolicy.

    The report stores both compact machine-checkable fields and richer context
    such as the latest user utterance and product summaries. That keeps failures
    inspectable without reopening the JSONL case file.
    """
    case_id: str
    category: str
    latest_user: str
    product_summaries: list[str]
    expected_action: str
    actual_action: str
    expected_product_id: str | None
    actual_product_id: str | None
    passed: bool
    reason: str
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluator config selector.

    Every setting that defines an evaluation run lives in the selected INI file.
    This leaves one reproducible source of truth instead of mixing config values
    with command-line overrides.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate ConversationPolicy.decide_next_action on JSONL cases."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the evaluator config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class ProductResponseDecisionEvalConfig:
    """Resolved settings for one product-response decision evaluation run."""

    model_name: str
    temperature: float
    cases_path: Path
    limit: int | None
    output_dir: Path
    output_prefix: str
    detail_level: str


def _required(config: configparser.ConfigParser, section: str, option: str) -> str:
    """Return a required non-empty INI value with a clear error message."""
    if not config.has_section(section):
        raise ValueError(f"Missing required config section [{section}]")
    if not config.has_option(section, option):
        raise ValueError(f"Missing required config setting [{section}] {option}")
    value = config.get(section, option).strip()
    if not value:
        raise ValueError(f"Config setting [{section}] {option} must not be empty")
    return value


def _optional_limit(value: str) -> int | None:
    """Parse a positive integer limit or the explicit value ``none``."""
    value = value.strip().lower()
    if value in {"", "none"}:
        return None
    limit = int(value)
    if limit <= 0:
        raise ValueError("Config setting [eval] limit must be positive or 'none'")
    return limit


def _parse_detail_level(value: str) -> str:
    """Parse the shared report-detail setting."""
    detail_level = value.strip().lower()
    if detail_level not in {"all", "failures"}:
        raise ValueError("Config setting [output] detail_level must be 'all' or 'failures'")
    return detail_level


def load_eval_config(path: Path) -> ProductResponseDecisionEvalConfig:
    """Load and validate the evaluator-specific INI file."""
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read evaluator config: {path}")

    return ProductResponseDecisionEvalConfig(
        model_name=_required(parser, "model", "model_name"),
        temperature=float(_required(parser, "model", "temperature")),
        cases_path=Path(_required(parser, "eval", "cases")),
        limit=_optional_limit(_required(parser, "eval", "limit")),
        output_dir=Path(_required(parser, "output", "output_dir")),
        output_prefix=_required(parser, "output", "output_prefix"),
        detail_level=_parse_detail_level(_required(parser, "output", "detail_level")),
    )


def build_conversation_policy(model_name: str, temperature: float) -> ConversationPolicy:
    """Build only the LLM policy, not the full recommender runtime."""
    from langchain_openai import ChatOpenAI

    api_key = load_openai_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    chat_model = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )
    return ConversationPolicy(chat_model)


def load_cases(path: Path, limit: int | None = None) -> list[EvalCase]:
    """Load JSONL cases and convert them to EvalCase objects.

    Empty lines are ignored to make the case file easier to edit by hand. JSON
    syntax errors include the line number so a broken case can be fixed quickly.
    """
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}")
    if path.suffix != ".jsonl":
        raise ValueError(f"Product-response case file must be JSONL: {path}")

    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                case_data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            cases.append(parse_case(case_data, line_number=line_number))
            if limit is not None and len(cases) >= limit:
                break

    return cases


def parse_case(case_data: dict[str, Any], *, line_number: int) -> EvalCase:
    """Normalize one raw JSON case into the evaluator's internal shape.

    If ``source_knowledge`` is omitted, it is built from the candidate products so
    cases do not have to duplicate the same product text twice.
    """
    case_id = case_data.get("case_id") or f"line_{line_number}"
    conversation_history = parse_messages(case_data.get("messages", []))
    found_products = parse_products(case_data.get("products", []))
    source_knowledge = case_data.get("source_knowledge") or build_source_knowledge(found_products)

    expected = case_data.get("expected", {})
    expected_action = expected.get("action")
    if expected_action is None:
        raise ValueError(f"Case {case_id!r} is missing expected.action")

    return EvalCase(
        case_id=case_id,
        category=case_data.get("category", "uncategorized"),
        conversation_history=conversation_history,
        found_products=found_products,
        source_knowledge=source_knowledge,
        expected_action=expected_action,
        expected_product_id=expected.get("product_id"),
        reason=case_data.get("reason", ""),
    )


def parse_messages(messages: Iterable[dict[str, str]]) -> list[Any]:
    """Convert JSON message dictionaries into LangChain message objects.

    The ConversationPolicy expects LangChain message classes, while the case file
    uses plain JSON objects so it remains readable and version-control friendly.
    """
    parsed_messages: list[Any] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role in {"user", "human"}:
            parsed_messages.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            parsed_messages.append(AIMessage(content=content))
        elif role == "system":
            parsed_messages.append(SystemMessage(content=content))
        else:
            raise ValueError(f"Unsupported message role: {role!r}")

    return parsed_messages


def parse_products(products: Iterable[dict[str, Any]]) -> dict[str, ProductCandidate]:
    """Convert JSON product stubs into ProductCandidate objects.

    The products are keyed by product_id because ``decide_next_action`` receives
    the same dictionary shape that the real recommender uses after retrieval.
    """
    found_products: dict[str, ProductCandidate] = {}
    for product in products:
        product_id = product["product_id"]
        found_products[product_id] = ProductCandidate(
            product_id=product_id,
            item_name=product.get("item_name", product_id),
            score=float(product.get("score", 0.0)),
            image_paths=tuple(product.get("image_paths", [])),
            product_type=product.get("product_type", "unknown"),
            llm_str=product.get("llm_str", product.get("item_name", product_id)),
        )
    return found_products


def build_source_knowledge(found_products: dict[str, ProductCandidate]) -> str:
    """Build the product text block passed to the LLM decision prompt.

    This mirrors the source-knowledge style used by the application: each
    product is rendered with a stable ID, title, type, and description.
    """
    return "\n\n".join(
        f"product_id: {product.product_id}\n"
        f"item_name: {product.item_name}\n"
        f"product_type: {product.product_type}\n"
        f"description: {product.llm_str}"
        for product in found_products.values()
    )


def latest_user_message(conversation_history: list[Any]) -> str:
    """Return the latest user/human message content for readable reporting."""
    for message in reversed(conversation_history):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def summarize_products(found_products: dict[str, ProductCandidate]) -> list[str]:
    """Return compact product summaries for readable reporting."""
    summaries: list[str] = []
    for product in found_products.values():
        summaries.append(
            f"{product.product_id}: {product.item_name} "
            f"[{product.product_type}, score={product.score:.3f}]"
        )
    return summaries


def run_eval(policy: ConversationPolicy, cases: list[EvalCase]) -> list[EvalResult]:
    """Run all cases, recording case-level errors without aborting the suite."""
    results: list[EvalResult] = []
    for case in tqdm(cases):
        try:
            action = policy.decide_next_action(
                conversation_history=case.conversation_history,
                found_products=case.found_products,
                source_knowledge=case.source_knowledge,
            )
            actual_product_id = action.product_id if action.action == "recommend" else None
            passed = (
                action.action == case.expected_action
                and actual_product_id == case.expected_product_id
            )
            result = EvalResult(
                case_id=case.case_id,
                category=case.category,
                latest_user=latest_user_message(case.conversation_history),
                product_summaries=summarize_products(case.found_products),
                expected_action=case.expected_action,
                actual_action=action.action,
                expected_product_id=case.expected_product_id,
                actual_product_id=actual_product_id,
                passed=passed,
                reason=case.reason,
                error=None,
            )
        except Exception as exc:
            result = EvalResult(
                case_id=case.case_id,
                category=case.category,
                latest_user=latest_user_message(case.conversation_history),
                product_summaries=summarize_products(case.found_products),
                expected_action=case.expected_action,
                actual_action=f"ERROR: {type(exc).__name__}",
                expected_product_id=case.expected_product_id,
                actual_product_id=None,
                passed=False,
                reason=case.reason,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
    return results


def next_numbered_output_path(output_dir: Path, *, prefix: str, suffix: str = ".txt") -> Path:
    """Return the next available numbered output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = output_dir / f"{prefix}_{index:03d}{suffix}"
        if not candidate.exists() and not candidate.with_suffix(".json").exists():
            return candidate
        index += 1


def print_run_info(*, config_path: Path, cases_path: Path, model_name: str, temperature: float) -> None:
    """Print eval run configuration."""
    print("Run configuration")
    print("-----------------")
    print(f"config:      {config_path}")
    print(f"cases:       {cases_path}")
    print(f"model:       {model_name}")
    print(f"temperature: {temperature}")


def build_summary(results: list[EvalResult]) -> dict[str, Any]:
    """Build common statistics plus product-selection statistics."""
    total = len(results)
    passed = sum(result.passed for result in results)
    action_correct = sum(result.actual_action == result.expected_action for result in results)
    expected_recommendations = [r for r in results if r.expected_action == "recommend"]
    product_correct = sum(
        r.actual_product_id == r.expected_product_id for r in expected_recommendations
    )
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({result.category for result in results}):
        category_results = [result for result in results if result.category == category]
        category_passed = sum(result.passed for result in category_results)
        by_category[category] = {
            "total": len(category_results),
            "passed": category_passed,
            "accuracy": category_passed / len(category_results),
        }
    confusion = Counter((result.expected_action, result.actual_action) for result in results)
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else None,
        "action_accuracy": action_correct / total if total else None,
        "product_id_accuracy_on_expected_recommendations": (
            product_correct / len(expected_recommendations) if expected_recommendations else None
        ),
        "errors": sum(result.error is not None for result in results),
        "by_category": by_category,
        "confusion_counts": [
            {"expected_action": expected, "actual_action": actual, "count": count}
            for (expected, actual), count in sorted(confusion.items())
        ],
    }


def print_summary(results: list[EvalResult]) -> None:
    """Print overall, category, action, and product-selection statistics."""
    summary = build_summary(results)
    print()
    print("Overall")
    print("-------")
    if not results:
        print("No cases evaluated.")
        return
    print(f"joint:      {summary['passed']}/{summary['total']} = {summary['accuracy']:.1%}")
    print(f"action:     {summary['action_accuracy']:.1%}")
    product_accuracy = summary["product_id_accuracy_on_expected_recommendations"]
    print(f"product ID: {product_accuracy:.1%}" if product_accuracy is not None else "product ID: n/a")
    print(f"errors:     {summary['errors']}")

    print()
    print("By category")
    print("-----------")
    for category, values in summary["by_category"].items():
        print(f"{category}: {values['passed']}/{values['total']} = {values['accuracy']:.1%}")

    print()
    print("Confusion counts")
    print("----------------")
    for item in summary["confusion_counts"]:
        print(
            f"expected={item['expected_action']:22s} "
            f"actual={item['actual_action']:22s} count={item['count']}"
        )


def print_failures(results: list[EvalResult]) -> None:
    """Print detailed failure information."""
    failures = [result for result in results if not result.passed]
    print()
    print("Failures")
    print("--------")
    if not failures:
        print("None.")
        return
    for result in failures:
        print(f"{result.case_id} [{result.category}]")
        print(f"  latest_user:        {result.latest_user}")
        print(f"  expected_action:    {result.expected_action}")
        print(f"  actual_action:      {result.actual_action}")
        print(f"  expected_product:   {result.expected_product_id}")
        print(f"  actual_product:     {result.actual_product_id}")
        print(f"  reason:             {result.reason}")
        if result.error:
            print(f"  error:              {result.error}")
        print()


def print_detailed_cases(results: list[EvalResult], *, detail_level: str) -> None:
    """Print evaluated cases grouped by category."""
    selected = results if detail_level == "all" else [r for r in results if not r.passed]
    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for result in selected:
        by_category[result.category].append(result)

    print()
    print("Detailed cases by category")
    print("--------------------------")
    if not selected:
        print("None.")
        return

    for category in sorted(by_category):
        category_results = by_category[category]
        category_passed = sum(result.passed for result in category_results)
        category_total = len(category_results)
        print()
        print(f"{category} ({category_passed}/{category_total} passed)")
        print("~" * (len(category) + len(f" ({category_passed}/{category_total} passed)")))
        for index, result in enumerate(category_results, start=1):
            status = "PASS" if result.passed else "FAIL"
            print(f"{index}. {result.case_id}: {status}")
            print(f"   Tested:       {result.latest_user}")
            print(
                f"   Expected:     action={result.expected_action}, "
                f"product_id={result.expected_product_id or '<none>'}"
            )
            print(
                f"   Actual:       action={result.actual_action}, "
                f"product_id={result.actual_product_id or '<none>'}"
            )
            print("   Products:")
            if result.product_summaries:
                for product_summary in result.product_summaries:
                    print(f"     - {product_summary}")
            else:
                print("     - <none>")
            if result.reason:
                print(f"   Reason:       {result.reason}")
            if result.error:
                print(f"   Error:        {result.error}")
            print()


def write_report(
    results: list[EvalResult],
    *,
    output_path: Path,
    detail_level: str,
    cases_path: Path,
    model_name: str,
    temperature: float,
    config_path: Path,
) -> None:
    """Write the human-readable eval report to a file."""
    with output_path.open("w", encoding="utf-8") as outfile:
        with redirect_stdout(outfile):
            print_run_info(
                config_path=config_path,
                cases_path=cases_path,
                model_name=model_name,
                temperature=temperature,
            )
            print_summary(results)
            print_failures(results)
            print_detailed_cases(results, detail_level=detail_level)


def write_json_results(
    results: list[EvalResult],
    *,
    output_path: Path,
    config_path: Path,
    cases_path: Path,
    model_name: str,
    temperature: float,
) -> None:
    """Write the shared machine-readable evaluation result structure."""
    payload = {
        "run": {
            "config": str(config_path),
            "cases": str(cases_path),
            "model": model_name,
            "temperature": temperature,
        },
        "summary": build_summary(results),
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the focused post-retrieval decision eval."""
    args = parse_args(argv)
    eval_config = load_eval_config(args.config)
    cases = load_cases(eval_config.cases_path, limit=eval_config.limit)
    policy = build_conversation_policy(eval_config.model_name, eval_config.temperature)
    results = run_eval(policy, cases)

    output_path = next_numbered_output_path(
        eval_config.output_dir,
        prefix=eval_config.output_prefix,
    )
    write_report(
        results,
        output_path=output_path,
        detail_level=eval_config.detail_level,
        config_path=args.config,
        cases_path=eval_config.cases_path,
        model_name=eval_config.model_name,
        temperature=eval_config.temperature,
    )
    json_output_path = output_path.with_suffix(".json")
    write_json_results(
        results,
        output_path=json_output_path,
        config_path=args.config,
        cases_path=eval_config.cases_path,
        model_name=eval_config.model_name,
        temperature=eval_config.temperature,
    )
    print(f"Wrote eval report to {output_path}")
    print(f"Wrote JSON results to {json_output_path}")
    return 0 if all(result.passed for result in results) else 1

if __name__ == "__main__":
    raise SystemExit(main())
