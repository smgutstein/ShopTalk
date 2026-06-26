"""Evaluate the LLM product-decision layer without running the full app.

This evaluator intentionally targets the post-retrieval decision made by
``ConversationPolicy.decide_next_action``:

    conversation history + displayed/retrieved products -> RecommendationAction

It does not evaluate vector retrieval, ImageBind embeddings, Gradio behavior, or
whether the system should run a new search before retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # Supports: python -m server.evals.eval_llm_decision
    from ..recommender_core.config import load_shoptalk_config
    from ..recommender_core.conversation_policy import ConversationPolicy
    from ..recommender_core.product_candidate import ProductCandidate
    from ..recommender_core.utils import load_openai_api_key
    from ..shoptalk_paths import DEFAULT_CONFIG_PATH
except ImportError:  # Supports running from inside server/: python -m evals.eval_llm_decision
    from recommender_core.config import load_shoptalk_config
    from recommender_core.conversation_policy import ConversationPolicy
    from recommender_core.product_candidate import ProductCandidate
    from recommender_core.utils import load_openai_api_key
    from shoptalk_paths import DEFAULT_CONFIG_PATH

from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage


DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases_llm_decision.jsonl")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("eval_results")


@dataclass(frozen=True)
class EvalCase:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ConversationPolicy.decide_next_action on JSONL cases."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to JSONL eval cases.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ShopTalk config file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the eval model name from the config file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the eval temperature from the config file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to run.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write detailed CSV results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for numbered result files. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output-prefix",
        default="llm_decision_eval",
        help="Filename prefix for numbered result files.",
    )
    return parser.parse_args()


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
    results: list[EvalResult] = []
    for case in cases:
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

        results.append(
            EvalResult(
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
            )
        )
    return results


def next_numbered_output_path(
    output_dir: Path,
    *,
    prefix: str,
    suffix: str = ".txt",
) -> Path:
    """Return the next available numbered output path.

    Example: llm_decision_eval_001.txt, llm_decision_eval_002.txt, ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = output_dir / f"{prefix}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def write_report(
    results: list[EvalResult],
    *,
    output_path: Path,
    cases_path: Path,
    model_name: str,
    temperature: float,
) -> None:
    """Write the human-readable eval report to a file."""
    with output_path.open("w", encoding="utf-8") as outfile:
        with redirect_stdout(outfile):
            print_run_info(
                cases_path=cases_path,
                model_name=model_name,
                temperature=temperature,
            )
            print_results(results)
            print_detailed_cases(results)


def print_run_info(
    *,
    cases_path: Path,
    model_name: str,
    temperature: float,
) -> None:
    """Print eval run configuration."""
    print("Run configuration")
    print("-----------------")
    print(f"cases:       {cases_path}")
    print(f"model:       {model_name}")
    print(f"temperature: {temperature}")


def print_results(results: list[EvalResult]) -> None:
    passed_count = sum(result.passed for result in results)
    total_count = len(results)

    print(f"\nPassed {passed_count}/{total_count} cases")
    print("-" * 96)
    print(
        f"{'case_id':36} {'expected':14} {'actual':14} "
        f"{'expected_pid':14} {'actual_pid':14} result"
    )
    print("-" * 96)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.case_id[:36]:36} "
            f"{result.expected_action[:14]:14} "
            f"{result.actual_action[:14]:14} "
            f"{(result.expected_product_id or '')[:14]:14} "
            f"{(result.actual_product_id or '')[:14]:14} "
            f"{status}"
        )




def print_detailed_cases(results: list[EvalResult]) -> None:
    """Print every evaluated case grouped by category."""
    by_category: dict[str, list[EvalResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)

    print()
    print("Detailed cases by category")
    print("--------------------------")
    if not results:
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
            print(f"   Expected:     action={result.expected_action}, "
                  f"product_id={result.expected_product_id or '<none>'}")
            print(f"   Actual:       action={result.actual_action}, "
                  f"product_id={result.actual_product_id or '<none>'}")
            print("   Products:")
            if result.product_summaries:
                for product_summary in result.product_summaries:
                    print(f"     - {product_summary}")
            else:
                print("     - <none>")
            if result.reason:
                print(f"   Reason:       {result.reason}")
            print()

def write_csv(results: list[EvalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "case_id",
                "expected_action",
                "actual_action",
                "expected_product_id",
                "actual_product_id",
                "passed",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "expected_action": result.expected_action,
                    "actual_action": result.actual_action,
                    "expected_product_id": result.expected_product_id,
                    "actual_product_id": result.actual_product_id,
                    "passed": result.passed,
                }
            )


def main() -> int:
    args = parse_args()
    file_config = load_shoptalk_config(args.config)
    model_name = args.model or file_config.eval_model_name
    temperature = (
        args.temperature
        if args.temperature is not None
        else file_config.eval_temperature
    )
    cases = load_cases(args.cases, limit=args.limit)
    policy = build_conversation_policy(model_name, temperature)
    results = run_eval(policy, cases)

    output_path = next_numbered_output_path(
        args.output_dir,
        prefix=args.output_prefix,
    )
    write_report(
        results,
        output_path=output_path,
        cases_path=args.cases,
        model_name=model_name,
        temperature=temperature,
    )
    print(f"Wrote eval results to {output_path}")

    if args.output_csv is not None:
        write_csv(results, args.output_csv)
        print(f"Wrote CSV results to {args.output_csv}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
