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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # Supports: python -m server.evals.eval_llm_decision
    from ..recommender_core.conversation_policy import ConversationPolicy
    from ..recommender_core.product_candidate import ProductCandidate
    from ..recommender_core.utils import load_openai_api_key
except ImportError:  # Supports running from inside server/: python -m evals.eval_llm_decision
    from recommender_core.conversation_policy import ConversationPolicy
    from recommender_core.product_candidate import ProductCandidate
    from recommender_core.utils import load_openai_api_key

from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage


DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases_llm_decision.jsonl")


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    conversation_history: list[Any]
    found_products: dict[str, ProductCandidate]
    source_knowledge: str
    expected_action: str
    expected_product_id: str | None


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    expected_action: str
    actual_action: str
    expected_product_id: str | None
    actual_product_id: str | None
    passed: bool


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
        "--model",
        default="gpt-4o",
        help="OpenAI model name used for the LLM decision layer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the decision model. Use 0.0 for repeatability.",
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
        conversation_history=conversation_history,
        found_products=found_products,
        source_knowledge=source_knowledge,
        expected_action=expected_action,
        expected_product_id=expected.get("product_id"),
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
                expected_action=case.expected_action,
                actual_action=action.action,
                expected_product_id=case.expected_product_id,
                actual_product_id=actual_product_id,
                passed=passed,
            )
        )
    return results


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
    cases = load_cases(args.cases, limit=args.limit)
    policy = build_conversation_policy(args.model, args.temperature)
    results = run_eval(policy, cases)
    print_results(results)

    if args.output_csv is not None:
        write_csv(results, args.output_csv)
        print(f"\nWrote CSV results to {args.output_csv}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
