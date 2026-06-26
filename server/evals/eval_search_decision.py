"""Evaluate the ShopTalk LLM search-decision layer.

This evaluator targets the pre-retrieval decision made by
ConversationPolicy.decide_search_action:

    conversation history -> SearchDecision

It does not evaluate vector retrieval, ImageBind embeddings, product selection,
Gradio behavior, or final response generation.

Run from the repository root:

    python -m server.evals.eval_search_decision

Optionally:

    python -m server.evals.eval_search_decision \
        --cases server/evals/search_decision_cases.jsonl \
        --category boundary \
        --show-passes
"""

from __future__ import annotations

import argparse
import json
import os
from tqdm import tqdm

from contextlib import redirect_stdout
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_classic.schema import AIMessage, HumanMessage

try:  # Supports: python -m server.evals.eval_search_decision
    from ..recommender_core.config import load_shoptalk_config
    from ..recommender_core.conversation_policy import ConversationPolicy
    from ..recommender_core.utils import load_openai_api_key
    from ..shoptalk_paths import DEFAULT_CONFIG_PATH
except ImportError:  # Supports running from inside server/: python -m evals.eval_search_decision
    from recommender_core.config import load_shoptalk_config
    from recommender_core.conversation_policy import ConversationPolicy
    from recommender_core.utils import load_openai_api_key
    from shoptalk_paths import DEFAULT_CONFIG_PATH


DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases_search_decision.jsonl")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("eval_results")


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    category: str
    latest_user: str
    expected_action: str
    actual_action: str
    search_query: str | None
    passed: bool
    reason: str


def build_conversation_policy(model_name: str, temperature: float) -> ConversationPolicy:
    """Build the ConversationPolicy used for this evaluation."""
    from langchain_openai import ChatOpenAI

    api_key = load_openai_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    chat_model = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )

    return ConversationPolicy(chat_model)


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load search-decision eval cases from a JSON or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}")

    if path.suffix == ".jsonl":
        cases: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc
        return cases

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        cases = payload
    else:
        cases = payload.get("cases", [])

    if not isinstance(cases, list):
        raise ValueError(f"Expected a list of cases in {path}")

    return cases


def message_from_dict(message: dict[str, str]):
    """Convert a case-history message dict into a LangChain message."""
    role = message.get("role")
    content = message.get("content", "")

    if role == "user":
        return HumanMessage(content=content)

    if role == "assistant":
        return AIMessage(content=content)

    raise ValueError(f"Unsupported message role: {role!r}")


def build_conversation_history(case: dict[str, Any]):
    """Build conversation history including the latest user message."""
    history = [message_from_dict(message) for message in case.get("history", [])]
    history.append(HumanMessage(content=case["latest_user"]))
    return history


def evaluate_case(policy: ConversationPolicy, case: dict[str, Any]) -> EvalResult:
    """Run one eval case through the SearchDecision layer."""
    conversation_history = build_conversation_history(case)
    decision = policy.decide_search_action(conversation_history)

    expected_action = case["expected_search_action"]
    actual_action = decision.action
    passed = actual_action == expected_action

    return EvalResult(
        case_id=case["id"],
        category=case["category"],
        latest_user=case["latest_user"],
        expected_action=expected_action,
        actual_action=actual_action,
        search_query=decision.search_query,
        passed=passed,
        reason=case.get("reason", ""),
    )


def next_numbered_output_path(
    output_dir: Path,
    *,
    prefix: str,
    suffix: str = ".txt",
) -> Path:
    """Return the next available numbered output path.

    Example: search_decision_eval_001.txt, search_decision_eval_002.txt, ...
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
    show_passes: bool,
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
            print_summary(results)
            print_failures(results)
            print_detailed_cases(results)

            if show_passes:
                print_passes(results)


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


def print_summary(results: list[EvalResult]) -> None:
    """Print overall and category-level accuracy."""
    total = len(results)
    passed = sum(result.passed for result in results)

    print()
    print("Overall")
    print("-------")
    if total == 0:
        print("No cases evaluated.")
        return

    print(f"{passed}/{total} = {passed / total:.1%}")

    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    print()
    print("By category")
    print("-----------")
    for category in sorted(by_category):
        category_results = by_category[category]
        category_passed = sum(result.passed for result in category_results)
        category_total = len(category_results)
        print(
            f"{category}: "
            f"{category_passed}/{category_total} = "
            f"{category_passed / category_total:.1%}"
        )

    confusion = Counter(
        (result.expected_action, result.actual_action) for result in results
    )

    print()
    print("Confusion counts")
    print("----------------")
    for (expected, actual), count in sorted(confusion.items()):
        print(f"expected={expected:22s} actual={actual:22s} count={count}")


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
        print(f"  latest_user:  {result.latest_user}")
        print(f"  expected:     {result.expected_action}")
        print(f"  actual:       {result.actual_action}")
        print(f"  search_query: {result.search_query}")
        print(f"  reason:       {result.reason}")
        print()




def print_detailed_cases(results: list[EvalResult]) -> None:
    """Print every evaluated case grouped by category."""
    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

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
            print(f"   Expected:     {result.expected_action}")
            print(f"   Actual:       {result.actual_action}")
            print(f"   Search query: {result.search_query or '<none>'}")
            if result.reason:
                print(f"   Reason:       {result.reason}")
            print()

def print_passes(results: list[EvalResult]) -> None:
    """Print passing cases for inspection."""
    passes = [result for result in results if result.passed]

    print()
    print("Passes")
    print("------")
    if not passes:
        print("None.")
        return

    for result in passes:
        print(f"{result.case_id} [{result.category}]")
        print(f"  latest_user:  {result.latest_user}")
        print(f"  action:       {result.actual_action}")
        print(f"  search_query: {result.search_query}")
        print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SearchDecision behavior on labeled cases."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Path to JSON/JSONL case file. Default: {DEFAULT_CASES_PATH}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ShopTalk config file.",
    )
    parser.add_argument(
        "--model-name",
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
        help="Optionally evaluate only the first N selected cases.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Optionally evaluate only one or more categories. May be repeated.",
    )
    parser.add_argument(
        "--show-passes",
        action="store_true",
        help="Include passing cases in the written report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for numbered result files. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output-prefix",
        default="search_decision_eval",
        help="Filename prefix for numbered result files.",
    )
    return parser.parse_args(argv)


def select_cases(
    cases: list[dict[str, Any]],
    *,
    categories: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Filter cases according to CLI arguments."""
    selected = cases

    if categories:
        wanted = set(categories)
        selected = [case for case in selected if case.get("category") in wanted]

    if limit is not None:
        selected = selected[:limit]

    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cases = load_cases(args.cases)
    cases = select_cases(
        cases,
        categories=args.category,
        limit=args.limit,
    )

    if not cases:
        print("No cases selected.")
        return 2

    file_config = load_shoptalk_config(args.config)
    model_name = args.model_name or file_config.eval_model_name
    temperature = (
        args.temperature
        if args.temperature is not None
        else file_config.eval_temperature
    )
    policy = build_conversation_policy(
        model_name=model_name,
        temperature=temperature,
    )

    results: list[EvalResult] = []
    for case in tqdm(cases):
        try:
            result = evaluate_case(policy, case)
        except Exception as exc:  # Eval harness should report case-level failures.
            result = EvalResult(
                case_id=case.get("id", "<missing id>"),
                category=case.get("category", "<missing category>"),
                latest_user=case.get("latest_user", ""),
                expected_action=case.get("expected_search_action", "<missing expected>"),
                actual_action=f"ERROR: {type(exc).__name__}",
                search_query=None,
                passed=False,
                reason=str(exc),
            )
        results.append(result)

    output_path = next_numbered_output_path(
        args.output_dir,
        prefix=args.output_prefix,
    )
    write_report(
        results,
        output_path=output_path,
        show_passes=args.show_passes,
        cases_path=args.cases,
        model_name=model_name,
        temperature=temperature,
    )

    print(f"Wrote eval results to {output_path}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
