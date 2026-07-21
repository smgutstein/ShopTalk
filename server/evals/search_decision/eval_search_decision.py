"""Evaluate the ShopTalk LLM search-decision layer.

This evaluator targets the pre-retrieval decision made by
ConversationPolicy.decide_search_action:

    conversation history -> SearchDecision

It does not evaluate vector retrieval, ImageBind embeddings, product selection,
Gradio behavior, or final response generation.

Run from the repository root:

    python -m server.evals.search_decision.eval_search_decision

To use a different complete run configuration:

    python -m server.evals.search_decision.eval_search_decision \
        --config path/to/search_decision_eval.ini
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
from tqdm import tqdm

from contextlib import redirect_stdout
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_classic.schema import AIMessage, HumanMessage

try:  # Supports: python -m server.evals.search_decision.eval_search_decision
    from ...recommender_core.conversation_policy import ConversationPolicy
    from ...recommender_core.utils import load_openai_api_key
except ImportError:  # Supports running from inside server/: python -m evals.search_decision.eval_search_decision
    from recommender_core.conversation_policy import ConversationPolicy
    from recommender_core.utils import load_openai_api_key


# Keep defaults beside this file so the command works from the repository root
# without requiring callers to remember the eval case path or output directory.
DEFAULT_EVAL_CONFIG_PATH = Path(__file__).with_name("search_decision_eval.ini")


@dataclass(frozen=True)
class EvalCase:
    """One normalized pre-retrieval search-decision test case."""

    case_id: str
    category: str
    conversation_history: list[Any]
    expected_action: str
    reason: str


@dataclass(frozen=True)
class EvalResult:
    """Result of one pre-retrieval search-routing case.

    This result captures only the routing decision and the optional query the LLM
    proposed. It deliberately does not include retrieved products because this
    evaluator stops before any vector search is executed.
    """
    case_id: str
    category: str
    latest_user: str
    expected_action: str
    actual_action: str
    search_query: str | None
    passed: bool
    reason: str
    error: str | None = None


def build_conversation_policy(model_name: str, temperature: float) -> ConversationPolicy:
    """Build the ConversationPolicy used for this evaluation.

    Only the policy wrapper and chat model are needed here. The full ShopTalk
    recommender, vector database, images, and product metadata are intentionally
    left out so this stays a targeted routing test.
    """
    from langchain_openai import ChatOpenAI

    api_key = load_openai_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    chat_model = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )

    return ConversationPolicy(chat_model)


def load_cases(path: Path) -> list[EvalCase]:
    """Load JSONL search-decision cases and normalize them to EvalCase objects."""
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}")
    if path.suffix != ".jsonl":
        raise ValueError(f"Search-decision case file must be JSONL: {path}")

    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case_data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            cases.append(parse_case(case_data, line_number=line_number))

    return cases


def parse_case(case_data: dict[str, Any], *, line_number: int) -> EvalCase:
    """Normalize one raw JSONL case into the search evaluator's case object."""
    case_id = case_data.get("case_id") or f"line_{line_number}"
    conversation_history = [
        message_from_dict(message) for message in case_data.get("messages", [])
    ]

    expected = case_data.get("expected", {})
    expected_action = expected.get("action")
    if expected_action is None:
        raise ValueError(f"Case {case_id!r} is missing expected.action")

    return EvalCase(
        case_id=case_id,
        category=case_data.get("category", "uncategorized"),
        conversation_history=conversation_history,
        expected_action=expected_action,
        reason=case_data.get("reason", ""),
    )


def message_from_dict(message: dict[str, str]):
    """Convert a case message dict into a LangChain message."""
    role = message.get("role")
    content = message.get("content", "")

    if role == "user":
        return HumanMessage(content=content)

    if role == "assistant":
        return AIMessage(content=content)

    raise ValueError(f"Unsupported message role: {role!r}")


def latest_user_message(conversation_history: list[Any]) -> str:
    """Return the latest user message for readable reporting."""
    for message in reversed(conversation_history):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def evaluate_case(policy: ConversationPolicy, case: EvalCase) -> EvalResult:
    """Run one EvalCase through the SearchDecision layer."""
    decision = policy.decide_search_action(case.conversation_history)

    actual_action = decision.action
    passed = actual_action == case.expected_action

    return EvalResult(
        case_id=case.case_id,
        category=case.category,
        latest_user=latest_user_message(case.conversation_history),
        expected_action=case.expected_action,
        actual_action=actual_action,
        search_query=decision.search_query,
        passed=passed,
        reason=case.reason,
        error=None,
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
        if not candidate.exists() and not candidate.with_suffix(".json").exists():
            return candidate
        index += 1


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
                cases_path=cases_path,
                model_name=model_name,
                temperature=temperature,
                config_path=config_path,
            )
            print_summary(results)
            print_failures(results)
            print_detailed_cases(results, detail_level=detail_level)


def print_run_info(
    *,
    cases_path: Path,
    model_name: str,
    temperature: float,
    config_path: Path,
) -> None:
    """Print eval run configuration."""
    print("Run configuration")
    print("-----------------")
    print(f"config:      {config_path}")
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
    print(f"errors: {sum(result.error is not None for result in results)}")

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
            print(f"   Expected:     {result.expected_action}")
            print(f"   Actual:       {result.actual_action}")
            print(f"   Search query: {result.search_query or '<none>'}")
            if result.reason:
                print(f"   Reason:       {result.reason}")
            if result.error:
                print(f"   Error:        {result.error}")
            print()


def build_summary(results: list[EvalResult]) -> dict[str, Any]:
    """Build the common machine-readable summary block."""
    total = len(results)
    passed = sum(result.passed for result in results)
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
        "errors": sum(result.error is not None for result in results),
        "by_category": by_category,
        "confusion_counts": [
            {"expected_action": expected, "actual_action": actual, "count": count}
            for (expected, actual), count in sorted(confusion.items())
        ],
    }


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluator config-file option."""
    parser = argparse.ArgumentParser(
        description="Evaluate SearchDecision behavior on labeled cases."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help=f"Path to the evaluator config file. Default: {DEFAULT_EVAL_CONFIG_PATH}",
    )
    return parser.parse_args(argv)


def _optional_int(value: str, *, field_name: str) -> int | None:
    """Parse an optional non-negative integer from the INI file."""
    normalized = value.strip().lower()
    if normalized in {"", "none"}:
        return None

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer or 'none': {value!r}") from exc

    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative: {parsed}")
    return parsed


def _optional_categories(value: str) -> list[str] | None:
    """Parse a comma-separated category filter from the INI file."""
    categories = [category.strip() for category in value.split(",") if category.strip()]
    return categories or None


def _parse_detail_level(value: str) -> str:
    """Parse the shared report-detail setting."""
    detail_level = value.strip().lower()
    if detail_level not in {"all", "failures"}:
        raise ValueError("output.detail_level must be 'all' or 'failures'")
    return detail_level


def load_eval_config(config_path: Path) -> argparse.Namespace:
    """Load all search-decision evaluation settings from one INI file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    required = {
        "model": ("model_name", "temperature"),
        "eval": ("cases", "limit", "categories"),
        "output": ("output_dir", "output_prefix", "detail_level"),
    }
    for section, options in required.items():
        if not parser.has_section(section):
            raise ValueError(f"Missing [{section}] section in {config_path}")
        for option in options:
            if not parser.has_option(section, option):
                raise ValueError(
                    f"Missing {option!r} in [{section}] section of {config_path}"
                )

    return argparse.Namespace(
        config_path=config_path,
        model_name=parser.get("model", "model_name").strip(),
        temperature=parser.getfloat("model", "temperature"),
        cases=Path(parser.get("eval", "cases")),
        limit=_optional_int(parser.get("eval", "limit"), field_name="eval.limit"),
        category=_optional_categories(parser.get("eval", "categories")),
        output_dir=Path(parser.get("output", "output_dir")),
        output_prefix=parser.get("output", "output_prefix").strip(),
        detail_level=_parse_detail_level(parser.get("output", "detail_level")),
    )


def select_cases(
    cases: list[EvalCase],
    *,
    categories: list[str] | None,
    limit: int | None,
) -> list[EvalCase]:
    """Filter cases according to the evaluator configuration.

    Category filtering is useful when debugging one decision boundary, while
    ``limit`` keeps quick smoke runs cheap during prompt iteration.
    """
    selected = cases

    if categories:
        wanted = set(categories)
        selected = [case for case in selected if case.category in wanted]

    if limit is not None:
        selected = selected[:limit]

    return selected


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the pre-retrieval search-decision eval."""
    cli_args = parse_args(argv)
    args = load_eval_config(cli_args.config)

    cases = load_cases(args.cases)
    cases = select_cases(
        cases,
        categories=args.category,
        limit=args.limit,
    )

    if not cases:
        print("No cases selected.")
        return 2

    policy = build_conversation_policy(
        model_name=args.model_name,
        temperature=args.temperature,
    )

    results: list[EvalResult] = []
    for case in tqdm(cases):
        try:
            result = evaluate_case(policy, case)
        except Exception as exc:  # Eval harness should report case-level failures.
            result = EvalResult(
                case_id=case.case_id,
                category=case.category,
                latest_user=latest_user_message(case.conversation_history),
                expected_action=case.expected_action,
                actual_action=f"ERROR: {type(exc).__name__}",
                search_query=None,
                passed=False,
                reason=case.reason,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)

    output_path = next_numbered_output_path(
        args.output_dir,
        prefix=args.output_prefix,
    )
    write_report(
        results,
        output_path=output_path,
        detail_level=args.detail_level,
        cases_path=args.cases,
        model_name=args.model_name,
        temperature=args.temperature,
        config_path=args.config_path,
    )

    json_output_path = output_path.with_suffix(".json")
    write_json_results(
        results,
        output_path=json_output_path,
        config_path=args.config_path,
        cases_path=args.cases,
        model_name=args.model_name,
        temperature=args.temperature,
    )

    print(f"Wrote eval report to {output_path}")
    print(f"Wrote JSON results to {json_output_path}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
