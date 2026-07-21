import json
from pathlib import Path

from server.evals.product_response_decision.eval_product_response_decision import (
    EvalResult as ProductEvalResult,
    build_summary as build_product_summary,
    run_eval,
    write_json_results as write_product_json,
)
from server.evals.search_decision.eval_search_decision import (
    EvalResult as SearchEvalResult,
    write_json_results as write_search_json,
)


def test_both_evaluators_write_shared_json_envelope(tmp_path):
    search_result = SearchEvalResult(
        case_id="search_1",
        category="initial_request",
        latest_user="Find shoes",
        expected_action="search",
        actual_action="search",
        search_query="wide walking shoes",
        passed=True,
        reason="",
        error=None,
    )
    product_result = ProductEvalResult(
        case_id="product_1",
        category="recommend",
        latest_user="Find shoes",
        product_summaries=["p1: Shoe"],
        expected_action="recommend",
        actual_action="recommend",
        expected_product_id="p1",
        actual_product_id="p1",
        passed=True,
        reason="",
        error=None,
    )

    search_path = tmp_path / "search.json"
    product_path = tmp_path / "product.json"
    common = dict(
        config_path=Path("eval.ini"),
        cases_path=Path("cases.jsonl"),
        model_name="test-model",
        temperature=0.0,
    )
    write_search_json([search_result], output_path=search_path, **common)
    write_product_json([product_result], output_path=product_path, **common)

    search_payload = json.loads(search_path.read_text())
    product_payload = json.loads(product_path.read_text())

    assert search_payload.keys() == product_payload.keys() == {"run", "summary", "results"}
    assert search_payload["run"].keys() == product_payload["run"].keys()
    for field in ("total", "passed", "accuracy", "errors", "by_category", "confusion_counts"):
        assert field in search_payload["summary"]
        assert field in product_payload["summary"]


def test_product_summary_separates_action_and_product_accuracy():
    results = [
        ProductEvalResult(
            case_id="one",
            category="recommend",
            latest_user="",
            product_summaries=[],
            expected_action="recommend",
            actual_action="recommend",
            expected_product_id="p1",
            actual_product_id="p2",
            passed=False,
            reason="",
            error=None,
        ),
        ProductEvalResult(
            case_id="two",
            category="wrong_track",
            latest_user="",
            product_summaries=[],
            expected_action="wrong_track",
            actual_action="wrong_track",
            expected_product_id=None,
            actual_product_id=None,
            passed=True,
            reason="",
            error=None,
        ),
    ]

    summary = build_product_summary(results)

    assert summary["accuracy"] == 0.5
    assert summary["action_accuracy"] == 1.0
    assert summary["product_id_accuracy_on_expected_recommendations"] == 0.0


def test_product_eval_records_case_error_and_continues():
    class FailingPolicy:
        def decide_next_action(self, **kwargs):
            raise RuntimeError("boom")

    from server.evals.product_response_decision.eval_product_response_decision import EvalCase

    case = EvalCase(
        case_id="bad_case",
        category="recommend",
        conversation_history=[],
        found_products={},
        source_knowledge="",
        expected_action="recommend",
        expected_product_id="p1",
        reason="expected recommendation",
    )

    results = run_eval(FailingPolicy(), [case])

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].actual_action == "ERROR: RuntimeError"
    assert results[0].reason == "expected recommendation"
    assert results[0].error == "RuntimeError: boom"
