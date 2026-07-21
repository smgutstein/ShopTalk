import json
from pathlib import Path

import pytest
from langchain_classic.schema import AIMessage, HumanMessage

from server.evals.product_response_decision import eval_product_response_decision as product_eval
from server.evals.search_decision import eval_search_decision as search_eval


def test_search_case_uses_common_envelope(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "search_001",
                "category": "boundary",
                "messages": [
                    {"role": "user", "content": "Find walking shoes."},
                    {"role": "assistant", "content": "Any color preference?"},
                    {"role": "user", "content": "Blue."},
                ],
                "expected": {"action": "search"},
                "reason": "New color constraint.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    case = search_eval.load_cases(path)[0]

    assert case.case_id == "search_001"
    assert case.expected_action == "search"
    assert isinstance(case.conversation_history[0], HumanMessage)
    assert isinstance(case.conversation_history[1], AIMessage)
    assert search_eval.latest_user_message(case.conversation_history) == "Blue."


def test_search_case_requires_expected_action(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"case_id": "search_001", "messages": [], "expected": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing expected.action"):
        search_eval.load_cases(path)


def test_search_role_parsing_remains_restricted():
    with pytest.raises(ValueError, match="Unsupported message role"):
        search_eval.message_from_dict({"role": "human", "content": "Hello"})


@pytest.mark.parametrize(
    ("loader", "message"),
    [
        (search_eval.load_cases, "Search-decision case file must be JSONL"),
        (product_eval.load_cases, "Product-response case file must be JSONL"),
    ],
)
def test_evaluators_reject_non_jsonl_files(tmp_path: Path, loader, message: str):
    path = tmp_path / "cases.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        loader(path)
