import json

import pytest

from server.evals.retrieval_llm_response.retrieval_llm_eval import load_jsonl_cases


BASE_CASE = {
    "case_id": "text_001",
    "query_type": "text_only",
    "category": "office_furniture",
    "query": "I need a gray office chair.",
    "image_path": None,
    "target_product_id": "B07Q44L76B",
    "target_title": "Gray office chair",
    "expected_available": True,
    "requires_image": True,
}


@pytest.mark.parametrize(
    "prefix",
    [
        "valid-match case.",
        "difficult/ambiguous match case.",
        "no appropriate product case.",
    ],
)
def test_load_jsonl_cases_accepts_required_case_note_prefixes(tmp_path, prefix):
    case = {**BASE_CASE, "notes": f"{prefix} Additional explanation."}
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    loaded = load_jsonl_cases(cases_path)

    assert loaded[0].notes == case["notes"]


def test_load_jsonl_cases_rejects_missing_case_note_prefix(tmp_path):
    case = {**BASE_CASE, "notes": "The target is a clear match."}
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="notes must begin with one of these exact prefixes"):
        load_jsonl_cases(cases_path)
