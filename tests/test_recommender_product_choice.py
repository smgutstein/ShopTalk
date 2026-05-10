import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recommender


def test_parse_product_choice_selects_known_product_id():
    product = {"item_name": "Known product"}

    chosen_pid, chosen_product, dive_deeper = recommender.parse_product_choice(
        "<B123>",
        {"B123": product},
    )

    assert chosen_pid == "B123"
    assert chosen_product is product
    assert dive_deeper is False


def test_parse_product_choice_allows_extra_text_around_product_id():
    product = {"item_name": "Known product"}

    chosen_pid, chosen_product, dive_deeper = recommender.parse_product_choice(
        "I choose <B123> for this shopper.",
        {"B123": product},
    )

    assert chosen_pid == "B123"
    assert chosen_product is product
    assert dive_deeper is False


def test_parse_product_choice_unknown_product_id_returns_no_product():
    chosen_pid, chosen_product, dive_deeper = recommender.parse_product_choice(
        "<UNKNOWN>",
        {"B123": {"item_name": "Known product"}},
    )

    assert chosen_pid is None
    assert chosen_product == {}
    assert dive_deeper is False


def test_parse_product_choice_dive_deeper_returns_no_product():
    chosen_pid, chosen_product, dive_deeper = recommender.parse_product_choice(
        "<DIVE DEEPER>",
        {"B123": {"item_name": "Known product"}},
    )

    assert chosen_pid is None
    assert chosen_product == {}
    assert dive_deeper is True


def test_parse_product_choice_wrong_track_returns_no_product():
    chosen_pid, chosen_product, dive_deeper = recommender.parse_product_choice(
        "<WRONG TRACK>",
        {"B123": {"item_name": "Known product"}},
    )

    assert chosen_pid is None
    assert chosen_product == {}
    assert dive_deeper is False


def test_parse_product_choice_garbage_response_returns_no_product():
    chosen_pid, chosen_product, dive_deeper = recommender.parse_product_choice(
        "this is not a control response",
        {"B123": {"item_name": "Known product"}},
    )

    assert chosen_pid is None
    assert chosen_product == {}
    assert dive_deeper is False


def test_extract_bracketed_choice_strips_whitespace():
    assert recommender.extract_bracketed_choice("<  B123  >") == "B123"


def test_extract_bracketed_choice_rejects_empty_choice():
    assert recommender.extract_bracketed_choice("<   >") is None


def test_recommender_parse_product_choice_wrapper_delegates_to_helper(monkeypatch):
    calls = []

    def fake_parse_product_choice(llm_response, found_products):
        calls.append((llm_response, found_products))
        return "PID", {"item_name": "Wrapped"}, False

    monkeypatch.setattr(recommender, "parse_product_choice", fake_parse_product_choice)
    instance = object.__new__(recommender.ShopTalkRecommender)

    assert instance._parse_product_choice("<PID>", {"PID": {}}) == (
        "PID",
        {"item_name": "Wrapped"},
        False,
    )
    assert calls == [("<PID>", {"PID": {}})]


def test_recommender_extract_bracketed_choice_wrapper_delegates_to_helper(monkeypatch):
    monkeypatch.setattr(recommender, "extract_bracketed_choice", lambda response: "CHOICE")

    assert recommender.ShopTalkRecommender._extract_bracketed_choice("anything") == "CHOICE"
