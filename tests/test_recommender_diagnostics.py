import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recommender


def make_found_products():
    return {
        "pid-1": {
            "item_name": "Red Shoes",
            "score": 0.91,
            "product_type": "shoes",
            "image_paths": ["red.jpg"],
            "image_urls": ["/static/images/red.jpg"],
            "llm_str": "Description for Red Shoes",
        },
        "pid-2": {
            "item_name": "Blue Shoes",
            "score": 0.82,
            "product_type": "shoes",
            "image_paths": ["blue.jpg"],
            "image_urls": ["/static/images/blue.jpg"],
            "llm_str": "Description for Blue Shoes",
        },
    }


def test_determine_embedding_mode():
    assert recommender.determine_embedding_mode("red shoes", None) == "text"
    assert recommender.determine_embedding_mode(None, "query.jpg") == "image"
    assert recommender.determine_embedding_mode("red shoes", "query.jpg") == "text_image"

    with pytest.raises(ValueError, match="At least one query modality"):
        recommender.determine_embedding_mode(None, None)


def test_summarize_top_products_keeps_compact_retrieval_details():
    summary = recommender.summarize_top_products(make_found_products())

    assert summary == [
        {
            "product_id": "pid-1",
            "item_name": "Red Shoes",
            "score": 0.91,
            "product_type": "shoes",
            "image_paths": ["red.jpg"],
            "image_urls": ["/static/images/red.jpg"],
        },
        {
            "product_id": "pid-2",
            "item_name": "Blue Shoes",
            "score": 0.82,
            "product_type": "shoes",
            "image_paths": ["blue.jpg"],
            "image_urls": ["/static/images/blue.jpg"],
        },
    ]


def test_infer_recommendation_decision():
    assert recommender.infer_recommendation_decision("pid-1", False, "<pid-1>") == "recommend"
    assert recommender.infer_recommendation_decision(None, True, "<DIVE DEEPER>") == "dive_deeper"
    assert recommender.infer_recommendation_decision(None, False, "<WRONG TRACK>") == "wrong_track"
    assert recommender.infer_recommendation_decision(None, False, "nonsense") == "unknown"


def test_build_recommendation_diagnostics():
    diagnostics = recommender.build_recommendation_diagnostics(
        embedding_mode="text_image",
        llm_search_query="red running shoes",
        found_products=make_found_products(),
        initial_llm_response="<pid-1>",
        chosen_pid="pid-1",
        dive_deeper=False,
        total_seconds=1.25,
    )

    assert diagnostics["embedding_mode"] == "text_image"
    assert diagnostics["llm_search_query"] == "red running shoes"
    assert diagnostics["initial_llm_response"] == "<pid-1>"
    assert diagnostics["chosen_pid"] == "pid-1"
    assert diagnostics["decision"] == "recommend"
    assert diagnostics["timings"] == {"total_seconds": 1.25}
    assert diagnostics["top_products"][0]["product_id"] == "pid-1"
    assert diagnostics["top_products"][0]["score"] == 0.91
    assert diagnostics["top_products"][0]["image_paths"] == ["red.jpg"]
    assert diagnostics["top_products"][0]["image_urls"] == ["/static/images/red.jpg"]


def make_minimal_recommender(found_products, initial_llm_response="<pid-1>"):
    rec = object.__new__(recommender.ShopTalkRecommender)
    rec.debug = False
    rec.personality = "test-personality"
    rec.conversation_history = []

    rec._build_search_query = lambda: "red running shoes"
    rec._search_products = lambda search_query=None, top_k=10, image_path=None: found_products
    rec._choose_product_or_next_action = lambda found_products, source_knowledge: initial_llm_response
    rec._build_final_response = lambda **kwargs: "Final assistant response"
    return rec


def test_generate_reply_returns_diagnostics_for_text_query():
    found_products = make_found_products()
    rec = make_minimal_recommender(found_products)

    result = rec.generate_reply(user_input="I need red shoes")

    assert result["diagnostics"]["embedding_mode"] == "text"
    assert result["diagnostics"]["llm_search_query"] == "red running shoes"
    assert result["diagnostics"]["initial_llm_response"] == "<pid-1>"
    assert result["diagnostics"]["chosen_pid"] == "pid-1"
    assert result["diagnostics"]["decision"] == "recommend"
    assert result["diagnostics"]["top_products"][0]["product_id"] == "pid-1"
    assert result["diagnostics"]["timings"]["total_seconds"] >= 0


def test_generate_reply_returns_image_diagnostics_without_search_query():
    found_products = make_found_products()
    rec = make_minimal_recommender(found_products, initial_llm_response="<DIVE DEEPER>")

    result = rec.generate_reply(image_path="query.jpg")

    assert result["diagnostics"]["embedding_mode"] == "image"
    assert result["diagnostics"]["llm_search_query"] is None
    assert result["diagnostics"]["initial_llm_response"] == "<DIVE DEEPER>"
    assert result["diagnostics"]["chosen_pid"] is None
    assert result["diagnostics"]["decision"] == "dive_deeper"
