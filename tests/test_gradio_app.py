import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gradio_app


class FakeRecommender:
    def __init__(self):
        self.calls = []

    def generate_reply(self, user_input=None, image_path=None):
        self.calls.append({"user_input": user_input, "image_path": image_path})
        return {
            "conversation": [
                {"type": "HumanMessage", "content": user_input or "image query"},
                {"type": "AIMessage", "content": "Here is a recommendation."},
            ],
            "chosen_product": {
                "item_name": "Test Product",
                "score": 0.875,
                "product_type": "widget",
                "image_paths": ["test.jpg"],
            },
            "personality": "test",
            "diagnostics": {
                "embedding_mode": "text_image",
                "decision": "recommend",
                "llm_search_query": "red shoes",
                "chosen_pid": "product-1",
                "initial_llm_response": "<product-1>",
                "top_products": [
                    {"product_id": "product-1", "item_name": "Test Product", "score": 0.875},
                    {"product_id": "product-2", "item_name": "Backup Product", "score": 0.5},
                ],
                "timings": {"total_seconds": 1.23456},
            },
        }


def test_handle_message_passes_text_query_to_recommender():
    fake = FakeRecommender()

    history, cleared_text, cleared_image, product_md, diagnostics_summary, diagnostics = gradio_app.handle_message(
        " red shoes ",
        None,
        [],
        fake,
    )

    assert fake.calls == [{"user_input": "red shoes", "image_path": None}]
    assert history == [("red shoes", "Here is a recommendation.")]
    assert cleared_text == ""
    assert cleared_image is None
    assert "Test Product" in product_md
    assert "0.8750" in product_md
    assert "text_image" in diagnostics_summary
    assert "recommend" in diagnostics_summary
    assert "red shoes" in diagnostics_summary
    assert "Test Product" in diagnostics_summary
    assert '"embedding_mode": "text_image"' in diagnostics


def test_handle_message_passes_image_only_query_to_recommender():
    fake = FakeRecommender()

    history, _, _, product_md, _, _ = gradio_app.handle_message(
        "",
        "/tmp/query.jpg",
        [],
        fake,
    )

    assert fake.calls == [{"user_input": None, "image_path": "/tmp/query.jpg"}]
    assert history == [("[image uploaded]", "Here is a recommendation.")]
    assert "Test Product" in product_md


def test_handle_message_passes_text_and_image_query_to_recommender():
    fake = FakeRecommender()

    history, _, _, _, _, _ = gradio_app.handle_message(
        "match this style",
        "/tmp/query.jpg",
        [],
        fake,
    )

    assert fake.calls == [{"user_input": "match this style", "image_path": "/tmp/query.jpg"}]
    assert history == [("match this style\n\n[image uploaded]", "Here is a recommendation.")]


def test_handle_message_rejects_empty_submission_without_calling_recommender():
    fake = FakeRecommender()

    history, cleared_text, cleared_image, product_md, diagnostics_summary, diagnostics = gradio_app.handle_message(
        "   ",
        None,
        [("previous", "reply")],
        fake,
    )

    assert fake.calls == []
    assert history == [("previous", "reply")]
    assert cleared_text == ""
    assert cleared_image is None
    assert "Enter text" in product_md
    assert diagnostics_summary == "No diagnostics reported yet."
    assert diagnostics == "{}"


def test_format_chosen_product_handles_empty_product():
    assert gradio_app.format_chosen_product({}) == "No product selected yet."


def test_latest_ai_message_returns_last_ai_content():
    conversation = [
        {"type": "AIMessage", "content": "first"},
        {"type": "HumanMessage", "content": "user"},
        {"type": "AIMessage", "content": "second"},
    ]

    assert gradio_app.latest_ai_message(conversation) == "second"


def test_format_diagnostics_summary_handles_empty_result():
    assert gradio_app.format_diagnostics_summary({}) == "No diagnostics reported yet."


def test_format_top_products_handles_scores_and_missing_scores():
    markdown = gradio_app.format_top_products([
        {"product_id": "p1", "item_name": "One", "score": 0.12345},
        {"product_id": "p2", "item_name": "Two"},
    ])

    assert "`p1`" in markdown
    assert "One" in markdown
    assert "0.1235" in markdown
    assert "`p2`" in markdown
    assert "Two" in markdown
