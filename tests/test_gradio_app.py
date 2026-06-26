from pathlib import Path

from server import gradio_app
from server.recommender_core.product_candidate import ProductCandidate


class FakeRecommender:
    def __init__(self, image_paths=None):
        self.calls = []
        self.image_paths = tuple(image_paths or [])

    def generate_reply(self, user_input=None, image_path=None):
        self.calls.append({"user_input": user_input, "image_path": image_path})
        return {
            "conversation": [
                {"type": "HumanMessage", "content": user_input or "image query"},
                {"type": "AIMessage", "content": "Here is a recommendation."},
            ],
            "chosen_product": ProductCandidate(
                product_id="product-1",
                item_name="Test Product",
                score=0.875,
                product_type="widget",
                image_paths=self.image_paths,
                llm_str="Test product details",
            ),
            "personality": "test",
            "diagnostics": {
                "embedding_mode": "text_image",
                "search_performed": True,
                "decision": "recommend",
                "llm_search_query": "red shoes",
                "chosen_pid": "product-1",
                "initial_llm_response": "<product-1>",
                "top_products": [
                    {
                        "product_id": "product-1",
                        "item_name": "Test Product",
                        "score": 0.875,
                        "image_paths": list(self.image_paths),
                    },
                    {
                        "product_id": "product-2",
                        "item_name": "Backup Product",
                        "score": 0.5,
                        "image_paths": list(reversed(self.image_paths)),
                    },
                ],
                "timings": {"total_seconds": 1.23456},
            },
        }


def user_msg(content):
    return {"role": "user", "content": content}


def assistant_msg(content):
    return {"role": "assistant", "content": content}


def test_handle_message_passes_text_query_to_recommender(tmp_path):
    image1 = tmp_path / "test.jpg"
    image2 = tmp_path / "detail.jpg"
    image1.write_text("fake image", encoding="utf-8")
    image2.write_text("fake image", encoding="utf-8")
    fake = FakeRecommender([str(image1), str(image2)])

    history, cleared_text, cleared_image, product_md, product_images, top_product_images, diagnostics_summary, diagnostics = gradio_app.handle_message(
        " red shoes ",
        None,
        [],
        fake,
    )

    assert fake.calls == [{"user_input": "red shoes", "image_path": None}]
    assert history == [user_msg("red shoes"), assistant_msg("Here is a recommendation.")]
    assert cleared_text == ""
    assert cleared_image is None
    assert "Test Product" in product_md
    assert "0.8750" in product_md
    assert product_images == [str(image1), str(image2)]
    assert top_product_images == [str(image1), str(image2)]
    assert "text_image" in diagnostics_summary
    assert "recommend" in diagnostics_summary
    assert "red shoes" in diagnostics_summary
    assert "Test Product" in diagnostics_summary
    assert '"embedding_mode": "text_image"' in diagnostics


def test_handle_message_passes_image_only_query_to_recommender(tmp_path):
    image1 = tmp_path / "test.jpg"
    image2 = tmp_path / "detail.jpg"
    image1.write_text("fake image", encoding="utf-8")
    image2.write_text("fake image", encoding="utf-8")
    fake = FakeRecommender([str(image1), str(image2)])

    history, _, _, product_md, product_images, top_product_images, _, _ = gradio_app.handle_message(
        "",
        "/tmp/query.jpg",
        [],
        fake,
    )

    assert fake.calls == [{"user_input": None, "image_path": "/tmp/query.jpg"}]
    assert history == [user_msg("[image uploaded]"), assistant_msg("Here is a recommendation.")]
    assert "Test Product" in product_md
    assert product_images == [str(image1), str(image2)]
    assert top_product_images == [str(image1), str(image2)]


def test_handle_message_passes_text_and_image_query_to_recommender(tmp_path):
    image1 = tmp_path / "test.jpg"
    image2 = tmp_path / "detail.jpg"
    image1.write_text("fake image", encoding="utf-8")
    image2.write_text("fake image", encoding="utf-8")
    fake = FakeRecommender([str(image1), str(image2)])

    history, _, _, _, _, _, _, _ = gradio_app.handle_message(
        "match this style",
        "/tmp/query.jpg",
        [],
        fake,
    )

    assert fake.calls == [{"user_input": "match this style", "image_path": "/tmp/query.jpg"}]
    assert history == [
        user_msg("match this style\n\n[image uploaded]"),
        assistant_msg("Here is a recommendation."),
    ]


def test_handle_message_rejects_empty_submission_without_calling_recommender():
    fake = FakeRecommender()
    existing_history = [user_msg("previous"), assistant_msg("reply")]

    history, cleared_text, cleared_image, product_md, product_images, top_product_images, diagnostics_summary, diagnostics = gradio_app.handle_message(
        "   ",
        None,
        existing_history,
        fake,
    )

    assert fake.calls == []
    assert history == existing_history
    assert cleared_text == ""
    assert cleared_image is None
    assert "Enter text" in product_md
    assert product_images == []
    assert top_product_images == []
    assert diagnostics_summary == "No diagnostics reported yet."
    assert diagnostics == "{}"


def test_format_chosen_product_handles_empty_product():
    assert gradio_app.format_chosen_product({}) == "No product selected yet."


def test_chosen_product_image_paths_returns_local_image_paths(tmp_path):
    one = tmp_path / "one.jpg"
    two = tmp_path / "two.jpg"
    one.write_text("fake", encoding="utf-8")
    two.write_text("fake", encoding="utf-8")
    product = ProductCandidate(
        product_id="p1",
        item_name="One",
        score=1.0,
        image_paths=(str(one), str(two)),
        product_type="widget",
        llm_str="details",
    )

    assert gradio_app.chosen_product_image_paths(product) == [str(one), str(two)]
    assert gradio_app.chosen_product_image_paths({}) == []


def test_top_product_image_paths_returns_unique_retrieval_images_in_order(tmp_path):
    one = tmp_path / "one.jpg"
    shared = tmp_path / "shared.jpg"
    two = tmp_path / "two.jpg"
    for path in (one, shared, two):
        path.write_text("fake", encoding="utf-8")

    result = {
        "diagnostics": {
            "top_products": [
                {"image_paths": [str(one), str(shared)]},
                {"image_paths": [str(two), str(shared)]},
            ]
        }
    }

    assert gradio_app.top_product_image_paths(result) == [str(one), str(shared), str(two)]
    assert gradio_app.top_product_image_paths(result, max_images=2) == [str(one), str(shared)]
    assert gradio_app.top_product_image_paths({}) == []


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


def test_normalize_image_input_accepts_common_gradio_shapes():
    assert gradio_app.normalize_image_input(None) is None
    assert gradio_app.normalize_image_input(" /tmp/query.jpg ") == "/tmp/query.jpg"
    assert gradio_app.normalize_image_input(Path("/tmp/query.jpg")) == "/tmp/query.jpg"
    assert gradio_app.normalize_image_input({"path": "/tmp/query.jpg"}) == "/tmp/query.jpg"
    assert gradio_app.normalize_image_input({"name": "/tmp/from-name.jpg"}) == "/tmp/from-name.jpg"


def test_normalize_image_input_rejects_unsupported_shape():
    try:
        gradio_app.normalize_image_input(["/tmp/query.jpg"])
    except TypeError as exc:
        assert "Unsupported Gradio image input type" in str(exc)
    else:
        raise AssertionError("Expected TypeError for unsupported image input shape")


def test_handle_message_normalizes_image_metadata_dict_before_calling_recommender(tmp_path):
    image1 = tmp_path / "test.jpg"
    image2 = tmp_path / "detail.jpg"
    image1.write_text("fake image", encoding="utf-8")
    image2.write_text("fake image", encoding="utf-8")
    fake = FakeRecommender([str(image1), str(image2)])

    history, _, _, _, _, _, _, _ = gradio_app.handle_message(
        "match this",
        {"path": "/tmp/query.jpg"},
        [],
        fake,
    )

    assert fake.calls == [{"user_input": "match this", "image_path": "/tmp/query.jpg"}]
    assert history == [
        user_msg("match this\n\n[image uploaded]"),
        assistant_msg("Here is a recommendation."),
    ]


def test_initial_assistant_greeting_uses_recommender_personality():
    fake = FakeRecommender()
    fake.personality = "1920s gangster"

    greeting = gradio_app.initial_assistant_greeting(fake)

    assert greeting == (
        "I'm your 1920s gangster shopping assistant. "
        "What would you like to shop for today?"
    )


def test_initial_assistant_greeting_rejects_missing_personality():
    fake = FakeRecommender()

    try:
        gradio_app.initial_assistant_greeting(fake)
    except AttributeError:
        pass
    else:
        raise AssertionError("Expected missing personality to fail loudly.")


def test_initial_assistant_greeting_rejects_blank_personality():
    fake = FakeRecommender()
    fake.personality = "   "

    try:
        gradio_app.initial_assistant_greeting(fake)
    except ValueError as exc:
        assert "non-empty personality" in str(exc)
    else:
        raise AssertionError("Expected blank personality to fail loudly.")


def test_initial_chat_history_seeds_assistant_greeting():
    fake = FakeRecommender()
    fake.personality = "test personality"

    assert gradio_app.initial_chat_history(fake) == [
        {
            "role": "assistant",
            "content": (
                "I'm your test personality shopping assistant. "
                "What would you like to shop for today?"
            ),
        }
    ]
