import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recommender
from langchain_classic.schema import HumanMessage


class FakeQueryEmbedder:
    def __init__(self):
        self.text_calls = []
        self.image_calls = []

    def embed_query(self, text):
        self.text_calls.append(text)
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_image(self, image_path):
        self.image_calls.append(image_path)
        return np.array([0.0, 1.0], dtype=np.float32)


class FakeProductStore:
    def __init__(self):
        self.calls = []
        self.result = {
            "PID": {
                "item_name": "Fake product",
                "score": 0.75,
            }
        }

    def search(self, embedded_query, top_k, image_id_to_path):
        self.calls.append(
            {
                "embedded_query": embedded_query,
                "top_k": top_k,
                "image_id_to_path": image_id_to_path,
            }
        )
        return self.result


def make_recommender_without_init():
    instance = object.__new__(recommender.ShopTalkRecommender)
    instance.query_embedder = FakeQueryEmbedder()
    instance.product_store = FakeProductStore()
    instance.image_id_to_path = {"img-1": "image-one.jpg"}
    return instance


def test_combine_query_embeddings_averages_and_normalizes_vectors():
    combined = recommender.combine_query_embeddings(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )

    assert np.allclose(combined, [2 ** -0.5, 2 ** -0.5])


def test_combine_query_embeddings_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one query embedding"):
        recommender.combine_query_embeddings([])


def test_search_products_uses_text_embedding_for_text_only_query():
    instance = make_recommender_without_init()

    found_products = instance._search_products("red shoes", top_k=3)

    assert found_products == instance.product_store.result
    assert instance.query_embedder.text_calls == ["red shoes"]
    assert instance.query_embedder.image_calls == []
    assert len(instance.product_store.calls) == 1
    call = instance.product_store.calls[0]
    assert np.allclose(call["embedded_query"], [1.0, 0.0])
    assert call["top_k"] == 3
    assert call["image_id_to_path"] == {"img-1": "image-one.jpg"}


def test_search_products_uses_image_embedding_for_image_only_query():
    instance = make_recommender_without_init()

    instance._search_products(image_path="query.jpg", top_k=4)

    assert instance.query_embedder.text_calls == []
    assert instance.query_embedder.image_calls == ["query.jpg"]
    call = instance.product_store.calls[0]
    assert np.allclose(call["embedded_query"], [0.0, 1.0])
    assert call["top_k"] == 4


def test_search_products_combines_text_and_image_embeddings():
    instance = make_recommender_without_init()

    instance._search_products("red shoes", image_path="query.jpg", top_k=5)

    assert instance.query_embedder.text_calls == ["red shoes"]
    assert instance.query_embedder.image_calls == ["query.jpg"]
    call = instance.product_store.calls[0]
    assert np.allclose(call["embedded_query"], [2 ** -0.5, 2 ** -0.5])
    assert call["top_k"] == 5


def test_generate_reply_accepts_image_only_query_without_building_text_search_query():
    instance = object.__new__(recommender.ShopTalkRecommender)
    instance.debug = False
    instance.personality = "test personality"
    instance.conversation_history = []
    calls = []

    def fail_if_text_search_query_is_built():
        raise AssertionError("Image-only search should not ask the LLM for text search terms")

    def fake_search_products(search_query=None, top_k=10, image_path=None):
        calls.append((search_query, top_k, image_path))
        return {"PID": {"item_name": "Fake product", "score": 0.75}}

    instance._build_search_query = fail_if_text_search_query_is_built
    instance._search_products = fake_search_products
    instance._format_source_knowledge = lambda found_products: "source knowledge"
    instance._choose_product_or_next_action = lambda found_products, source_knowledge: "<DIVE DEEPER>"
    instance._parse_product_choice = lambda llm_response, found_products: (None, {}, True)
    instance._build_final_response = lambda **kwargs: "final response"

    result = instance.generate_reply(image_path="query.jpg")

    assert calls == [(None, 10, "query.jpg")]
    assert isinstance(instance.conversation_history[0], HumanMessage)
    assert "uploaded an image" in instance.conversation_history[0].content
    assert result["chosen_product"] == {}
    assert result["personality"] == "test personality"
