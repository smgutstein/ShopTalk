import numpy as np
import pytest

from langchain_classic.schema import HumanMessage

from server.recommender_core.product_candidate import ProductCandidate
from server.recommender_core.reply_types import RecommendationDecision, SearchDecision
from server.recommender_core.shop_talk_recommender import ShopTalkRecommender
from server.recommender_core.vector_query import combine_query_embeddings


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
            "PID": ProductCandidate(
                product_id="PID",
                item_name="Fake product",
                score=0.75,
                image_paths=(),
                product_type="widget",
                llm_str="Fake product details",
            )
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
    instance = object.__new__(ShopTalkRecommender)
    instance.query_embedder = FakeQueryEmbedder()
    instance.product_store = FakeProductStore()
    instance.image_id_to_path = {"img-1": "image-one.jpg"}
    return instance


def test_combine_query_embeddings_averages_and_normalizes_vectors():
    combined = combine_query_embeddings(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )

    assert np.allclose(combined, [2 ** -0.5, 2 ** -0.5])


def test_combine_query_embeddings_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one query embedding"):
        combine_query_embeddings([])


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


def test_generate_reply_accepts_image_only_query_without_search_decision_llm():
    instance = object.__new__(ShopTalkRecommender)
    instance.debug = False
    instance.personality = "test personality"
    instance.conversation_history = []
    instance.top_k = 10
    calls = []

    class FakeConversationPolicy:
        def decide_search_action(self, conversation_history):
            raise AssertionError("Image-only search should not ask the LLM whether to search")

        def decide_next_response(self, conversation_history, search_result):
            return RecommendationDecision(
                initial_llm_response="<DIVE DEEPER>",
                chosen_pid=None,
                chosen_product=None,
                dive_deeper=True,
            )

        def generate_final_response(self, conversation_history, decision, source_knowledge):
            return "final response"

        def generate_response_without_search(self, conversation_history):
            raise AssertionError("Image-only input should perform product search")

    def fake_search_products(search_query=None, top_k=10, image_path=None):
        calls.append((search_query, top_k, image_path))
        return {
            "PID": ProductCandidate(
                product_id="PID",
                item_name="Fake product",
                score=0.75,
                image_paths=(),
                product_type="widget",
                llm_str="Fake product details",
            )
        }

    instance.conversation_policy = FakeConversationPolicy()
    instance._search_products = fake_search_products

    result = instance.generate_reply(image_path="query.jpg")

    from pathlib import Path
    assert calls == [(None, 10, Path("query.jpg"))]
    assert isinstance(instance.conversation_history[0], HumanMessage)
    assert "uploaded an image" in instance.conversation_history[0].content
    assert result["chosen_product"] is None
    assert result["personality"] == "test personality"
    assert result["diagnostics"].search_performed is True
    assert result["diagnostics"].embedding_mode == "image"


def test_text_query_can_skip_product_search_when_policy_says_no_search():
    instance = object.__new__(ShopTalkRecommender)
    instance.debug = False
    instance.personality = "test personality"
    instance.conversation_history = []
    instance.top_k = 10

    class FakeConversationPolicy:
        def decide_search_action(self, conversation_history):
            return SearchDecision(action="answer_without_search", search_query=None)

        def generate_response_without_search(self, conversation_history):
            return "no-search response"

    instance.conversation_policy = FakeConversationPolicy()
    instance._search_products = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("No-search decision should not query product store")
    )

    result = instance.generate_reply(user_input="Thanks")

    assert result["diagnostics"].search_performed is False
    assert result["diagnostics"].llm_search_query is None
    assert result["diagnostics"].initial_llm_response == "<NO SEARCH>"
    assert result["diagnostics"].decision == "unknown"
