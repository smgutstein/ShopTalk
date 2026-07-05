from server.recommender_core.conversation_policy import ConversationPolicy
from server.recommender_core.product_candidate import ProductCandidate
from server.recommender_core.reply_types import (
    ProductSearchResult,
    RecommendationAction,
    SearchDecision,
)


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, recommendation_action):
        self.recommendation_action_model = FakeStructuredModel(recommendation_action)
        self.search_decision_model = FakeStructuredModel(
            SearchDecision(action="answer_without_search")
        )
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        if schema is RecommendationAction:
            return self.recommendation_action_model
        if schema is SearchDecision:
            return self.search_decision_model
        raise AssertionError(f"Unexpected structured output schema: {schema!r}")


def make_candidate(product_id):
    return ProductCandidate(
        product_id=product_id,
        item_name=f"Product {product_id}",
        score=0.9,
        image_paths=(),
        product_type="test product",
        llm_str=f"Details for {product_id}",
    )


def test_decide_next_response_recovers_from_recommended_product_outside_retrieved_set():
    chat_model = FakeChatModel(
        RecommendationAction(action="recommend", product_id="missing_pid")
    )
    policy = ConversationPolicy(chat_model)
    search_result = ProductSearchResult(
        search_performed=True,
        llm_search_query="test query",
        found_products={"pid_a": make_candidate("pid_a")},
        source_knowledge="product_id: pid_a",
    )

    decision = policy.decide_next_response(
        conversation_history=[],
        search_result=search_result,
    )

    assert decision.initial_llm_response == "<missing_pid>"
    assert decision.chosen_pid is None
    assert decision.chosen_product is None
    assert decision.dive_deeper is True


def test_decide_next_response_keeps_valid_recommended_product():
    chat_model = FakeChatModel(
        RecommendationAction(action="recommend", product_id="pid_a")
    )
    policy = ConversationPolicy(chat_model)
    product = make_candidate("pid_a")
    search_result = ProductSearchResult(
        search_performed=True,
        llm_search_query="test query",
        found_products={"pid_a": product},
        source_knowledge="product_id: pid_a",
    )

    decision = policy.decide_next_response(
        conversation_history=[],
        search_result=search_result,
    )

    assert decision.initial_llm_response == "<pid_a>"
    assert decision.chosen_pid == "pid_a"
    assert decision.chosen_product is product
    assert decision.dive_deeper is False
