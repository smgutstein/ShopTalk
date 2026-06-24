import pytest
from pydantic import ValidationError

from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage

from server.recommender_core.conversation_policy import ConversationPolicy
from server.recommender_core.llm_prompts import build_search_decision_prompt
from server.recommender_core.reply_types import SearchDecision


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.response


class FakeChatModel:
    def __init__(self, search_response):
        self.search_structured_model = FakeStructuredModel(search_response)
        self.requested_schemas = []

    def with_structured_output(self, schema):
        self.requested_schemas.append(schema)
        if schema is SearchDecision:
            return self.search_structured_model
        return FakeStructuredModel(None)


def test_build_search_decision_prompt_requests_search_gate_and_compact_query():
    prompt = build_search_decision_prompt()

    assert "requires a new product search" in prompt
    assert "action='search'" in prompt
    assert "action='answer_without_search'" in prompt
    assert "at most 10 words" in prompt
    assert "product-relevant terms only" in prompt


def test_search_decision_requires_query_when_searching():
    with pytest.raises(ValidationError, match="search_query is required"):
        SearchDecision(action="search", search_query=None)


def test_search_decision_rejects_query_without_search():
    with pytest.raises(ValidationError, match="search_query must be null"):
        SearchDecision(action="answer_without_search", search_query="red shoes")


def test_decide_search_action_appends_prompt_to_conversation_history():
    fake_chat = FakeChatModel(
        SearchDecision(action="search", search_query="waterproof hiking boots")
    )
    policy = ConversationPolicy(fake_chat)
    conversation_history = [
        SystemMessage(content="system instructions"),
        AIMessage(content="What would you like to shop for today?"),
        HumanMessage(content="I need boots for muddy trails."),
    ]

    search_decision = policy.decide_search_action(conversation_history)

    assert search_decision == SearchDecision(
        action="search",
        search_query="waterproof hiking boots",
    )
    assert len(fake_chat.search_structured_model.invocations) == 1
    invoked_messages = fake_chat.search_structured_model.invocations[0]
    assert invoked_messages[:-1] == conversation_history
    assert isinstance(invoked_messages[-1], SystemMessage)
    assert invoked_messages[-1].content == build_search_decision_prompt()
