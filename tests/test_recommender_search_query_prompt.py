import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recommender
from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage


class FakeChatOpenAI:
    def __init__(self, response_text):
        self.response_text = response_text
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)

        class Response:
            content = self.response_text

        return Response()


def make_recommender_without_init():
    return object.__new__(recommender.ShopTalkRecommender)


def test_build_search_query_prompt_requests_compact_product_search_terms():
    prompt = recommender.build_search_query_prompt()

    assert "what sort of product should we search for" in prompt
    assert "maximum of 10 words" in prompt
    assert "automated search system" in prompt
    assert "ignore your personality" in prompt


def test_build_search_query_appends_prompt_to_conversation_history():
    instance = make_recommender_without_init()
    fake_chat = FakeChatOpenAI("waterproof hiking boots")
    instance.chat_openai = fake_chat
    instance.conversation_history = [
        SystemMessage(content="system instructions"),
        AIMessage(content="What would you like to shop for today?"),
        HumanMessage(content="I need boots for muddy trails."),
    ]

    search_query = instance._build_search_query()

    assert search_query == "waterproof hiking boots"
    assert len(fake_chat.invocations) == 1
    invoked_messages = fake_chat.invocations[0]
    assert invoked_messages[:-1] == instance.conversation_history
    assert isinstance(invoked_messages[-1], SystemMessage)
    assert invoked_messages[-1].content == recommender.build_search_query_prompt()
