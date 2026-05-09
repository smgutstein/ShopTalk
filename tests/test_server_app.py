"""Tests for the Flask app wrapper in server.py.

These tests deliberately use a fake recommender so importing/constructing the
web app does not load ImageBind, OpenAI, or vector-database artifacts.
"""

import importlib.util
import sys
from pathlib import Path


class FakeRecommender:
    def __init__(self):
        self.conversation_history = []
        self.personality = "test-personality"
        self.calls = []

    def generate_reply(self, user_input):
        self.calls.append(user_input)
        return {
            "conversation": [
                {"type": "HumanMessage", "content": user_input},
                {"type": "AIMessage", "content": "fake response"},
            ],
            "chosen_product": {"item_name": "Fake Product"},
            "personality": self.personality,
        }


def load_server_module():
    """Load server/server.py without requiring server/ to be a package."""
    repo_root = Path(__file__).resolve().parents[1]
    server_dir = repo_root / "server"
    server_path = server_dir / "server.py"

    # server.py imports recommender as a top-level module, so make the server
    # directory importable for this test without changing the application code.
    sys.path.insert(0, str(server_dir))

    spec = importlib.util.spec_from_file_location("shoptalk_server", server_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_route_calls_recommender_and_returns_json():
    server_module = load_server_module()
    fake_recommender = FakeRecommender()
    app = server_module.create_app(fake_recommender)

    client = app.test_client()
    response = client.post("/generate", json={"user_input": "I need a backpack"})

    assert response.status_code == 200
    assert fake_recommender.calls == ["I need a backpack"]
