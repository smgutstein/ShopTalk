import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recommender


def test_load_openai_api_key_requires_env_var(monkeypatch):
    monkeypatch.setattr(recommender, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        recommender.load_openai_api_key()


def test_load_openai_api_key_returns_existing_env_var(monkeypatch):
    monkeypatch.setattr(recommender, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    assert recommender.load_openai_api_key() == "test-api-key"


def test_missing_openai_key_fails_before_expensive_setup(monkeypatch):
    calls = []

    def fail_if_database_loads(self, *args, **kwargs):
        calls.append("database")
        raise AssertionError("Database loading should not run without an OpenAI key")

    def fail_if_imagebind_loads(self):
        calls.append("imagebind")
        raise AssertionError("ImageBind loading should not run without an OpenAI key")

    monkeypatch.setattr(recommender, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        recommender.ShopTalkRecommender,
        "_load_database",
        fail_if_database_loads,
    )
    monkeypatch.setattr(
        recommender.ShopTalkRecommender,
        "_load_imagebind_model",
        fail_if_imagebind_loads,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        recommender.ShopTalkRecommender(force_cpu=True)

    assert calls == []
