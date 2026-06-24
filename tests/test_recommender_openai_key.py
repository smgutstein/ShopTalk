import pytest

from server.recommender_core import recommender_factory
from server.recommender_core.config import RecommenderConfig
from server.recommender_core.utils import load_openai_api_key


def test_load_openai_api_key_requires_env_var(monkeypatch):
    monkeypatch.setattr("server.recommender_core.utils.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_openai_api_key()


def test_load_openai_api_key_returns_existing_env_var(monkeypatch):
    monkeypatch.setattr("server.recommender_core.utils.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    assert load_openai_api_key() == "test-api-key"


def test_missing_openai_key_fails_before_expensive_setup(monkeypatch):
    calls = []

    def fake_choose_device(config):
        calls.append("device")
        return "cpu"

    def missing_key():
        calls.append("api-key")
        raise ValueError("OPENAI_API_KEY missing")

    def fail_if_database_loads(config):
        calls.append("database")
        raise AssertionError("Database loading should not run without an OpenAI key")

    def fail_if_imagebind_loads(device):
        calls.append("imagebind")
        raise AssertionError("ImageBind loading should not run without an OpenAI key")

    monkeypatch.setattr(recommender_factory, "choose_device", fake_choose_device)
    monkeypatch.setattr(recommender_factory, "load_openai_api_key", missing_key)
    monkeypatch.setattr(recommender_factory, "load_product_store", fail_if_database_loads)
    monkeypatch.setattr(recommender_factory, "build_query_embedder", fail_if_imagebind_loads)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        recommender_factory.build_recommender(RecommenderConfig())

    assert calls == ["device", "api-key"]
