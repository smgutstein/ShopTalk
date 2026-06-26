from types import SimpleNamespace

import pytest

from server.recommender_core.config import RecommenderConfig
from server.recommender_core.shop_talk_recommender import ShopTalkRecommender
from server.recommender_core import recommender_factory


def test_recommender_config_from_args_maps_cli_fields():
    args = SimpleNamespace(
        personality=4,
        debug=True,
        cpu=True,
        model="gpt-test",
        vector_db_output_dir="artifacts/test_vector_db",
        vector_backend="faiss",
        top_k=7,
        product_blurbs="tests/fixtures/product_blurbs.json",
        images_csv="tests/fixtures/images.csv",
    )

    config = RecommenderConfig.from_args(args)

    assert config.personality_index == 4
    assert config.debug is True
    assert config.force_cpu is True
    assert config.model_name == "gpt-test"
    assert str(config.vector_db_output_dir) == "artifacts/test_vector_db"
    assert config.vector_backend == "faiss"
    assert config.top_k == 7
    assert str(config.blurbs_path) == "tests/fixtures/product_blurbs.json"
    assert str(config.images_csv_path) == "tests/fixtures/images.csv"


def test_shoptalk_recommender_from_args_delegates_to_factory(monkeypatch):
    calls = []
    fake_recommender = object()

    def fake_build_recommender(config):
        calls.append(config)
        return fake_recommender

    monkeypatch.setattr(
        "server.recommender_core.recommender_factory.build_recommender",
        fake_build_recommender,
    )

    args = SimpleNamespace(
        personality=2,
        debug=False,
        cpu=False,
        model="gpt-test",
        vector_db_output_dir="artifacts/test_vector_db",
        vector_backend="faiss",
        top_k=5,
        product_blurbs="tests/fixtures/product_blurbs.json",
        images_csv="tests/fixtures/images.csv",
    )

    result = ShopTalkRecommender.from_args(args)

    assert result is fake_recommender
    assert len(calls) == 1
    assert isinstance(calls[0], RecommenderConfig)
    assert calls[0].personality_index == 2
    assert calls[0].top_k == 5


def test_choose_personality_rejects_empty_personality_list(monkeypatch):
    monkeypatch.setattr(recommender_factory, "PERSONALITIES", [])

    with pytest.raises(ValueError, match="at least one personality"):
        recommender_factory.choose_personality(-1)


def test_choose_personality_rejects_blank_resolved_personality(monkeypatch):
    monkeypatch.setattr(recommender_factory, "PERSONALITIES", ["   "])

    with pytest.raises(ValueError, match="non-empty string"):
        recommender_factory.choose_personality(0)


def test_choose_personality_strips_resolved_personality(monkeypatch):
    monkeypatch.setattr(recommender_factory, "PERSONALITIES", ["  pirate  "])

    assert recommender_factory.choose_personality(0) == "pirate"
