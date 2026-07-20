from types import SimpleNamespace

import pytest

from server.recommender_core.config import RecommenderConfig
from server.recommender_core.shop_talk_recommender import ShopTalkRecommender
from server.recommender_core import recommender_factory


def test_shoptalk_recommender_from_args_delegates_to_factory(monkeypatch, tmp_path):
    calls = []
    fake_recommender = object()
    config_path = tmp_path / "shoptalk_config.ini"
    config_path.write_text(
        "[app]\npersonality = 2\n"
        "[retrieval]\ntop_k = 5\n",
        encoding="utf-8",
    )

    def fake_build_recommender(config):
        calls.append(config)
        return fake_recommender

    monkeypatch.setattr(
        "server.recommender_core.recommender_factory.build_recommender",
        fake_build_recommender,
    )

    args = SimpleNamespace(config=config_path, debug=False, cpu=False)
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
