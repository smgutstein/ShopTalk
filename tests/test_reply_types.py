import pytest
from pydantic import ValidationError

from server.recommender_core.reply_types import RecommendationAction


def test_recommendation_action_requires_product_id_for_recommend():
    with pytest.raises(ValidationError, match="product_id is required"):
        RecommendationAction(action="recommend")


def test_recommendation_action_accepts_product_id_for_recommend():
    action = RecommendationAction(action="recommend", product_id="B001")

    assert action.action == "recommend"
    assert action.product_id == "B001"


@pytest.mark.parametrize("action_name", ["dive_deeper", "wrong_track"])
def test_recommendation_action_rejects_product_id_for_non_recommend_actions(action_name):
    with pytest.raises(ValidationError, match="product_id must be null"):
        RecommendationAction(action=action_name, product_id="B001")


@pytest.mark.parametrize("action_name", ["dive_deeper", "wrong_track"])
def test_recommendation_action_accepts_null_product_id_for_non_recommend_actions(action_name):
    action = RecommendationAction(action=action_name)

    assert action.action == action_name
    assert action.product_id is None
