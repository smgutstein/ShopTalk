import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recommender


def make_recommender_without_init():
    return object.__new__(recommender.ShopTalkRecommender)


def test_format_source_knowledge_lists_product_ids_and_names_in_order():
    instance = make_recommender_without_init()
    found_products = {
        "B001": {"item_name": "First product"},
        "B002": {"item_name": "Second product"},
    }

    source_knowledge = recommender.format_source_knowledge(found_products)

    assert source_knowledge == (
        "product_id: B001, item_name: First product"
        "\n\n;\n\n"
        "product_id: B002, item_name: Second product"
    )


def test_format_source_knowledge_returns_empty_string_for_no_products():
    instance = make_recommender_without_init()

    assert recommender.format_source_knowledge({}) == ""


def test_build_augmented_prompt_contains_control_options_and_source_knowledge():
    instance = make_recommender_without_init()
    source_knowledge = "product_id: B001, item_name: First product"

    prompt = recommender.build_augmented_prompt(source_knowledge)

    assert "<B071K17SWD>" in prompt
    assert "<WRONG TRACK>" in prompt
    assert "<DIVE DEEPER>" in prompt
    assert ">>Suggested Products<<" in prompt
    assert source_knowledge in prompt


def test_build_no_product_reprompt_for_dive_deeper_includes_source_knowledge():
    instance = make_recommender_without_init()
    source_knowledge = "product_id: B001, item_name: First product"

    reprompt, log_message = recommender.build_no_product_reprompt(
        dive_deeper=True,
        source_knowledge=source_knowledge,
    )

    assert log_message == "No-rec LLM Response"
    assert "find better product matches" in reprompt
    assert "Don't recommend any specific products" in reprompt
    assert source_knowledge in reprompt


def test_build_no_product_reprompt_for_wrong_track_includes_source_knowledge():
    instance = make_recommender_without_init()
    source_knowledge = "product_id: B001, item_name: First product"

    reprompt, log_message = recommender.build_no_product_reprompt(
        dive_deeper=False,
        source_knowledge=source_knowledge,
    )

    assert log_message == "No-rec LLM Response"
    assert "better served by our stock" in reprompt
    assert "apologize" in reprompt
    assert "Don't recommend any specific products" in reprompt
    assert source_knowledge in reprompt


def test_shoptalk_prompt_methods_delegate_to_module_helpers():
    instance = make_recommender_without_init()
    found_products = {"B001": {"item_name": "First product"}}
    source_knowledge = recommender.format_source_knowledge(found_products)

    assert instance._format_source_knowledge(found_products) == source_knowledge
    assert instance._build_augmented_prompt(source_knowledge) == recommender.build_augmented_prompt(source_knowledge)
    assert instance._build_no_product_reprompt(True, source_knowledge) == recommender.build_no_product_reprompt(True, source_knowledge)
    assert instance._build_no_product_reprompt(False, source_knowledge) == recommender.build_no_product_reprompt(False, source_knowledge)
