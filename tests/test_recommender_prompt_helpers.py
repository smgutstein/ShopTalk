from server.recommender_core.llm_prompts import (
    build_no_product_reprompt,
    build_product_decision_context,
    format_source_knowledge,
)
from server.recommender_core.product_candidate import ProductCandidate


def make_candidate(product_id, name, score=0.91, product_type="shoes", details=None):
    return ProductCandidate(
        product_id=product_id,
        item_name=name,
        score=score,
        image_paths=(),
        product_type=product_type,
        llm_str=details or f"Details for {name}",
    )


def test_format_source_knowledge_lists_rich_product_context_in_order():
    found_products = {
        "B001": make_candidate("B001", "First product", score=0.91),
        "B002": make_candidate("B002", "Second product", score=0.82),
    }

    source_knowledge = format_source_knowledge(found_products)

    assert "rank: 1" in source_knowledge
    assert "product_id: B001" in source_knowledge
    assert "item_name: First product" in source_knowledge
    assert "product_type: shoes" in source_knowledge
    assert "similarity_score: 0.9100" in source_knowledge
    assert "details: Details for First product" in source_knowledge
    assert source_knowledge.index("product_id: B001") < source_knowledge.index("product_id: B002")


def test_format_source_knowledge_truncates_long_details():
    found_products = {
        "B001": make_candidate("B001", "First product", details="x" * 20),
    }

    source_knowledge = format_source_knowledge(found_products, max_detail_chars=5)

    assert "details: xxxxx..." in source_knowledge


def test_format_source_knowledge_returns_empty_string_for_no_products():
    assert format_source_knowledge({}) == ""


def test_build_product_decision_context_uses_structured_decision_language():
    source_knowledge = "product_id: B001\nitem_name: First product"

    prompt = build_product_decision_context(source_knowledge)

    assert "available product set" in prompt
    assert "Do not recommend products outside this list" in prompt
    assert "similarity_score is retrieval evidence" in prompt
    assert source_knowledge in prompt
    assert "<DIVE DEEPER>" not in prompt
    assert "<WRONG TRACK>" not in prompt
    assert ">>Suggested Products<<" not in prompt


def test_build_no_product_reprompt_for_dive_deeper_includes_source_knowledge():
    source_knowledge = "product_id: B001, item_name: First product"

    reprompt, log_message = build_no_product_reprompt(
        dive_deeper=True,
        source_knowledge=source_knowledge,
    )

    assert log_message == "No-rec LLM Response"
    assert "find better product matches" in reprompt
    assert "Don't recommend any specific products" in reprompt
    assert source_knowledge in reprompt


def test_build_no_product_reprompt_for_wrong_track_includes_source_knowledge():
    source_knowledge = "product_id: B001, item_name: First product"

    reprompt, log_message = build_no_product_reprompt(
        dive_deeper=False,
        source_knowledge=source_knowledge,
    )

    assert log_message == "No-rec LLM Response"
    assert "better served by our stock" in reprompt
    assert "apologize" in reprompt
    assert "Don't recommend any specific products" in reprompt
    assert source_knowledge in reprompt
