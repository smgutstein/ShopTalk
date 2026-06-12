import logging

def build_search_query_prompt():
    """Build the prompt that asks the LLM for compact vector-search terms."""
    return (
        "Based on the current conversation, what sort of product should we search for? "
        "Please ignore your personality and limit your answer to a maximum of 10 words - "
        "words an automated search system would find useful."
    )

def build_search_decision_prompt():
    """Build prompt for deciding whether this turn needs product retrieval."""
    return (
        "Decide whether the user's latest message requires a new product search.\n\n"
        "Use action='search' when the user provides new product requirements, "
        "preferences, constraints, corrections, or an uploaded image that should change "
        "the retrieved products.\n\n"
        "Use action='answer_without_search' when the user's message is conversational, "
        "asks about a previous recommendation, asks a general clarification question, "
        "says thanks, or can be answered from the existing conversation without retrieving "
        "new products.\n\n"
        "When action='search', provide a relevant, compact search_query of at most 10 words, "
        "designed to find a product matching the user's request. "
        "The search_query should contain product-relevant terms only. Do not include "
        "personality, apologies, explanations, or conversational filler."
    )

def _truncate_text(text, max_chars=700):
    """Return compact text for LLM decision prompts."""
    text = str(text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def format_source_knowledge(found_products, *, max_detail_chars=700):
    """Format retrieved products as evidence for the LLM decision prompt."""
    product_blocks = []

    for rank, (pid, info) in enumerate(found_products.items(), start=1):
        product_blocks.append(
            (
                f"rank: {rank}\n"
                f"product_id: {pid}\n"
                f"item_name: {info.item_name}\n"
                f"product_type: {info.product_type}\n"
                f"similarity_score: {info.score:.4f}\n"
                f"details: {_truncate_text(info.llm_str, max_detail_chars)}"
            )
        )

    return "\n\n---\n\n".join(product_blocks)


def build_product_decision_context(source_knowledge):
    """Build product context for the structured recommendation decision."""
    return (
        "Use the following retrieved products as the available product set for this decision.\n"
        "Do not recommend products outside this list.\n"
        "The similarity_score is retrieval evidence, not the sole determinant of product suitability. Use it in addition to product details and user request\n\n"
        "Retrieved products:\n"
        f"{source_knowledge}"
    )

def build_no_product_reprompt(dive_deeper, source_knowledge):
    """Build a follow-up prompt when no product should be recommended yet."""
    if dive_deeper:
        reprompt_str = (
            "Let's continue the conversation so we can find better product matches. "
            "Don't recommend any specific products - we're trying to learn more so we can make better recommendations. "
            "For context, here are the latest top search results, which we find promising and want to be able to dive deeper into:\n"
            f"{source_knowledge}"
        )
        logging.info("Asking the user for more details")
    else:
        reprompt_str = (
            "Let's continue the conversation to see if we can find a search area that's better served by our stock. "
            "Don't recommend any specific products - we're trying to learn more so we can see if we have anything that suits the buyer. "
            "You may want to apologize to them, since we're not finding any relevant products in our searches so far. "
            "For context, here are the latest top search results, which we're finding lacking:\n"
            f"{source_knowledge}"
        )
        logging.info("Redirect the user to another search area")

    return reprompt_str, "No-rec LLM Response"