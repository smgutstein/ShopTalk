

def summarize_top_products(found_products):
    """Create compact diagnostics for retrieved products.

    Include image paths/URLs so UIs can show retrieval thumbnails alongside
    scores. These are already lightweight strings, not image payloads.
    """
    summaries = []
    for product_id, product in found_products.items():
        summaries.append(
            {
                "product_id": product_id,
                "item_name": product.get("item_name"),
                "score": product.get("score"),
                "product_type": product.get("product_type"),
                "image_paths": list(product.get("image_paths") or []),
                "image_urls": list(product.get("image_urls") or []),
            }
        )
    return summaries


def infer_recommendation_decision(chosen_pid, dive_deeper, initial_llm_response):
    """Convert the LLM control response into a coarse diagnostic decision."""
    if chosen_pid:
        return "recommend"
    if dive_deeper:
        return "dive_deeper"
    if "WRONG TRACK" in initial_llm_response:
        return "wrong_track"
    return "unknown"


def build_recommendation_diagnostics(
    *,
    embedding_mode,
    llm_search_query,
    found_products,
    initial_llm_response,
    chosen_pid,
    dive_deeper,
    total_seconds,
):
    """Build internal diagnostics for retrieval and LLM-control behavior."""
    return {
        "embedding_mode": embedding_mode,
        "llm_search_query": llm_search_query,
        "top_products": summarize_top_products(found_products),
        "initial_llm_response": initial_llm_response,
        "chosen_pid": chosen_pid,
        "decision": infer_recommendation_decision(
            chosen_pid=chosen_pid,
            dive_deeper=dive_deeper,
            initial_llm_response=initial_llm_response,
        ),
        "timings": {"total_seconds": total_seconds},
    }
