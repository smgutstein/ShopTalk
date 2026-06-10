
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RecommendationDiagnostics:
    embedding_mode: str
    llm_search_query: str | None
    top_products: list[dict]
    initial_llm_response: str
    chosen_pid: str | None
    decision: str
    timings: dict

    def to_dict(self):
        return asdict(self)


def diagnostics_to_dict(diagnostics):
    if diagnostics is None:
        return {}
    if hasattr(diagnostics, "to_dict"):
        return diagnostics.to_dict()
    return diagnostics


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
                "item_name": product.item_name,
                "score": product.score,
                "product_type": product.product_type,
                "image_paths": list(product.image_paths),
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
    return RecommendationDiagnostics(
        embedding_mode=embedding_mode,
        llm_search_query=llm_search_query,
        top_products=summarize_top_products(found_products),
        initial_llm_response=initial_llm_response,
        chosen_pid=chosen_pid,
        decision=infer_recommendation_decision(
            chosen_pid=chosen_pid,
            dive_deeper=dive_deeper,
            initial_llm_response=initial_llm_response,
        ),
        timings={"total_seconds": total_seconds},
    )
