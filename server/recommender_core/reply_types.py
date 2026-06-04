# server/recommender_core/reply_types.py

from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal

from .product_candidate import ProductCandidate



@dataclass(frozen=True)
class ReplyRequest:
    """Normalized user request for a single recommender turn."""

    user_input: str | None
    image_path: Path | None
    embedding_mode: str
    message_content: str


@dataclass(frozen=True)
class ProductSearchResult:
    """Products and supporting text retrieved for a recommender turn."""

    llm_search_query: str | None
    found_products: dict[str, ProductCandidate]
    source_knowledge: str


@dataclass(frozen=True)
class RecommendationDecision:
    """LLM's intermediate decision before the final user-facing response."""

    initial_llm_response: str
    chosen_pid: str | None
    chosen_product: ProductCandidate | None
    dive_deeper: bool

@dataclass(frozen=True)
class ReplyTiming:
    minutes: int
    seconds: int
    total_seconds: float

class RecommendationAction(BaseModel):
    """Structured LLM decision about the next recommendation action."""

    action: Literal["recommend", "dive_deeper", "wrong_track"] = Field(
        description=(
            "The next action. Use 'recommend' only when one retrieved product "
            "is clearly suitable. Use 'dive_deeper' when more user preference "
            "information is needed. Use 'wrong_track' when the retrieved products "
            "do not match the user's needs."
        )
    )
    product_id: str | None = Field(
        default=None,
        description=(
            "The selected product id. Required when action is 'recommend'. "
            "Must exactly match one of the retrieved product ids. Must be null "
            "for 'dive_deeper' or 'wrong_track'."
        ),
    )