# server/recommender_core/reply_types.py

from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
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

    search_performed: bool
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


    @model_validator(mode="after")
    def validate_product_id(self):
        """Enforce action/product_id consistency on structured LLM output."""
        if self.action == "recommend" and not self.product_id:
            raise ValueError("product_id is required when action='recommend'")

        if self.action != "recommend" and self.product_id is not None:
            raise ValueError("product_id must be null unless action='recommend'")

        return self
    
class SearchDecision(BaseModel):
    """Structured LLM decision about whether retrieval is needed."""

    action: Literal["search", "answer_without_search"] = Field(
        description=(
            "Whether to search the product database. Use 'search' when the user "
            "has provided new product preferences, constraints, corrections, or an image "
            "that should affect product retrieval. Use 'answer_without_search' for "
            "thanks, clarification questions, conversational replies, or cases where "
            "retrieval would not improve the next response."
        )
    )

    search_query: str | None = Field(
        default=None,
        description=(
            "Compact product search query. Required when action is 'search'. "
            "Must be null when action is 'answer_without_search'."
        ),
    )

    @model_validator(mode="after")
    def validate_search_query(self):
        """Enforce action/search_query consistency on structured LLM output."""
        if self.action == "search" and not self.search_query:
            raise ValueError("search_query is required when action='search'")

        if self.action != "search" and self.search_query is not None:
            raise ValueError("search_query must be null unless action='search'")

        return self