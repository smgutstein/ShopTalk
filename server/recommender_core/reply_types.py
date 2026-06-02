# server/recommender_core/reply_types.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    found_products: dict[str, dict[str, Any]]
    source_knowledge: str


@dataclass(frozen=True)
class RecommendationDecision:
    """LLM's intermediate decision before the final user-facing response."""

    initial_llm_response: str
    chosen_pid: str | None
    chosen_product: dict[str, Any] | None
    dive_deeper: bool

@dataclass(frozen=True)
class ReplyTiming:
    minutes: int
    seconds: int
    total_seconds: float