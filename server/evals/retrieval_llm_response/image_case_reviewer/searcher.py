"""Web-image search wrapper used by the reviewer UI."""

from __future__ import annotations

from ddgs import DDGS

from .models import ImageCandidate
from .utils import safe_positive_int


class ImageSearcher:
    """Search the web for image candidates through DDGS.

    Search behavior is configured once when the object is constructed.  The
    Gradio UI therefore only needs to provide the search phrase; operational
    choices such as region, safe-search level, and result count stay in the
    INI file where they can be reviewed and reproduced.

    Keeping DDGS inside this class also prevents provider-specific result keys
    from leaking into the rest of the application.  If DDGS changes later,
    this is the only module that should normally need adjustment.
    """

    def __init__(
        self,
        *,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> None:
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch

    def search(self, query: str) -> list[ImageCandidate]:
        """Return normalized image candidates for one search phrase."""
        query = query.strip()
        if not query:
            raise ValueError("The image-search query is empty.")

        raw_results = DDGS().images(
            query=query,
            region=self.region,
            safesearch=self.safesearch,
            max_results=self.max_results,
        )

        candidates: list[ImageCandidate] = []
        seen_image_urls: set[str] = set()

        for result in raw_results:
            image_url = str(result.get("image") or "").strip()
            if not image_url or image_url in seen_image_urls:
                continue

            seen_image_urls.add(image_url)
            candidates.append(
                ImageCandidate(
                    image_url=image_url,
                    thumbnail_url=str(
                        result.get("thumbnail") or image_url
                    ).strip(),
                    title=str(result.get("title") or "").strip(),
                    source_url=str(result.get("url") or "").strip(),
                    width=safe_positive_int(result.get("width")),
                    height=safe_positive_int(result.get("height")),
                )
            )

        return candidates
