from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ProductCandidate:
    """Retrieved product metadata used by recommendation and UI layers."""

    product_id: str
    item_name: str
    score: float
    image_paths: tuple[str, ...]
    product_type: str
    llm_str: str

    @classmethod
    def from_blurb(cls, product_id, blurb, score, image_paths):
        """Build a candidate from vector-db product metadata."""
        return cls(
            product_id=product_id,
            item_name=blurb["item_name"],
            score=float(score),
            image_paths=tuple(image_paths),
            product_type=blurb["feature_fields"]["product_type"],
            llm_str=blurb["llm_str"],
        )

    def with_additional_image_paths(self, image_paths):
        """Return a copy with new image paths appended without duplicates."""
        merged_paths = list(self.image_paths)
        for image_path in image_paths:
            if image_path not in merged_paths:
                merged_paths.append(image_path)

        return replace(self, image_paths=tuple(merged_paths))
