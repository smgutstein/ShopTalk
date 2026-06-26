from dataclasses import dataclass
from pathlib import Path

from ..shoptalk_paths import (
    COMBINED_BLURBS_PATH,
    DEFAULT_VECTOR_BACKEND,
    IMAGES_CSV,
    VECTOR_DB_OUTPUT_DIR,
)

DEFAULT_LLM_MODEL = os.environ.get("SHOPTALK_LLM_MODEL", "gpt-4o")

@dataclass(frozen=True)
class RecommenderConfig:
    personality_index: int = -1
    debug: bool = False
    force_cpu: bool = False
    model_name: str = DEFAULT_LLM_MODEL

    vector_db_output_dir: Path = VECTOR_DB_OUTPUT_DIR
    vector_backend: str = DEFAULT_VECTOR_BACKEND
    top_k: int = 10
    index_path: Path | None = None
    blurbs_path: Path = COMBINED_BLURBS_PATH
    product_ids_path: Path | None = None
    images_csv_path: Path = IMAGES_CSV

    @classmethod
    def from_args(cls, args):
        return cls(
            personality_index=args.personality,
            debug=args.debug,
            force_cpu=args.cpu,
            model_name=args.model,
            vector_db_output_dir=Path(args.vector_db_output_dir),
            vector_backend=args.vector_backend,
            top_k=args.top_k,
            blurbs_path=Path(args.product_blurbs),
            images_csv_path=Path(args.images_csv),
        )