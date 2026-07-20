from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

try:  # Supports: python -m server...
    from ..shoptalk_paths import (
        COMBINED_BLURBS_PATH,
        DEFAULT_CONFIG_PATH,
        DEFAULT_VECTOR_BACKEND,
        IMAGES_CSV,
        VECTOR_DB_OUTPUT_DIR,
    )
except ImportError:  # Supports running from inside server/: python -m ...
    from shoptalk_paths import (
        COMBINED_BLURBS_PATH,
        DEFAULT_CONFIG_PATH,
        DEFAULT_VECTOR_BACKEND,
        IMAGES_CSV,
        VECTOR_DB_OUTPUT_DIR,
    )

DEFAULT_APP_LLM_MODEL = "gpt-4o"
DEFAULT_APP_LLM_TEMPERATURE = 0.1
DEFAULT_EVAL_LLM_MODEL = "gpt-4o"
DEFAULT_EVAL_LLM_TEMPERATURE = 0.0
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7860


@dataclass(frozen=True)
class ShopTalkFileConfig:
    app_model_name: str = DEFAULT_APP_LLM_MODEL
    app_temperature: float = DEFAULT_APP_LLM_TEMPERATURE
    eval_model_name: str = DEFAULT_EVAL_LLM_MODEL
    eval_temperature: float = DEFAULT_EVAL_LLM_TEMPERATURE
    personality_index: int = -1
    vector_db_output_dir: Path = VECTOR_DB_OUTPUT_DIR
    vector_backend: str = DEFAULT_VECTOR_BACKEND
    top_k: int = 10
    product_blurbs_path: Path = COMBINED_BLURBS_PATH
    images_csv_path: Path = IMAGES_CSV
    server_name: str = DEFAULT_SERVER_NAME
    server_port: int = DEFAULT_SERVER_PORT


def load_shoptalk_config(config_path: str | Path | None = None) -> ShopTalkFileConfig:
    """Load application, evaluation, artifact, and server settings from INI."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    parser = ConfigParser()
    if path.exists():
        parser.read(path, encoding="utf-8")

    return ShopTalkFileConfig(
        app_model_name=parser.get(
            "llm", "model_name", fallback=DEFAULT_APP_LLM_MODEL
        ),
        app_temperature=parser.getfloat(
            "llm", "temperature", fallback=DEFAULT_APP_LLM_TEMPERATURE
        ),
        eval_model_name=parser.get(
            "evals", "model_name", fallback=DEFAULT_EVAL_LLM_MODEL
        ),
        eval_temperature=parser.getfloat(
            "evals", "temperature", fallback=DEFAULT_EVAL_LLM_TEMPERATURE
        ),
        personality_index=parser.getint("app", "personality", fallback=-1),
        vector_db_output_dir=Path(
            parser.get(
                "retrieval",
                "vector_db_output_dir",
                fallback=str(VECTOR_DB_OUTPUT_DIR),
            )
        ),
        vector_backend=parser.get(
            "retrieval", "vector_backend", fallback=DEFAULT_VECTOR_BACKEND
        ),
        top_k=parser.getint("retrieval", "top_k", fallback=10),
        product_blurbs_path=Path(
            parser.get(
                "data", "product_blurbs", fallback=str(COMBINED_BLURBS_PATH)
            )
        ),
        images_csv_path=Path(
            parser.get("data", "images_csv", fallback=str(IMAGES_CSV))
        ),
        server_name=parser.get(
            "server", "server_name", fallback=DEFAULT_SERVER_NAME
        ),
        server_port=parser.getint(
            "server", "server_port", fallback=DEFAULT_SERVER_PORT
        ),
    )


@dataclass(frozen=True)
class RecommenderConfig:
    personality_index: int = -1
    debug: bool = False
    force_cpu: bool = False
    model_name: str = DEFAULT_APP_LLM_MODEL
    temperature: float = DEFAULT_APP_LLM_TEMPERATURE

    vector_db_output_dir: Path = VECTOR_DB_OUTPUT_DIR
    vector_backend: str = DEFAULT_VECTOR_BACKEND
    top_k: int = 10
    index_path: Path | None = None
    blurbs_path: Path = COMBINED_BLURBS_PATH
    product_ids_path: Path | None = None
    images_csv_path: Path = IMAGES_CSV

    @classmethod
    def from_args(cls, args):
        file_config = load_shoptalk_config(
            getattr(args, "config", DEFAULT_CONFIG_PATH)
        )
        return cls(
            personality_index=file_config.personality_index,
            debug=args.debug,
            force_cpu=args.cpu,
            model_name=file_config.app_model_name,
            temperature=file_config.app_temperature,
            vector_db_output_dir=file_config.vector_db_output_dir,
            vector_backend=file_config.vector_backend,
            top_k=file_config.top_k,
            blurbs_path=file_config.product_blurbs_path,
            images_csv_path=file_config.images_csv_path,
        )
