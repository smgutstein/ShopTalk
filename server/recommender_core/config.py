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


@dataclass(frozen=True)
class ShopTalkFileConfig:
    app_model_name: str = DEFAULT_APP_LLM_MODEL
    app_temperature: float = DEFAULT_APP_LLM_TEMPERATURE
    eval_model_name: str = DEFAULT_EVAL_LLM_MODEL
    eval_temperature: float = DEFAULT_EVAL_LLM_TEMPERATURE


def load_shoptalk_config(config_path: str | Path | None = None) -> ShopTalkFileConfig:
    """Load app/eval LLM settings from the ShopTalk INI config file.

    Missing files and missing keys fall back to explicit code defaults. That keeps
    tests and ad-hoc scripts usable while still making the config file the normal
    source of user-editable model settings.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    parser = ConfigParser()
    if path.exists():
        parser.read(path, encoding="utf-8")

    app_model_name = parser.get("llm", "model_name", fallback=DEFAULT_APP_LLM_MODEL)
    app_temperature = parser.getfloat(
        "llm",
        "temperature",
        fallback=DEFAULT_APP_LLM_TEMPERATURE,
    )
    eval_model_name = parser.get(
        "evals",
        "model_name",
        fallback=DEFAULT_EVAL_LLM_MODEL,
    )
    eval_temperature = parser.getfloat(
        "evals",
        "temperature",
        fallback=DEFAULT_EVAL_LLM_TEMPERATURE,
    )

    return ShopTalkFileConfig(
        app_model_name=app_model_name,
        app_temperature=app_temperature,
        eval_model_name=eval_model_name,
        eval_temperature=eval_temperature,
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
        config_path = getattr(args, "config", DEFAULT_CONFIG_PATH)
        file_config = load_shoptalk_config(config_path)
        model_name = getattr(args, "model", None) or file_config.app_model_name
        arg_temperature = getattr(args, "temperature", None)
        temperature = (
            arg_temperature
            if arg_temperature is not None
            else file_config.app_temperature
        )

        return cls(
            personality_index=args.personality,
            debug=args.debug,
            force_cpu=args.cpu,
            model_name=model_name,
            temperature=temperature,
            vector_db_output_dir=Path(args.vector_db_output_dir),
            vector_backend=args.vector_backend,
            top_k=args.top_k,
            blurbs_path=Path(args.product_blurbs),
            images_csv_path=Path(args.images_csv),
        )
