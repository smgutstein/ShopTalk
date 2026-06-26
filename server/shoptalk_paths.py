"""Project-relative paths for the ShopTalk application."""

from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "shoptalk_config.ini"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTOR_DB_OUTPUT_DIR = ARTIFACTS_DIR / "vector_db"
DEFAULT_VECTOR_BACKEND = "faiss"

EDA_DIR = PROJECT_ROOT / "EDA"
PRODUCT_BLURBS_DIR = EDA_DIR / "product_blurbs"
COMBINED_BLURBS_PATH = PRODUCT_BLURBS_DIR / "combined_blurb_dict.json"

IMAGES_CSV = PROJECT_ROOT / "images.csv"

STATIC_DIR = SERVER_DIR / "static"
STATIC_IMAGES_DIR = STATIC_DIR / "images"

DEBUG_FILE = PROJECT_ROOT / "debug.txt"
