# server/shoptalk_paths.py

from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
PRODUCT_DATA_DIR = PROJECT_ROOT / "product_data"

IMAGES_CSV = DATA_DIR / "images.csv"
FAISS_INDEX_PATH = VECTOR_DB_DIR / "index.faiss"
BLURBS_PATH = VECTOR_DB_DIR / "blurbs.json"
PRODUCT_IDS_PATH = VECTOR_DB_DIR / "product_ids.json"

STATIC_DIR = SERVER_DIR / "static"
STATIC_IMAGES_DIR = STATIC_DIR / "images"