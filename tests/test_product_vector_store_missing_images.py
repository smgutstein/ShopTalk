import logging
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = PROJECT_ROOT / "server"
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
for path in (str(STUBS_DIR), str(SERVER_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recommender import ProductVectorStore, all_img_paths


class FakeFaissIndex:
    def __init__(self, distances, indices):
        self.distances = np.array([distances], dtype=np.float32)
        self.indices = np.array([indices], dtype=np.int64)

    def search(self, query, k):
        return self.distances, self.indices


def make_blurb(main_image_id, other_image_id=None):
    return {
        "item_name": "Test Product",
        "main_image_id": main_image_id,
        "other_image_id": other_image_id or [],
        "feature_fields": {"product_type": "test product"},
        "llm_str": "A test product.",
    }


def test_all_img_paths_skips_missing_image_ids(caplog):
    blurb = make_blurb(
        main_image_id="main-img",
        other_image_id=["missing-img", "other-img"],
    )
    image_id_to_path = {
        "main-img": "images/main.jpg",
        "other-img": "images/other.jpg",
    }

    with caplog.at_level(logging.WARNING):
        image_paths = all_img_paths(blurb, image_id_to_path)

    assert image_paths == ["images/main.jpg", "images/other.jpg"]
    assert "Skipping image_id missing-img" in caplog.text


def test_product_vector_store_search_does_not_crash_on_missing_image_id(caplog):
    product_store = ProductVectorStore(
        faiss_index=FakeFaissIndex(distances=[0.91], indices=[0]),
        product_ids=["P1"],
        blurbs={"P1": make_blurb(main_image_id="missing-main")},
    )

    with caplog.at_level(logging.WARNING):
        found_products = product_store.search(
            embedded_query=[0.1, 0.2],
            top_k=1,
            image_id_to_path={},
        )

    assert found_products["P1"]["image_paths"] == []
    assert found_products["P1"]["image_urls"] == []
    assert "Skipping image_id missing-main" in caplog.text
