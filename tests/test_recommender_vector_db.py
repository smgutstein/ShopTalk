import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = PROJECT_ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import recommender


class FakeFaissIndex:
    def __init__(self, ntotal=2, d=1024):
        self.ntotal = ntotal
        self.d = d


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_vector_db_reads_faiss_index_and_product_ids_json(monkeypatch, tmp_path):
    index_path = tmp_path / "embeddings.faiss"
    product_ids_path = tmp_path / "product_ids.json"
    blurbs_path = tmp_path / "combined_blurb_dict.json"

    # The file only needs to exist; faiss.read_index is mocked below.
    index_path.write_bytes(b"fake faiss bytes")
    write_json(product_ids_path, ["product-1", "product-2"])
    write_json(
        blurbs_path,
        {
            "product-1": {"item_name": "First product"},
            "product-2": {"item_name": "Second product"},
        },
    )

    seen_paths = []

    def fake_read_index(path):
        seen_paths.append(path)
        return FakeFaissIndex(ntotal=2, d=1024)

    monkeypatch.setattr(recommender.faiss, "read_index", fake_read_index)

    index, product_ids, blurbs = recommender.load_vector_db(
        index_path=index_path,
        blurbs_path=blurbs_path,
        product_ids_path=product_ids_path,
    )

    assert isinstance(index, FakeFaissIndex)
    assert seen_paths == [str(index_path)]
    assert product_ids == ["product-1", "product-2"]
    assert blurbs["product-1"]["item_name"] == "First product"


def test_load_vector_db_rejects_mismatched_index_and_product_id_counts(monkeypatch, tmp_path):
    index_path = tmp_path / "embeddings.faiss"
    product_ids_path = tmp_path / "product_ids.json"
    blurbs_path = tmp_path / "combined_blurb_dict.json"

    index_path.write_bytes(b"fake faiss bytes")
    write_json(product_ids_path, ["product-1", "product-2"])
    write_json(blurbs_path, {})

    def fake_read_index(path):
        return FakeFaissIndex(ntotal=3, d=1024)

    monkeypatch.setattr(recommender.faiss, "read_index", fake_read_index)

    with pytest.raises(ValueError, match="FAISS index contains 3 vectors"):
        recommender.load_vector_db(
            index_path=index_path,
            blurbs_path=blurbs_path,
            product_ids_path=product_ids_path,
        )
