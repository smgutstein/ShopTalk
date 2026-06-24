import json

import pytest

from server.recommender_core import vector_db
from server.recommender_core.product_vector_store import ProductVectorStore


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

    monkeypatch.setattr(vector_db.faiss, "read_index", fake_read_index)

    index, product_ids, blurbs = vector_db.load_vector_db(
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

    monkeypatch.setattr(vector_db.faiss, "read_index", fake_read_index)

    with pytest.raises(ValueError, match="FAISS index contains 3 vectors"):
        vector_db.load_vector_db(
            index_path=index_path,
            blurbs_path=blurbs_path,
            product_ids_path=product_ids_path,
        )


class FakeSearchIndex:
    ntotal = 3
    d = 1024

    def __init__(self, distances, indices):
        self.distances = distances
        self.indices = indices
        self.seen_queries = []
        self.seen_k = []

    def search(self, query, k):
        self.seen_queries.append(query)
        self.seen_k.append(k)
        return self.distances, self.indices


def make_blurb(item_name, main_image_id, other_image_id=None):
    return {
        "item_name": item_name,
        "main_image_id": main_image_id,
        "other_image_id": other_image_id or [],
        "feature_fields": {"product_type": "widget"},
        "llm_str": f"Description for {item_name}",
    }


def test_product_vector_store_search_maps_rows_to_product_records():
    fake_index = FakeSearchIndex(
        distances=[[0.9, 0.8]],
        indices=[[1, 0]],
    )
    store = ProductVectorStore(
        faiss_index=fake_index,
        product_ids=["product-0", "product-1"],
        blurbs={
            "product-0": make_blurb("Zero", "img-zero"),
            "product-1": make_blurb("One", "img-one", ["img-one-extra"]),
        },
    )
    image_id_to_path = {
        "img-zero": "zero.jpg",
        "img-one": "one.jpg",
        "img-one-extra": "one-extra.jpg",
    }

    found_products = store.search(
        embedded_query=[0.1, 0.2, 0.3],
        top_k=2,
        image_id_to_path=image_id_to_path,
    )

    assert fake_index.seen_k == [2]
    assert list(found_products) == ["product-1", "product-0"]
    assert found_products["product-1"].item_name == "One"
    assert found_products["product-1"].score == 0.9
    assert found_products["product-1"].image_paths == ("one.jpg", "one-extra.jpg")
    assert found_products["product-1"].product_type == "widget"
    assert found_products["product-1"].llm_str == "Description for One"
    assert found_products["product-0"].image_paths == ("zero.jpg",)


def test_product_vector_store_search_deduplicates_repeated_product_ids():
    fake_index = FakeSearchIndex(
        distances=[[0.95, 0.90]],
        indices=[[0, 1]],
    )
    store = ProductVectorStore(
        faiss_index=fake_index,
        product_ids=["product-1", "product-1"],
        blurbs={
            "product-1": make_blurb("One", "img-one", ["img-one-extra"]),
        },
    )
    image_id_to_path = {
        "img-one": "one.jpg",
        "img-one-extra": "one-extra.jpg",
    }

    found_products = store.search(
        embedded_query=[0.1],
        top_k=2,
        image_id_to_path=image_id_to_path,
    )

    assert list(found_products) == ["product-1"]
    assert found_products["product-1"].score == 0.95
    assert found_products["product-1"].image_paths == (
        "one.jpg",
        "one-extra.jpg",
    )


def test_product_vector_store_search_rejects_out_of_range_faiss_row():
    fake_index = FakeSearchIndex(
        distances=[[0.7]],
        indices=[[3]],
    )
    store = ProductVectorStore(
        faiss_index=fake_index,
        product_ids=["product-0"],
        blurbs={"product-0": make_blurb("Zero", "img-zero")},
    )

    with pytest.raises(IndexError, match="FAISS returned row 3"):
        store.search(
            embedded_query=[0.1],
            top_k=1,
            image_id_to_path={"img-zero": "zero.jpg"},
        )
