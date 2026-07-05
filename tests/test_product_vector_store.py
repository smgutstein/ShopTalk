import numpy as np
import pytest

from server.recommender_core.product_vector_store import ProductVectorStore


class FakeFaissIndex:
    def __init__(self, distances, indices):
        self.distances = np.array([distances], dtype=np.float32)
        self.indices = np.array([indices], dtype=np.int64)

    def search(self, query_matrix, k):
        self.query_matrix = query_matrix
        self.k = k
        return self.distances, self.indices


def make_blurb(name, main_image_id, other_image_id=None, product_type="test product"):
    return {
        "item_name": name,
        "main_image_id": main_image_id,
        "other_image_id": other_image_id,
        "feature_fields": {"product_type": product_type},
        "llm_str": f"Description for {name}",
    }


def test_search_maps_faiss_rows_to_product_records():
    index = FakeFaissIndex(distances=[0.91, 0.73], indices=[1, 0])
    store = ProductVectorStore(
        faiss_index=index,
        product_ids=["pid_a", "pid_b"],
        blurbs={
            "pid_a": make_blurb("Alpha", "img_a"),
            "pid_b": make_blurb("Beta", "img_b", ["img_b2"]),
        },
    )
    image_id_to_path = {
        "img_a": "alpha.jpg",
        "img_b": "beta.jpg",
        "img_b2": "beta_detail.jpg",
    }

    found = store.search(
        embedded_query=[0.1, 0.2, 0.3],
        top_k=2,
        image_id_to_path=image_id_to_path,
    )

    assert list(found.keys()) == ["pid_b", "pid_a"]
    beta = found["pid_b"]
    assert beta.product_id == "pid_b"
    assert beta.item_name == "Beta"
    assert beta.score == pytest.approx(0.91)
    assert beta.image_paths == ("beta.jpg", "beta_detail.jpg")
    assert beta.product_type == "test product"
    assert beta.llm_str == "Description for Beta"
    assert found["pid_a"].score == pytest.approx(0.73)
    assert index.k == 2
    assert index.query_matrix.dtype == np.float32
    assert index.query_matrix.shape == (1, 3)


def test_search_ignores_negative_faiss_indices():
    index = FakeFaissIndex(distances=[0.91, -3.4], indices=[0, -1])
    store = ProductVectorStore(
        faiss_index=index,
        product_ids=["pid_a"],
        blurbs={"pid_a": make_blurb("Alpha", "img_a")},
    )

    found = store.search(
        embedded_query=[0.1, 0.2, 0.3],
        top_k=2,
        image_id_to_path={"img_a": "alpha.jpg"},
    )

    assert list(found.keys()) == ["pid_a"]


def test_search_merges_duplicate_product_rows_without_duplicate_image_paths():
    index = FakeFaissIndex(distances=[0.91, 0.88], indices=[0, 1])
    store = ProductVectorStore(
        faiss_index=index,
        product_ids=["pid_a", "pid_a"],
        blurbs={"pid_a": make_blurb("Alpha", "img_a", ["img_a2"])},
    )

    found = store.search(
        embedded_query=[0.1, 0.2, 0.3],
        top_k=2,
        image_id_to_path={"img_a": "alpha.jpg", "img_a2": "alpha_detail.jpg"},
    )

    assert list(found.keys()) == ["pid_a"]
    assert found["pid_a"].score == pytest.approx(0.91)
    assert found["pid_a"].image_paths == ("alpha.jpg", "alpha_detail.jpg")


def test_search_raises_if_faiss_returns_unknown_row():
    index = FakeFaissIndex(distances=[0.91], indices=[3])
    store = ProductVectorStore(
        faiss_index=index,
        product_ids=["pid_a"],
        blurbs={"pid_a": make_blurb("Alpha", "img_a")},
    )

    with pytest.raises(IndexError, match="FAISS returned row 3"):
        store.search(
            embedded_query=[0.1, 0.2, 0.3],
            top_k=1,
            image_id_to_path={"img_a": "alpha.jpg"},
        )


class AdaptiveFakeFaissIndex:
    def __init__(self, rows):
        self.rows = rows
        self.requested_k_values = []

    def search(self, query_matrix, k):
        self.requested_k_values.append(k)
        returned_rows = self.rows[:k]
        distances = [[score for score, _idx in returned_rows]]
        indices = [[idx for _score, idx in returned_rows]]
        return np.array(distances, dtype=np.float32), np.array(indices, dtype=np.int64)


def test_search_overfetches_until_top_k_unique_products_are_found():
    index = AdaptiveFakeFaissIndex(
        rows=[
            (0.99, 0),
            (0.98, 1),
            (0.97, 2),
            (0.96, 3),
            (0.95, 4),
        ]
    )
    store = ProductVectorStore(
        faiss_index=index,
        product_ids=["pid_a", "pid_a", "pid_b", "pid_c", "pid_d"],
        blurbs={
            "pid_a": make_blurb("Alpha", "img_a"),
            "pid_b": make_blurb("Beta", "img_b"),
            "pid_c": make_blurb("Gamma", "img_c"),
            "pid_d": make_blurb("Delta", "img_d"),
        },
    )

    found = store.search(
        embedded_query=[0.1, 0.2, 0.3],
        top_k=3,
        image_id_to_path={
            "img_a": "alpha.jpg",
            "img_b": "beta.jpg",
            "img_c": "gamma.jpg",
            "img_d": "delta.jpg",
        },
    )

    assert index.requested_k_values == [3, 5]
    assert list(found) == ["pid_a", "pid_b", "pid_c"]
    assert len(found) == 3


def test_search_stops_when_index_exhausted_even_if_fewer_unique_products_exist():
    index = AdaptiveFakeFaissIndex(
        rows=[
            (0.99, 0),
            (0.98, 1),
            (0.97, 2),
            (0.96, 3),
        ]
    )
    store = ProductVectorStore(
        faiss_index=index,
        product_ids=["pid_a", "pid_a", "pid_b", "pid_b"],
        blurbs={
            "pid_a": make_blurb("Alpha", "img_a"),
            "pid_b": make_blurb("Beta", "img_b"),
        },
    )

    found = store.search(
        embedded_query=[0.1, 0.2, 0.3],
        top_k=5,
        image_id_to_path={"img_a": "alpha.jpg", "img_b": "beta.jpg"},
    )

    assert index.requested_k_values == [4]
    assert list(found) == ["pid_a", "pid_b"]
    assert len(found) == 2
