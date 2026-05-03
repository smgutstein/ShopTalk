import importlib
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def import_generate_vector_db(monkeypatch):
    """Import generate_vector_db without requiring FAISS or ImageBind."""
    fake_faiss = types.ModuleType("faiss")

    class FakeIndexFlatIP:
        def __init__(self, dim):
            self.dim = dim
            self.added = None

        def add(self, vectors):
            self.added = vectors

    fake_faiss.IndexFlatIP = FakeIndexFlatIP
    fake_faiss.write_index = lambda index, path: Path(path).write_text("fake index")

    fake_imagebind = types.ModuleType("imagebind")
    fake_data = types.ModuleType("imagebind.data")
    fake_models = types.ModuleType("imagebind.models")
    fake_imagebind_model = types.ModuleType("imagebind.models.imagebind_model")

    class FakeModalityType:
        TEXT = "text"
        VISION = "vision"

    def fake_imagebind_huge(pretrained=True):
        raise AssertionError("ImageBind model should not be loaded by helper tests")

    fake_imagebind.data = fake_data
    fake_models.imagebind_model = fake_imagebind_model
    fake_imagebind_model.imagebind_huge = fake_imagebind_huge
    fake_imagebind_model.ModalityType = FakeModalityType

    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(sys.modules, "imagebind", fake_imagebind)
    monkeypatch.setitem(sys.modules, "imagebind.data", fake_data)
    monkeypatch.setitem(sys.modules, "imagebind.models", fake_models)
    monkeypatch.setitem(sys.modules, "imagebind.models.imagebind_model", fake_imagebind_model)

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))
    sys.modules.pop("generate_vector_db", None)
    return importlib.import_module("generate_vector_db")


@pytest.fixture
def gvdb(monkeypatch):
    return import_generate_vector_db(monkeypatch)


def test_normalize_returns_unit_vectors(gvdb):
    vectors = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    normalized = gvdb.normalize(vectors)

    assert torch.allclose(
        normalized,
        torch.tensor([[0.6, 0.8], [0.0, 1.0]]),
    )


def test_normalize_rejects_zero_vectors(gvdb):
    vectors = torch.tensor([[0.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector norm"):
        gvdb.normalize(vectors)


def test_build_product_description_uses_defaults(gvdb):
    assert gvdb.build_product_description({}) == "Uncategorized/Unknown"
    assert gvdb.build_product_description({"feature_fields": {"product_type": "Chair"}}) == "Uncategorized/Chair"


def test_build_product_description_uses_feature_fields(gvdb):
    blurb = {
        "feature_fields": {
            "categories": "Home/Kitchen",
            "product_type": "Coffee Maker",
        }
    }

    assert gvdb.build_product_description(blurb) == "Home/Kitchen/Coffee Maker"


def test_load_img_mappings_reads_required_columns(tmp_path, gvdb):
    csv_path = tmp_path / "images.csv"
    csv_path.write_text("image_id,path\nimg-1,00/example.jpg\n", encoding="utf-8")

    assert gvdb.load_img_mappings(csv_path) == {"img-1": "00/example.jpg"}


def test_load_img_mappings_rejects_missing_columns(tmp_path, gvdb):
    csv_path = tmp_path / "images.csv"
    csv_path.write_text("image_id,filename\nimg-1,example.jpg\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        gvdb.load_img_mappings(csv_path)


def test_validate_args_accepts_existing_inputs(tmp_path, gvdb):
    product_blurbs = tmp_path / "blurbs.json"
    image_root = tmp_path / "images"
    images_csv = tmp_path / "images.csv"

    product_blurbs.write_text("{}", encoding="utf-8")
    image_root.mkdir()
    images_csv.write_text("image_id,path\n", encoding="utf-8")

    args = SimpleNamespace(
        batch_size=1,
        product_blurbs=str(product_blurbs),
        image_root=str(image_root),
        images_csv=str(images_csv),
    )

    gvdb.validate_args(args)


def test_validate_args_rejects_nonpositive_batch_size(tmp_path, gvdb):
    args = SimpleNamespace(
        batch_size=0,
        product_blurbs=str(tmp_path / "blurbs.json"),
        image_root=str(tmp_path),
        images_csv=str(tmp_path / "images.csv"),
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        gvdb.validate_args(args)


def test_collect_products_with_images_keeps_products_with_existing_images(tmp_path, gvdb):
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "one.jpg").write_bytes(b"fake image bytes")

    blurbs_by_id = {
        "product_1": {"main_image_id": "img_1"},
        "product_2": {"main_image_id": "img_missing_from_csv"},
    }
    image_map = {"img_1": "one.jpg"}

    pairs = gvdb.collect_products_with_images(
        blurbs_by_id,
        image_map,
        image_root,
    )

    assert pairs == [("product_1", blurbs_by_id["product_1"])]

def test_collect_products_with_images_fails_fast_on_missing_image(tmp_path, gvdb):
    image_root = tmp_path / "images"
    image_root.mkdir()

    blurbs_by_id = {
        "p1": {"main_image_id": "img-1"},
        "p2": {"main_image_id": "img-missing"},
    }
    image_map = {
        "img-1": "exists.jpg",
        "img-missing": "missing.jpg",
    }
    (image_root / "exists.jpg").write_text("not a real image", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Use --skip_missing_images"):
        gvdb.collect_products_with_images(blurbs_by_id, image_map, image_root)


def test_collect_products_with_images_can_skip_missing_images(tmp_path, gvdb, caplog):
    image_root = tmp_path / "images"
    image_root.mkdir()

    blurbs_by_id = {
        "p1": {"main_image_id": "img-1"},
        "p2": {"main_image_id": "img-missing"},
        "p3": {"main_image_id": "not-in-csv"},
    }
    image_map = {
        "img-1": "exists.jpg",
        "img-missing": "missing.jpg",
    }
    (image_root / "exists.jpg").write_text("not a real image", encoding="utf-8")

    pairs = gvdb.collect_products_with_images(
        blurbs_by_id,
        image_map,
        image_root,
        skip_missing_images=True,
    )

    assert pairs == [("p1", blurbs_by_id["p1"])]
    assert "missing image file" in caplog.text


def test_build_image_paths(tmp_path, gvdb):
    batch_pairs = [("p1", {"main_image_id": "img-1"})]
    image_map = {"img-1": "00/example.jpg"}

    assert gvdb.build_image_paths(batch_pairs, tmp_path, image_map) == [
        str(tmp_path / "00/example.jpg")
    ]


def test_save_vectordb_creates_output_dirs_and_mapping_file(tmp_path, gvdb, monkeypatch):
    index_output = tmp_path / "artifacts" / "vector_db" / "faiss_index.bin"
    mapping_output = tmp_path / "artifacts" / "vector_db" / "index_to_product_id.pkl"
    calls = []

    def fake_write_index(index, path):
        calls.append((index, path))
        Path(path).write_text("fake index", encoding="utf-8")

    monkeypatch.setattr(gvdb.faiss, "write_index", fake_write_index)

    index = object()
    mapping = {0: "product-1"}
    gvdb.save_vectordb(index, mapping, index_output, mapping_output)

    assert calls == [(index, str(index_output))]
    assert index_output.read_text(encoding="utf-8") == "fake index"
    with open(mapping_output, "rb") as file:
        assert pickle.load(file) == mapping
