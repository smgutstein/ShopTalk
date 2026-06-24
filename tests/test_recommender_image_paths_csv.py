import pytest

from server.recommender_core import recommender_factory
from server.recommender_core.product_images import load_image_paths_csv


def test_load_image_paths_csv_requires_existing_file(tmp_path):
    missing_csv = tmp_path / "missing_images.csv"

    with pytest.raises(FileNotFoundError, match="Image mapping CSV not found"):
        load_image_paths_csv(missing_csv)


def test_load_image_paths_csv_requires_image_id_and_path_columns(tmp_path):
    bad_csv = tmp_path / "images.csv"
    bad_csv.write_text("image_id,filename\nimg-1,one.jpg\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_image_paths_csv(bad_csv)


def test_load_image_paths_csv_returns_image_id_to_path_mapping(tmp_path):
    images_csv = tmp_path / "images.csv"
    images_csv.write_text(
        "image_id,path,unused\n"
        "img-1,one.jpg,ignored\n"
        "img-2,nested/two.jpg,ignored\n",
        encoding="utf-8",
    )

    assert load_image_paths_csv(images_csv) == {
        "img-1": "one.jpg",
        "img-2": "nested/two.jpg",
    }


def test_factory_load_image_paths_uses_csv_loader(monkeypatch):
    monkeypatch.setattr(
        recommender_factory,
        "load_image_paths_csv",
        lambda images_csv_path: {"img-1": f"loaded-from-{images_csv_path}"},
    )

    image_id_to_path, load_time = recommender_factory.load_image_paths("images.csv")

    assert image_id_to_path == {"img-1": "loaded-from-images.csv"}
    assert isinstance(load_time, str)
