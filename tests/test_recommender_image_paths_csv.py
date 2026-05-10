import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recommender


def test_load_image_paths_csv_requires_existing_file(tmp_path):
    missing_csv = tmp_path / "missing_images.csv"

    with pytest.raises(FileNotFoundError, match="Image mapping CSV not found"):
        recommender.load_image_paths_csv(missing_csv)


def test_load_image_paths_csv_requires_image_id_and_path_columns(tmp_path):
    bad_csv = tmp_path / "images.csv"
    bad_csv.write_text("image_id,filename\nimg-1,one.jpg\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        recommender.load_image_paths_csv(bad_csv)


def test_load_image_paths_csv_returns_image_id_to_path_mapping(tmp_path):
    images_csv = tmp_path / "images.csv"
    images_csv.write_text(
        "image_id,path,unused\n"
        "img-1,one.jpg,ignored\n"
        "img-2,nested/two.jpg,ignored\n",
        encoding="utf-8",
    )

    assert recommender.load_image_paths_csv(images_csv) == {
        "img-1": "one.jpg",
        "img-2": "nested/two.jpg",
    }


def test_shoptalk_recommender_load_image_paths_uses_csv_loader(monkeypatch):
    monkeypatch.setattr(
        recommender,
        "load_image_paths_csv",
        lambda images_csv_path: {"img-1": f"loaded-from-{images_csv_path}"},
    )

    instance = object.__new__(recommender.ShopTalkRecommender)
    image_id_to_path, load_time = instance._load_image_paths("images.csv")

    assert image_id_to_path == {"img-1": "loaded-from-images.csv"}
    assert isinstance(load_time, str)
