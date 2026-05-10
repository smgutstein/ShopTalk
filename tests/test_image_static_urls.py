from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = PROJECT_ROOT / "server"
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
for path in (str(STUBS_DIR), str(SERVER_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recommender import image_path_to_static_url, image_paths_to_static_urls


def test_image_path_to_static_url_handles_common_path_shapes():
    assert image_path_to_static_url("abc/def.jpg") == "/static/images/abc/def.jpg"
    assert image_path_to_static_url("images/abc/def.jpg") == "/static/images/abc/def.jpg"
    assert image_path_to_static_url("static/images/abc/def.jpg") == "/static/images/abc/def.jpg"
    assert image_path_to_static_url("server/static/images/abc/def.jpg") == "/static/images/abc/def.jpg"


def test_image_path_to_static_url_normalizes_windows_style_paths():
    assert image_path_to_static_url(r"abc\def.jpg") == "/static/images/abc/def.jpg"


def test_image_paths_to_static_urls_converts_each_path():
    assert image_paths_to_static_urls(["a.jpg", "images/b.jpg"]) == [
        "/static/images/a.jpg",
        "/static/images/b.jpg",
    ]
