import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def load_server_module_with_fakes(monkeypatch):
    fake_flask_module = types.ModuleType("flask")
    fake_flask_module.Flask = object
    fake_flask_module.jsonify = lambda value: value
    fake_flask_module.render_template = lambda *args, **kwargs: ""
    fake_flask_module.request = SimpleNamespace(json={})
    monkeypatch.setitem(sys.modules, "flask", fake_flask_module)

    fake_recommender_module = types.ModuleType("recommender")

    class FakeShopTalkRecommender:
        calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeShopTalkRecommender.calls.append(kwargs)

    fake_recommender_module.ShopTalkRecommender = FakeShopTalkRecommender
    monkeypatch.setitem(sys.modules, "recommender", fake_recommender_module)

    server_path = Path(__file__).resolve().parents[1] / "server" / "server.py"
    spec = importlib.util.spec_from_file_location("server_under_test", server_path)
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)

    return server_module, FakeShopTalkRecommender


def test_build_recommender_passes_cli_args_to_recommender(monkeypatch):
    server_module, fake_recommender_cls = load_server_module_with_fakes(monkeypatch)

    args = SimpleNamespace(
        personality=4,
        debug=True,
        cpu=True,
        model="gpt-test",
        vector_db_output_dir="artifacts/test_vector_db",
        vector_backend="faiss",
        product_blurbs="tests/fixtures/product_blurbs.json",
        images_csv="tests/fixtures/images.csv",
    )

    recommender = server_module.build_recommender(args)

    assert isinstance(recommender, fake_recommender_cls)
    assert fake_recommender_cls.calls == [
        {
            "personality_index": 4,
            "debug": True,
            "force_cpu": True,
            "model_name": "gpt-test",
            "vector_db_output_dir": "artifacts/test_vector_db",
            "vector_backend": "faiss",
            "blurbs_path": "tests/fixtures/product_blurbs.json",
            "images_csv_path": "tests/fixtures/images.csv",
        }
    ]
