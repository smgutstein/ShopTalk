import importlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = REPO_ROOT / "EDA"

sys.path.insert(0, str(EDA_DIR))

def import_description_generator(monkeypatch):
    """Import DescriptionGenerator while stubbing Preprocessor dependencies."""
    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = lambda model_name: object()

    fake_nltk = types.ModuleType("nltk")
    fake_nltk.download = lambda name: None

    fake_corpus = types.ModuleType("nltk.corpus")

    class FakeStopwords:
        @staticmethod
        def words(language):
            return []

    fake_corpus.stopwords = FakeStopwords

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, *args, **kwargs: iterable

    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
    monkeypatch.setitem(sys.modules, "nltk.corpus", fake_corpus)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "EDA"))
    sys.modules.pop("preprocessor", None)
    sys.modules.pop("DescriptionGenerator", None)
    return importlib.import_module("DescriptionGenerator")


@pytest.fixture
def description_module(monkeypatch):
    return import_description_generator(monkeypatch)


def make_generator(tmp_path, description_module, listing_rows=None):
    listing_dir = tmp_path / "listings"
    listing_dir.mkdir()
    data_file = listing_dir / "sample.json"

    if listing_rows is not None:
        with open(data_file, "w", encoding="utf-8") as file:
            for row in listing_rows:
                file.write(json.dumps(row) + "\n")
    else:
        data_file.write_text("", encoding="utf-8")

    return description_module.DescriptionGenerator(
        root_dir=tmp_path,
        listing_dir="listings",
        data_file="sample.json",
        country="US",
        language="en_US",
        config_file=str(tmp_path / "missing_config.ini"),
    )


def test_filter_for_country_keeps_matching_country(tmp_path, description_module):
    rows = [
        {"item_id": "us-1", "country": "US"},
        {"item_id": "de-1", "country": "DE"},
    ]
    dg = make_generator(tmp_path, description_module, rows)

    filtered = dg.filter_for_country()

    assert [item["item_id"] for item in filtered] == ["us-1"]


def test_filter_for_language_keeps_matching_language_tag(tmp_path, description_module):
    dg = make_generator(tmp_path, description_module)
    in_list = [
        {
            "item_id": "p1",
            "title": [
                {"language_tag": "en_US", "value": "English title"},
                {"language_tag": "de_DE", "value": "German title"},
                {"value": "language-neutral value"},
                "raw-list-value",
            ],
            "country": "US",
        }
    ]

    out = dg.filter_for_language(in_list)

    assert out == [
        {
            "country": "US",
            "item_id": "p1",
            "title": [
                {"language_tag": "en_US", "value": "English title"},
                {"value": "language-neutral value"},
                "raw-list-value",
            ],
        }
    ]


def test_make_data_dict_initializes_llm_and_feature_fields(tmp_path, description_module, monkeypatch):
    dg = make_generator(tmp_path, description_module)
    monkeypatch.setattr(
        dg,
        "get_filtered_product_list",
        lambda: [
            {
                "item_id": "p1",
                "country": "US",
                "item_name": [{"value": "Test Item"}],
            }
        ],
    )

    dg.make_data_dict()

    assert "p1" in dg.item_id_dict
    assert dg.item_id_dict["p1"]["llm_str"] == ""
    assert dg.item_id_dict["p1"]["feature_fields"] == {}
    assert "item_id" not in dg.item_id_dict["p1"]


def test_get_item_name_and_brand_text_sets_llm_and_features(tmp_path, description_module):
    dg = make_generator(tmp_path, description_module)
    dg.item_id_dict = {
        "p1": {
            "item_name": [{"value": "AmazonBasics - Cotton Towel"}],
            "brand": [{"value": "Acme"}],
            "llm_str": "",
            "feature_fields": {},
        }
    }

    dg.get_item_name_and_brand_text()

    assert dg.item_id_dict["p1"]["feature_fields"]["product_name"] == "Cotton Towel"
    assert dg.item_id_dict["p1"]["feature_fields"]["brand"] == "Acme"
    assert "Product ID is p1" in dg.item_id_dict["p1"]["llm_str"]
    assert "Its brand is Acme" in dg.item_id_dict["p1"]["llm_str"]


def test_make_blurb_dict_preserves_text_features_and_image_ids(tmp_path, description_module):
    dg = make_generator(tmp_path, description_module)
    dg.item_id_dict = {
        "p1": {
            "item_name": [{"value": "Test Item"}],
            "llm_str": "Product text",
            "feature_fields": {"color": "red"},
            "main_image_id": "img-main",
            "other_image_id": ["img-2", "img-3"],
        }
    }

    dg.make_blurb_dict()

    assert dg.blurb_dict["p1"] == {
        "llm_str": "Product text",
        "item_name": "Test Item",
        "feature_fields": {"color": "red"},
        "main_image_id": "img-main",
        "other_image_id": ["img-2", "img-3"],
    }


def test_save_full_blurb_dict_writes_json(tmp_path, description_module):
    dg = make_generator(tmp_path, description_module)
    output_dir = tmp_path / "product_blurbs"
    output_json = "combined_blurb_dict.json"
    blurb_dict = {"p1": {"llm_str": "Product text"}}

    dg.save_full_blurb_dict(output_dir, output_json, blurb_dict)

    with open(output_dir / output_json, "r", encoding="utf-8") as file:
        assert json.load(file) == blurb_dict
