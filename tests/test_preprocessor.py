import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = REPO_ROOT / "EDA"

sys.path.insert(0, str(EDA_DIR))


def import_preprocessor(monkeypatch):
    """Import EDA/preprocessor.py without requiring real spaCy/NLTK assets."""
    fake_spacy = types.ModuleType("spacy")

    class FakeToken:
        def __init__(self, text):
            self.lemma_ = text

    class FakeDoc:
        def __init__(self, text):
            self.text = text
            self.sents = [types.SimpleNamespace(text=text)] if text else []

        def __iter__(self):
            return iter(FakeToken(word) for word in self.text.split())

    class FakeNLP:
        def __call__(self, text):
            return FakeDoc(text)

    fake_spacy.load = lambda model_name: FakeNLP()

    fake_nltk = types.ModuleType("nltk")
    fake_nltk.download = lambda name: None

    fake_corpus = types.ModuleType("nltk.corpus")

    class FakeStopwords:
        @staticmethod
        def words(language):
            assert language == "english"
            return ["the", "and", "is"]

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
    return importlib.import_module("preprocessor")


@pytest.fixture
def preprocessor_module(monkeypatch):
    return import_preprocessor(monkeypatch)


def test_clean_text_normalizes_case_punctuation_and_whitespace(preprocessor_module):
    pp = preprocessor_module.Preprocessor({})

    assert pp.clean_text("  Red,   Shirt!!\n") == " red shirt "


def test_remove_stopwords_filters_common_words(preprocessor_module):
    pp = preprocessor_module.Preprocessor({})

    assert pp.remove_stopwords(["the", "red", "shirt", "is", "cotton"]) == [
        "red",
        "shirt",
        "cotton",
    ]


def test_chunk_text_splits_words_into_fixed_size_chunks(preprocessor_module):
    pp = preprocessor_module.Preprocessor({})

    assert pp.chunk_text("one two three four five", chunk_size=2) == [
        "one two",
        "three four",
        "five",
    ]


def test_num_chunked_words_counts_nested_chunks(preprocessor_module):
    pp = preprocessor_module.Preprocessor({})

    chunked_text = [["one two", "three"], ["four five six"]]

    assert pp.num_chunked_words(chunked_text) == 6


def test_preprocess_documents_adds_preprocessed_fields_with_fake_nlp(preprocessor_module):
    item_id_dict = {
        "product-1": {
            "llm_str": "The Red Shirt is Cotton.",
        }
    }
    pp = preprocessor_module.Preprocessor(item_id_dict)

    result = pp.preprocess_documents()

    assert "preproc_llm_str" in result["product-1"]
    assert result["product-1"]["preproc_llm_str"] == [["red shirt cotton"]]
    assert result["product-1"]["word_count"] == 3
