from types import SimpleNamespace

import pytest

from server.recommender_core.config import (
    DEFAULT_APP_LLM_MODEL,
    DEFAULT_APP_LLM_TEMPERATURE,
    DEFAULT_EVAL_LLM_MODEL,
    DEFAULT_EVAL_LLM_TEMPERATURE,
    RecommenderConfig,
    load_shoptalk_config,
)


def make_args(**overrides):
    defaults = {
        "personality": -1,
        "debug": False,
        "cpu": False,
        "config": None,
        "model": None,
        "temperature": None,
        "vector_db_output_dir": "artifacts/vector_db",
        "vector_backend": "faiss",
        "top_k": 10,
        "product_blurbs": "EDA/product_blurbs/combined_blurb_dict.json",
        "images_csv": "images.csv",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_load_shoptalk_config_returns_defaults_for_missing_file(tmp_path):
    config = load_shoptalk_config(tmp_path / "does_not_exist.ini")

    assert config.app_model_name == DEFAULT_APP_LLM_MODEL
    assert config.app_temperature == pytest.approx(DEFAULT_APP_LLM_TEMPERATURE)
    assert config.eval_model_name == DEFAULT_EVAL_LLM_MODEL
    assert config.eval_temperature == pytest.approx(DEFAULT_EVAL_LLM_TEMPERATURE)


def test_load_shoptalk_config_uses_defaults_for_missing_sections_and_keys(tmp_path):
    config_path = tmp_path / "shoptalk_config.ini"
    config_path.write_text(
        "[llm]\n"
        "model_name = gpt-custom-app\n",
        encoding="utf-8",
    )

    config = load_shoptalk_config(config_path)

    assert config.app_model_name == "gpt-custom-app"
    assert config.app_temperature == pytest.approx(DEFAULT_APP_LLM_TEMPERATURE)
    assert config.eval_model_name == DEFAULT_EVAL_LLM_MODEL
    assert config.eval_temperature == pytest.approx(DEFAULT_EVAL_LLM_TEMPERATURE)


def test_load_shoptalk_config_reads_app_and_eval_settings(tmp_path):
    config_path = tmp_path / "shoptalk_config.ini"
    config_path.write_text(
        "[llm]\n"
        "model_name = gpt-app\n"
        "temperature = 0.25\n"
        "\n"
        "[evals]\n"
        "model_name = gpt-eval\n"
        "temperature = 0.0\n",
        encoding="utf-8",
    )

    config = load_shoptalk_config(config_path)

    assert config.app_model_name == "gpt-app"
    assert config.app_temperature == pytest.approx(0.25)
    assert config.eval_model_name == "gpt-eval"
    assert config.eval_temperature == pytest.approx(0.0)


def test_recommender_config_uses_file_model_settings_when_cli_does_not_override(tmp_path):
    config_path = tmp_path / "shoptalk_config.ini"
    config_path.write_text(
        "[llm]\n"
        "model_name = gpt-from-file\n"
        "temperature = 0.33\n",
        encoding="utf-8",
    )

    config = RecommenderConfig.from_args(make_args(config=config_path))

    assert config.model_name == "gpt-from-file"
    assert config.temperature == pytest.approx(0.33)


def test_recommender_config_cli_model_settings_override_config_file(tmp_path):
    config_path = tmp_path / "shoptalk_config.ini"
    config_path.write_text(
        "[llm]\n"
        "model_name = gpt-from-file\n"
        "temperature = 0.33\n",
        encoding="utf-8",
    )

    config = RecommenderConfig.from_args(
        make_args(config=config_path, model="gpt-from-cli", temperature=0.7)
    )

    assert config.model_name == "gpt-from-cli"
    assert config.temperature == pytest.approx(0.7)
