from types import SimpleNamespace

import pytest

from server.recommender_core.config import (
    DEFAULT_APP_LLM_MODEL,
    DEFAULT_APP_LLM_TEMPERATURE,
    DEFAULT_EVAL_LLM_MODEL,
    DEFAULT_EVAL_LLM_TEMPERATURE,
    DEFAULT_SERVER_NAME,
    DEFAULT_SERVER_PORT,
    RecommenderConfig,
    load_shoptalk_config,
)


def make_args(**overrides):
    defaults = {
        "debug": False,
        "cpu": False,
        "config": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_load_shoptalk_config_returns_defaults_for_missing_file(tmp_path):
    config = load_shoptalk_config(tmp_path / "does_not_exist.ini")

    assert config.app_model_name == DEFAULT_APP_LLM_MODEL
    assert config.app_temperature == pytest.approx(DEFAULT_APP_LLM_TEMPERATURE)
    assert config.eval_model_name == DEFAULT_EVAL_LLM_MODEL
    assert config.eval_temperature == pytest.approx(DEFAULT_EVAL_LLM_TEMPERATURE)
    assert config.server_name == DEFAULT_SERVER_NAME
    assert config.server_port == DEFAULT_SERVER_PORT


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
    assert config.top_k == 10
    assert config.server_name == DEFAULT_SERVER_NAME


def test_load_shoptalk_config_reads_all_runtime_settings(tmp_path):
    config_path = tmp_path / "shoptalk_config.ini"
    config_path.write_text(
        "[llm]\n"
        "model_name = gpt-app\n"
        "temperature = 0.25\n"
        "\n"
        "[app]\n"
        "personality = 3\n"
        "\n"
        "[retrieval]\n"
        "vector_db_output_dir = artifacts/custom_db\n"
        "vector_backend = faiss\n"
        "top_k = 7\n"
        "\n"
        "[data]\n"
        "product_blurbs = data/blurbs.json\n"
        "images_csv = data/images.csv\n"
        "\n"
        "[server]\n"
        "server_name = 0.0.0.0\n"
        "server_port = 7861\n"
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
    assert config.personality_index == 3
    assert str(config.vector_db_output_dir) == "artifacts/custom_db"
    assert config.vector_backend == "faiss"
    assert config.top_k == 7
    assert str(config.product_blurbs_path) == "data/blurbs.json"
    assert str(config.images_csv_path) == "data/images.csv"
    assert config.server_name == "0.0.0.0"
    assert config.server_port == 7861


def test_recommender_config_uses_ini_settings_and_cli_runtime_overrides(tmp_path):
    config_path = tmp_path / "shoptalk_config.ini"
    config_path.write_text(
        "[llm]\n"
        "model_name = gpt-from-file\n"
        "temperature = 0.33\n"
        "[app]\n"
        "personality = 4\n"
        "[retrieval]\n"
        "vector_db_output_dir = artifacts/test_vector_db\n"
        "vector_backend = faiss\n"
        "top_k = 7\n"
        "[data]\n"
        "product_blurbs = tests/fixtures/product_blurbs.json\n"
        "images_csv = tests/fixtures/images.csv\n",
        encoding="utf-8",
    )

    config = RecommenderConfig.from_args(
        make_args(config=config_path, debug=True, cpu=True)
    )

    assert config.personality_index == 4
    assert config.debug is True
    assert config.force_cpu is True
    assert config.model_name == "gpt-from-file"
    assert config.temperature == pytest.approx(0.33)
    assert str(config.vector_db_output_dir) == "artifacts/test_vector_db"
    assert config.vector_backend == "faiss"
    assert config.top_k == 7
    assert str(config.blurbs_path) == "tests/fixtures/product_blurbs.json"
    assert str(config.images_csv_path) == "tests/fixtures/images.csv"


def test_resolve_server_name_auto_uses_localhost_outside_docker(monkeypatch):
    from server.recommender_core import config as config_module

    monkeypatch.setattr(config_module, "running_in_docker", lambda: False)

    assert config_module.resolve_server_name("auto") == "127.0.0.1"


def test_resolve_server_name_auto_binds_all_interfaces_in_docker(monkeypatch):
    from server.recommender_core import config as config_module

    monkeypatch.setattr(config_module, "running_in_docker", lambda: True)

    assert config_module.resolve_server_name("AUTO") == "0.0.0.0"


def test_resolve_server_name_preserves_explicit_address(monkeypatch):
    from server.recommender_core import config as config_module

    monkeypatch.setattr(config_module, "running_in_docker", lambda: True)

    assert config_module.resolve_server_name(" 127.0.0.1 ") == "127.0.0.1"
