from pathlib import Path

import pytest

from server.evals.product_response_decision.eval_product_response_decision import (
    DEFAULT_CONFIG_PATH,
    load_eval_config,
    parse_args,
)


def test_parse_args_defaults_to_evaluator_config():
    assert parse_args([]).config == DEFAULT_CONFIG_PATH


def test_parse_args_accepts_only_config_selector():
    selected = Path("custom.ini")
    assert parse_args(["--config", str(selected)]).config == selected

    with pytest.raises(SystemExit):
        parse_args(["--limit", "10"])


def test_load_eval_config_reads_complete_run_definition(tmp_path):
    config_path = tmp_path / "eval.ini"
    config_path.write_text(
        """
[model]
model_name = test-model
temperature = 0.25

[eval]
cases = cases.jsonl
limit = 7

[output]
output_dir = reports
output_prefix = product_response_test
output_csv = reports/details.csv
""".strip(),
        encoding="utf-8",
    )

    config = load_eval_config(config_path)

    assert config.model_name == "test-model"
    assert config.temperature == 0.25
    assert config.cases_path == Path("cases.jsonl")
    assert config.limit == 7
    assert config.output_dir == Path("reports")
    assert config.output_prefix == "product_response_test"
    assert config.output_csv == Path("reports/details.csv")


def test_load_eval_config_supports_unlimited_run_and_no_csv(tmp_path):
    config_path = tmp_path / "eval.ini"
    config_path.write_text(
        """
[model]
model_name = test-model
temperature = 0.0

[eval]
cases = cases.jsonl
limit = none

[output]
output_dir = reports
output_prefix = product_response_test
output_csv =
""".strip(),
        encoding="utf-8",
    )

    config = load_eval_config(config_path)

    assert config.limit is None
    assert config.output_csv is None
