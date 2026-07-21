from pathlib import Path

import pytest

from server.evals.search_decision.eval_search_decision import (
    DEFAULT_EVAL_CONFIG_PATH,
    load_eval_config,
    parse_args,
)


def test_cli_accepts_only_optional_config_path():
    args = parse_args([])
    assert args.config == DEFAULT_EVAL_CONFIG_PATH

    custom = Path("custom_search_eval.ini")
    args = parse_args(["--config", str(custom)])
    assert args.config == custom

    with pytest.raises(SystemExit):
        parse_args(["--limit", "5"])


def test_load_eval_config_reads_complete_run_definition(tmp_path):
    config_path = tmp_path / "search_eval.ini"
    config_path.write_text(
        """
[model]
model_name = test-model
temperature = 0.25

[eval]
cases = cases/example.jsonl
limit = 12
categories = boundary, acknowledgement

[output]
output_dir = results/search
output_prefix = trial
detail_level = failures
""".strip(),
        encoding="utf-8",
    )

    config = load_eval_config(config_path)

    assert config.config_path == config_path
    assert config.model_name == "test-model"
    assert config.temperature == 0.25
    assert config.cases == Path("cases/example.jsonl")
    assert config.limit == 12
    assert config.category == ["boundary", "acknowledgement"]
    assert config.output_dir == Path("results/search")
    assert config.output_prefix == "trial"
    assert config.detail_level == "failures"


def test_load_eval_config_supports_all_cases_and_categories(tmp_path):
    config_path = tmp_path / "search_eval.ini"
    config_path.write_text(
        """
[model]
model_name = test-model
temperature = 0.0

[eval]
cases = cases/example.jsonl
limit = none
categories =

[output]
output_dir = results/search
output_prefix = trial
detail_level = all
""".strip(),
        encoding="utf-8",
    )

    config = load_eval_config(config_path)

    assert config.limit is None
    assert config.category is None
    assert config.detail_level == "all"
