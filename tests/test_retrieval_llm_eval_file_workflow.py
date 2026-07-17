from __future__ import annotations

from pathlib import Path

import pytest

from server.evals.retrieval_llm_response.retrieval_llm_eval import (
    default_scored_output_path,
    latest_reviewed_judgment_path,
    load_eval_args,
    load_score_args,
    next_linked_judgment_paths,
    reviewed_path_for_generated,
)


def _write_config(path: Path, generated: Path, reviewed: Path, scored: Path) -> None:
    path.write_text(
        f"""
[eval]
cases = cases.jsonl
output = auto
output_dir = {generated}
reviewed_output_dir = {reviewed}
output_prefix = sample_judgments
limit = none

[runtime]
shoptalk_config = shoptalk_config.ini
model = config
temperature = config

[artifacts]
vector_db_output_dir = artifacts/vector_db
vector_backend = faiss
top_k = 5
product_blurbs = blurbs.json
images_csv = images.csv

[score]
judgments = auto
output = auto
output_dir = {scored}
allow_unjudged = false
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_next_linked_judgment_paths_reserves_same_number(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    reviewed = tmp_path / "reviewed"
    generated.mkdir()
    reviewed.mkdir()
    (reviewed / "sample_001_reviewed.json").write_text("{}", encoding="utf-8")

    generated_path, reviewed_path = next_linked_judgment_paths(
        generated,
        reviewed,
        prefix="sample",
    )

    assert generated_path == generated / "sample_002.json"
    assert reviewed_path == reviewed / "sample_002_reviewed.json"


def test_reviewed_path_is_derived_from_generated_name(tmp_path: Path) -> None:
    generated = tmp_path / "generated" / "sample_007.json"
    reviewed = tmp_path / "reviewed"

    assert reviewed_path_for_generated(generated, reviewed) == (
        reviewed / "sample_007_reviewed.json"
    )


def test_latest_reviewed_judgment_path_uses_highest_run_number(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed"
    reviewed.mkdir()
    for name in (
        "sample_002_reviewed.json",
        "sample_010_reviewed.json",
        "other_999_reviewed.json",
        "sample_notes.txt",
    ):
        (reviewed / name).write_text("{}", encoding="utf-8")

    assert latest_reviewed_judgment_path(reviewed, prefix="sample") == (
        reviewed / "sample_010_reviewed.json"
    )


def test_latest_reviewed_judgment_path_fails_clearly_when_empty(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed"
    reviewed.mkdir()

    with pytest.raises(FileNotFoundError, match="sample_NNN_reviewed.json"):
        latest_reviewed_judgment_path(reviewed, prefix="sample")


def test_config_links_generate_review_and_score_paths(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    reviewed = tmp_path / "reviewed"
    scored = tmp_path / "scored"
    reviewed.mkdir()
    newest = reviewed / "sample_judgments_003_reviewed.json"
    newest.write_text("{}", encoding="utf-8")
    config = tmp_path / "eval.ini"
    _write_config(config, generated, reviewed, scored)

    eval_args = load_eval_args(config)
    score_args = load_score_args(config)

    assert eval_args.output_dir == generated
    assert eval_args.reviewed_output_dir == reviewed
    assert score_args.judgments == newest
    assert score_args.output_dir == scored


def test_explicit_score_judgment_path_still_overrides_auto(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    reviewed = tmp_path / "reviewed"
    scored = tmp_path / "scored"
    explicit = tmp_path / "chosen.json"
    config = tmp_path / "eval.ini"
    _write_config(config, generated, reviewed, scored)
    text = config.read_text(encoding="utf-8").replace(
        "judgments = auto", f"judgments = {explicit}"
    )
    config.write_text(text, encoding="utf-8")

    assert load_score_args(config).judgments == explicit


def test_scored_report_name_identifies_reviewed_input(tmp_path: Path) -> None:
    judgment = tmp_path / "reviewed" / "sample_004_reviewed.json"

    first = default_scored_output_path(tmp_path / "scored", judgment)
    first.touch()
    second = default_scored_output_path(tmp_path / "scored", judgment)

    assert first.name == "sample_004_reviewed_scored_001.txt"
    assert second.name == "sample_004_reviewed_scored_002.txt"


def test_parse_args_accepts_positional_config_path(tmp_path):
    """The preferred CLI accepts the INI as the sole positional argument."""
    from server.evals.retrieval_llm_response.retrieval_llm_eval import parse_args

    config = tmp_path / "example.ini"

    generate_args = parse_args(["generate", str(config)])
    score_args = parse_args(["score", str(config)])

    assert generate_args.config == config
    assert score_args.config == config


def test_parse_args_still_accepts_config_option(tmp_path):
    """Keep -c/--config working so existing automation is not broken."""
    from server.evals.retrieval_llm_response.retrieval_llm_eval import parse_args

    config = tmp_path / "example.ini"

    args = parse_args(["generate", "-c", str(config)])

    assert args.config == config


def test_parse_args_rejects_conflicting_config_paths(tmp_path):
    """Two different config paths are ambiguous and must fail loudly."""
    from server.evals.retrieval_llm_response.retrieval_llm_eval import parse_args

    positional = tmp_path / "positional.ini"
    optional = tmp_path / "optional.ini"

    with pytest.raises(SystemExit) as exc_info:
        parse_args(["generate", str(positional), "-c", str(optional)])

    assert exc_info.value.code == 2
