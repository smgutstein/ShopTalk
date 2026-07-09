"""Backward-compatible wrapper for the unified retrieval/LLM eval module."""

from __future__ import annotations

import argparse
from pathlib import Path

from .retrieval_llm_eval import DEFAULT_EVAL_CONFIG_PATH, generate_main


def parse_args() -> argparse.Namespace:
    """Parse the legacy generator command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Run preset ShopTalk cases and write a hand-editable retrieval/LLM "
            "judgment JSON file."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help=f"Path to retrieval/LLM eval INI file. Default: {DEFAULT_EVAL_CONFIG_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point for the legacy generator module."""
    args = parse_args()
    return generate_main(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
