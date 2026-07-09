"""Backward-compatible wrapper for the unified retrieval/LLM eval module."""

from __future__ import annotations

import argparse
from pathlib import Path

from .retrieval_llm_eval import DEFAULT_EVAL_CONFIG_PATH, score_main


def parse_args() -> argparse.Namespace:
    """Parse the legacy scorer command line."""
    parser = argparse.ArgumentParser(
        description="Score a hand-edited ShopTalk retrieval/LLM judgment JSON file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help=f"Path to retrieval/LLM eval INI file. Default: {DEFAULT_EVAL_CONFIG_PATH}",
    )
    parser.add_argument(
        "--allow-unjudged",
        action="store_true",
        help=(
            "Compute partial metrics even when judgment fields are still null. "
            "This overrides [score] allow_unjudged=false in the INI."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point for the legacy scorer module."""
    args = parse_args()
    return score_main(args.config, allow_unjudged=args.allow_unjudged)


if __name__ == "__main__":
    raise SystemExit(main())
