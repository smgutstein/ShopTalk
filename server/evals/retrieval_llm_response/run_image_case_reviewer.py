"""Command-line entry point for the ShopTalk image-case reviewer.

Nearly all settings are read from an INI file.  The command line intentionally
contains only ``--config`` so configuration is not split between two competing
systems.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ...shoptalk_paths import PROJECT_ROOT
from .image_case_reviewer.app import ImageCaseReviewerApp
from .image_case_reviewer.config import load_reviewer_config
from .image_case_reviewer.downloader import ImageDownloader
from .image_case_reviewer.repository import ImageCaseRepository
from .image_case_reviewer.searcher import ImageSearcher


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "image_case_reviewer"
    / "image_case_reviewer.ini"
)


def parse_args() -> argparse.Namespace:
    """Parse the intentionally small command-line interface.

    Operational settings belong in the INI file.  The sole command-line option
    selects which INI file to use, making it easy to maintain one configuration
    per evaluation dataset without duplicating all options on the command line.
    """
    parser = argparse.ArgumentParser(
        description="Find and select web images for ShopTalk eval cases."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Reviewer INI file (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Construct the utility's objects and launch the Gradio application."""
    args = parse_args()

    # Configuration is parsed once and passed to the objects that need it.
    config = load_reviewer_config(
        args.config,
        project_root=PROJECT_ROOT,
    )

    # Each object owns one major responsibility.  The app coordinates them but
    # does not contain their file, network, or image-processing details.
    repository = ImageCaseRepository(
        source_path=config.source_cases,
        working_path=config.working_cases,
    )
    searcher = ImageSearcher(
        max_results=config.max_results,
        region=config.region,
        safesearch=config.safesearch,
    )
    downloader = ImageDownloader(
        output_dir=config.image_dir,
        project_root=config.project_root,
        max_download_bytes=config.max_download_bytes,
        request_timeout_seconds=config.request_timeout_seconds,
        jpeg_quality=config.jpeg_quality,
    )

    app = ImageCaseReviewerApp(
        config=config,
        repository=repository,
        searcher=searcher,
        downloader=downloader,
    )

    # Report the actual address after INI parsing and any launcher-provided
    # Docker override have both been applied.  This also stays accurate when a
    # different experiment INI chooses a different port.
    print(
        f"Image reviewer binding: "
        f"http://{config.server_name}:{config.server_port}"
    )
    if config.server_name == "0.0.0.0":
        print(
            "Open from host browser: "
            f"http://127.0.0.1:{config.server_port}"
        )
    print()

    app.launch()


if __name__ == "__main__":
    main()
