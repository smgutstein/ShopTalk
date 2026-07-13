"""INI loading and validation for the ShopTalk image-case reviewer."""

from __future__ import annotations

import configparser
import os
from pathlib import Path

from .models import ReviewerConfig
from .utils import resolve_project_path


class ReviewerConfigError(ValueError):
    """Raised when the reviewer INI is missing or internally inconsistent."""


def _required_value(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> str:
    """Read a required nonblank INI value with a useful error message."""
    if not parser.has_section(section):
        raise ReviewerConfigError(f"Missing INI section [{section}].")
    if not parser.has_option(section, option):
        raise ReviewerConfigError(f"Missing INI option [{section}] {option}.")

    value = parser.get(section, option).strip()
    if not value:
        raise ReviewerConfigError(f"Blank INI value for [{section}] {option}.")
    return value


def _positive_int(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> int:
    value = int(_required_value(parser, section, option))
    if value <= 0:
        raise ReviewerConfigError(f"[{section}] {option} must be positive.")
    return value


def _positive_float(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> float:
    value = float(_required_value(parser, section, option))
    if value <= 0:
        raise ReviewerConfigError(f"[{section}] {option} must be positive.")
    return value


def load_reviewer_config(
    config_path: Path,
    *,
    project_root: Path,
) -> ReviewerConfig:
    """Parse ``config_path`` into one validated :class:`ReviewerConfig`.

    Relative file paths are interpreted from the ShopTalk project root, not
    from the directory where the shell happens to be running.  This mirrors the
    repository-oriented path conventions used elsewhere in ShopTalk.
    """
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Reviewer INI file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    source_cases = resolve_project_path(
        _required_value(parser, "paths", "source_cases"), project_root
    )
    working_cases = resolve_project_path(
        _required_value(parser, "paths", "working_cases"), project_root
    )
    image_dir = resolve_project_path(
        _required_value(parser, "paths", "image_dir"), project_root
    )
    provenance_file = resolve_project_path(
        _required_value(parser, "paths", "provenance_file"), project_root
    )

    max_results = _positive_int(parser, "search", "max_results")
    region = _required_value(parser, "search", "region")
    safesearch = _required_value(parser, "search", "safesearch").lower()
    if safesearch not in {"on", "moderate", "off"}:
        raise ReviewerConfigError(
            "[search] safesearch must be one of: on, moderate, off."
        )

    max_download_mb = _positive_int(parser, "download", "max_download_mb")
    request_timeout_seconds = _positive_float(
        parser, "download", "request_timeout_seconds"
    )
    jpeg_quality = _positive_int(parser, "download", "jpeg_quality")
    if jpeg_quality > 100:
        raise ReviewerConfigError("[download] jpeg_quality cannot exceed 100.")

    # The INI provides ordinary host defaults.  The repository launcher may
    # override only the bind address through an environment variable when it
    # detects that it is running inside Docker.  This mirrors
    # ``run_ShopTalk_gradio.sh`` and keeps Docker-specific behavior out of the
    # Python modules.
    configured_name = _required_value(parser, "server", "server_name")
    server_name = os.environ.get(
        "IMAGE_REVIEWER_SERVER_NAME",
        configured_name,
    ).strip()
    if not server_name:
        raise ReviewerConfigError(
            "The image-reviewer server name cannot be blank."
        )

    server_port = _positive_int(parser, "server", "server_port")
    share = parser.getboolean("server", "share")

    if source_cases == working_cases:
        raise ReviewerConfigError(
            "Source and working JSONL paths must be different."
        )

    return ReviewerConfig(
        project_root=project_root.resolve(),
        source_cases=source_cases,
        working_cases=working_cases,
        image_dir=image_dir,
        provenance_file=provenance_file,
        max_results=max_results,
        region=region,
        safesearch=safesearch,
        max_download_bytes=max_download_mb * 1024 * 1024,
        request_timeout_seconds=request_timeout_seconds,
        jpeg_quality=jpeg_quality,
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
