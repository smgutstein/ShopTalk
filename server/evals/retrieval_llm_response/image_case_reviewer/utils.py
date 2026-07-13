"""Stateless helpers used by more than one reviewer module."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a configured path relative to the ShopTalk repository root.

    INI files are usually committed to a repository, so project-relative paths
    are more portable than machine-specific absolute paths.  Absolute paths are
    still accepted when a one-off local setup needs them.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def project_relative_string(path: Path, project_root: Path) -> str:
    """Return a POSIX-style path relative to the ShopTalk project root.

    ``image_path`` values stored in the working JSONL should remain portable.
    Therefore, downloaded images are required to live inside the repository.
    """
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()

    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Saved image is outside the ShopTalk repository: {resolved_path}"
        ) from exc


def safe_filename_component(value: str) -> str:
    """Convert a case ID into a conservative filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed_case"


def safe_positive_int(value: Any) -> int | None:
    """Convert optional search-result dimension metadata without failing."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def atomic_write_text(path: Path, text: str) -> None:
    """Write a text file without exposing a partially written final file.

    The complete content is written to a temporary file in the same directory.
    ``os.replace`` then swaps it into place.  This is a modest but useful guard
    for JSON and JSONL files that are rewritten after each selected image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
