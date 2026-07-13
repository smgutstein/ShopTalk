"""Small data objects shared by the image-case reviewer modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReviewerConfig:
    """Fully parsed settings for one reviewer run.

    Keeping configuration in one immutable object avoids passing a long list of
    unrelated values through the application.  Paths are already absolute by
    the time this object is created.
    """

    project_root: Path
    source_cases: Path
    working_cases: Path
    image_dir: Path
    provenance_file: Path

    max_results: int
    region: str
    safesearch: str

    max_download_bytes: int
    request_timeout_seconds: float
    jpeg_quality: int

    server_name: str
    server_port: int
    share: bool


@dataclass(frozen=True)
class ImageCandidate:
    """One candidate returned by the web-image search provider."""

    image_url: str
    thumbnail_url: str
    title: str
    source_url: str
    width: int | None = None
    height: int | None = None

    def caption(self, index: int) -> str:
        """Return the short caption displayed under a gallery image."""
        dimensions = ""
        if self.width and self.height:
            dimensions = f" — {self.width}×{self.height}"

        title = self.title.strip() or "Untitled image"
        return f"{index + 1}. {title}{dimensions}"


@dataclass(frozen=True)
class SavedImage:
    """Description of a selected image after it has been saved locally."""

    absolute_path: Path
    project_relative_path: str
    width: int
    height: int
    image_format: str
