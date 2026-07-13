"""Download, validate, normalize, and save selected web images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

from .models import ImageCandidate, SavedImage
from .utils import project_relative_string, safe_filename_component


DOWNLOAD_CHUNK_BYTES = 128 * 1024
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}


class ImageDownloader:
    """Own the web-download and local-image-storage workflow.

    Search results can point to JPEG, PNG, WebP, GIF, or poorly labelled data.
    This class converts the selected image into a predictable local JPEG or PNG
    before returning it to the rest of the application.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        project_root: Path,
        max_download_bytes: int,
        request_timeout_seconds: float,
        jpeg_quality: int,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.project_root = project_root.resolve()
        self.max_download_bytes = max_download_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self.jpeg_quality = jpeg_quality

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_candidate(
        self,
        candidate: ImageCandidate,
        *,
        case_id: str,
    ) -> SavedImage:
        """Download and save one selected candidate for ``case_id``."""
        downloaded_path = self._download_to_temporary_file(candidate.image_url)

        try:
            saved_image = self._normalize_and_install(downloaded_path, case_id)
        finally:
            downloaded_path.unlink(missing_ok=True)

        # A replacement can change from JPEG to PNG or vice versa.  Remove the
        # old alternate extension only after the new file exists successfully.
        self._remove_old_variants(case_id, keep=saved_image.absolute_path)
        return saved_image

    def _download_to_temporary_file(self, image_url: str) -> Path:
        """Stream a remote file to disk while enforcing a practical size cap."""
        response = requests.get(
            image_url,
            headers=HTTP_HEADERS,
            timeout=self.request_timeout_seconds,
            stream=True,
        )
        response.raise_for_status()

        declared_length = response.headers.get("Content-Length")
        if declared_length:
            try:
                declared_bytes = int(declared_length)
            except ValueError:
                declared_bytes = 0
            if declared_bytes > self.max_download_bytes:
                raise ValueError(
                    "Image exceeds the configured download-size limit."
                )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".image_download_",
            suffix=".tmp",
            dir=self.output_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)

        bytes_written = 0
        try:
            with temporary_path.open("wb") as handle:
                for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > self.max_download_bytes:
                        raise ValueError(
                            "Downloaded image exceeded the configured size limit."
                        )
                    handle.write(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        return temporary_path

    def _normalize_and_install(
        self,
        downloaded_path: Path,
        case_id: str,
    ) -> SavedImage:
        """Decode the download and save a stable JPEG or PNG representation."""
        stem = safe_filename_component(case_id)

        try:
            with Image.open(downloaded_path) as image:
                # For animated GIF/WebP files, a single representative frame is
                # sufficient for these retrieval evaluation examples.
                image.seek(0)
                image.load()

                width, height = image.size
                has_transparency = (
                    image.mode in {"RGBA", "LA"}
                    or "transparency" in image.info
                )

                if has_transparency:
                    output_path = self.output_dir / f"{stem}.png"
                    normalized = image.convert("RGBA")
                    normalized.save(output_path, format="PNG")
                    image_format = "PNG"
                else:
                    output_path = self.output_dir / f"{stem}.jpg"
                    normalized = image.convert("RGB")
                    normalized.save(
                        output_path,
                        format="JPEG",
                        quality=self.jpeg_quality,
                    )
                    image_format = "JPEG"
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The downloaded file is not a readable image.") from exc

        return SavedImage(
            absolute_path=output_path.resolve(),
            project_relative_path=project_relative_string(
                output_path, self.project_root
            ),
            width=width,
            height=height,
            image_format=image_format,
        )

    def _remove_old_variants(self, case_id: str, *, keep: Path) -> None:
        """Remove an obsolete alternate extension for the same case ID."""
        stem = safe_filename_component(case_id)
        for suffix in (".jpg", ".png"):
            candidate_path = self.output_dir / f"{stem}{suffix}"
            if candidate_path.resolve() != keep.resolve():
                candidate_path.unlink(missing_ok=True)
