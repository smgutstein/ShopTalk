"""Persistence for ShopTalk image-bearing evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import atomic_write_text, safe_filename_component


REQUIRED_CASE_FIELDS = {
    "case_id",
    "query_type",
    "category",
    "query",
    "image_path",
    "target_product_id",
    "target_title",
    "expected_available",
    "requires_image",
    "notes",
}

IMAGE_QUERY_TYPES = {"image_only", "text_plus_image"}


class ImageCaseRepository:
    """Own the source and working JSONL files used by the reviewer.

    The source file is treated as the untouched input.  The first run validates
    it and writes a complete working copy.  All later image selections update
    only the working file.

    Records remain dictionaries instead of strict dataclass instances.  That
    preserves the original field order and any additional fields that may be
    added to the evaluation format later.
    """

    def __init__(self, source_path: Path, working_path: Path) -> None:
        self.source_path = source_path.resolve()
        self.working_path = working_path.resolve()
        self.records: list[dict[str, Any]] = []

    def prepare(self) -> list[dict[str, Any]]:
        """Validate the inputs, create the working copy if needed, and load it."""
        if self.source_path == self.working_path:
            raise ValueError(
                "Source and working JSONL paths must be different."
            )

        if self.working_path.exists():
            self.records = self.load(self.working_path)
            return self.records

        # Validate before writing anything.  A malformed source should not
        # leave behind a malformed working copy that blocks the next run.
        source_records = self.load(self.source_path)
        self.records = source_records
        self.save()
        return self.records

    def load(self, path: Path | None = None) -> list[dict[str, Any]]:
        """Read, validate, and return every nonblank JSONL record."""
        path = (path or self.working_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        filename_stems: dict[str, str] = {}

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} of {path} must contain an object."
                    )

                self._validate_record(record, line_number)
                case_id = record["case_id"]

                if case_id in seen_ids:
                    raise ValueError(f"Duplicate case_id in {path}: {case_id!r}")
                seen_ids.add(case_id)

                # Since the case ID becomes the saved filename, detect the
                # uncommon situation where two IDs sanitize to the same stem.
                stem = safe_filename_component(case_id)
                previous = filename_stems.get(stem)
                if previous is not None and previous != case_id:
                    raise ValueError(
                        f"Case IDs {previous!r} and {case_id!r} both map to "
                        f"filename stem {stem!r}."
                    )
                filename_stems[stem] = case_id
                records.append(record)

        if not records:
            raise ValueError(f"No cases found in {path}.")
        return records

    def save(self) -> None:
        """Atomically write the current in-memory records to the working file."""
        text = "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in self.records
        )
        atomic_write_text(self.working_path, text)

    def update_image_path(self, case_id: str, image_path: str) -> None:
        """Update one record's ``image_path`` and immediately persist it."""
        for record in self.records:
            if record["case_id"] == case_id:
                record["image_path"] = image_path
                self.save()
                return
        raise KeyError(f"Unknown case_id: {case_id!r}")

    @staticmethod
    def _validate_record(record: dict[str, Any], line_number: int) -> None:
        """Validate the subset of the ShopTalk case schema needed here."""
        missing = REQUIRED_CASE_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Line {line_number} is missing fields: "
                + ", ".join(sorted(missing))
            )

        case_id = record["case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(
                f"Line {line_number}: case_id must be a non-empty string."
            )

        query_type = record["query_type"]
        if query_type not in IMAGE_QUERY_TYPES:
            raise ValueError(
                f"Line {line_number}, case {case_id!r}: this utility expects "
                f"query_type to be one of {sorted(IMAGE_QUERY_TYPES)}, got "
                f"{query_type!r}."
            )

        query = record["query"]
        if query_type == "text_plus_image":
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Line {line_number}, case {case_id!r}: text_plus_image "
                    "cases require a non-empty query."
                )

        image_path = record["image_path"]
        if image_path is not None and not isinstance(image_path, str):
            raise ValueError(
                f"Line {line_number}, case {case_id!r}: image_path must be "
                "a string or null."
            )
