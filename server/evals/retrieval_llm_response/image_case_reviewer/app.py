"""Gradio user interface for reviewing ShopTalk image-bearing cases."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gradio as gr

from .downloader import ImageDownloader
from .models import ImageCandidate, ReviewerConfig, SavedImage
from .repository import ImageCaseRepository
from .searcher import ImageSearcher
from .utils import atomic_write_text


class ImageCaseReviewerApp:
    """Coordinate the repository, searcher, downloader, and Gradio UI.

    This class deliberately contains UI behavior only.  It asks the other
    classes to perform persistence, search, and download work instead of mixing
    those details into Gradio callbacks.
    """

    def __init__(
        self,
        *,
        config: ReviewerConfig,
        repository: ImageCaseRepository,
        searcher: ImageSearcher,
        downloader: ImageDownloader,
    ) -> None:
        self.config = config
        self.repository = repository
        self.searcher = searcher
        self.downloader = downloader

        # ``prepare`` validates both the source and any existing working copy.
        # It also creates the working copy on the first run.
        self.records = self.repository.prepare()

    def build(self) -> gr.Blocks:
        """Build and return the complete Gradio application."""
        # Resume at the first case that does not yet have an assigned image.
        # If every case is complete, start at the first case so the reviewer
        # still opens in a valid, inspectable state.
        initial_values = self._case_values(self._first_incomplete_index())

        with gr.Blocks(title="ShopTalk Image Case Reviewer") as demo:
            # Gradio State carries non-visible data between callbacks.  Search
            # results are serialized dictionaries because those are easier for
            # Gradio to preserve than custom dataclass objects.
            candidates_state = gr.State([])
            index_state = gr.State(initial_values[0])
            selected_index_state = gr.State(-1)

            gr.Markdown("# ShopTalk Image Case Reviewer")
            gr.Markdown(
                "Review one evaluation case at a time, search for candidate "
                "images, compare them, and save one selected image. The "
                "original source JSONL is never modified."
            )

            with gr.Row():
                source_box = gr.Textbox(
                    label="Source JSONL (unchanged)",
                    value=str(self.config.source_cases),
                    interactive=False,
                )
                working_box = gr.Textbox(
                    label="Working JSONL (updated)",
                    value=str(self.config.working_cases),
                    interactive=False,
                )

            case_summary = gr.Markdown(initial_values[1])
            progress_summary = gr.Markdown(initial_values[2])

            # Previous/Next support the normal sequential workflow.  The
            # slider provides a quick way to jump directly to any case while
            # remaining synchronized with button-based navigation.
            with gr.Row():
                previous_button = gr.Button("← Previous")
                case_slider = gr.Slider(
                    minimum=1,
                    maximum=len(self.records),
                    step=1,
                    value=initial_values[3],
                    label="Go to case",
                    interactive=True,
                )
                next_button = gr.Button("Next →")

            user_query = gr.Textbox(
                label="Original user query",
                value=initial_values[4],
                lines=3,
                interactive=False,
            )

            # The editable search text is separate from the original query.
            # Natural-language user requests often benefit from a shorter,
            # product-oriented search phrase.
            with gr.Row():
                search_text = gr.Textbox(
                    label="Web image search",
                    value=initial_values[5],
                    scale=5,
                )
                search_button = gr.Button("Search", variant="primary", scale=1)

            gallery = gr.Gallery(
                label="Candidate images — click one to select it",
                value=[],
                columns=4,
                height="auto",
                object_fit="contain",
                allow_preview=True,
            )

            selected_preview = gr.Image(
                label="Selected candidate",
                value=None,
                height=420,
                interactive=False,
            )
            selected_details = gr.Markdown("No candidate selected.")

            save_button = gr.Button(
                "Download selected image and update working JSONL",
                variant="primary",
            )
            saved_file = gr.File(label="Most recently saved image")
            status = gr.Markdown()

            # These outputs fully reset the current case display.  In
            # particular, navigation clears the previous case's search results
            # and candidate selection so one case cannot accidentally save an
            # image chosen for another.
            case_outputs = [
                index_state,
                case_summary,
                progress_summary,
                case_slider,
                user_query,
                search_text,
                gallery,
                candidates_state,
                selected_index_state,
                selected_preview,
                selected_details,
            ]

            previous_button.click(
                fn=lambda index: self._move_case(index, -1),
                inputs=index_state,
                outputs=case_outputs,
            )

            next_button.click(
                fn=lambda index: self._move_case(index, 1),
                inputs=index_state,
                outputs=case_outputs,
            )

            case_slider.release(
                fn=self._jump_to_case,
                inputs=case_slider,
                outputs=case_outputs,
            )

            search_outputs = [
                candidates_state,
                gallery,
                selected_index_state,
                selected_preview,
                selected_details,
                status,
            ]

            search_button.click(
                fn=self._perform_search,
                inputs=search_text,
                outputs=search_outputs,
            )
            search_text.submit(
                fn=self._perform_search,
                inputs=search_text,
                outputs=search_outputs,
            )

            gallery.select(
                fn=self._choose_candidate,
                inputs=candidates_state,
                outputs=[
                    selected_index_state,
                    selected_preview,
                    selected_details,
                ],
            )

            save_button.click(
                fn=self._save_selection,
                inputs=[index_state, selected_index_state, candidates_state],
                outputs=[case_summary, progress_summary, saved_file, status],
            )

        return demo

    def launch(self) -> None:
        """Build and launch Gradio using server settings from the INI file."""
        self.build().launch(
            server_name=self.config.server_name,
            server_port=self.config.server_port,
            share=self.config.share,
        )

    def _case_values(self, index: int) -> tuple[Any, ...]:
        """Return every UI value required to display one case."""
        index = max(0, min(int(index), len(self.records) - 1))
        record = self.records[index]
        query = record.get("query") or ""

        return (
            index,
            self._case_markdown(record, index),
            self._progress_markdown(index),
            index + 1,  # slider uses human-friendly one-based case numbers
            query,
            query,
            [],       # gallery
            [],       # candidates state
            -1,       # selected candidate index
            None,     # selected preview
            "No candidate selected.",
        )

    def _first_incomplete_index(self) -> int:
        """Return the first case not yet saved by this reviewer.

        The source evaluation cases may already contain non-empty
        ``image_path`` values.  Those paths describe the intended evaluation
        inputs; they do not prove that this reviewer has processed the case.

        The provenance manifest is therefore the authoritative record of
        reviewer-completed cases.  Each successful save adds one entry keyed
        by ``case_id``.  On restart, the first case without such an entry is
        selected.  If every case has provenance, index 0 is returned so the UI
        still opens in a valid review state.
        """
        completed_case_ids = self._completed_case_ids()

        for index, record in enumerate(self.records):
            if record["case_id"] not in completed_case_ids:
                return index

        return 0

    def _move_case(self, index: int, amount: int) -> tuple[Any, ...]:
        """Move relative to the current zero-based case index."""
        return self._case_values(index + amount)

    def _jump_to_case(self, case_number: int | float) -> tuple[Any, ...]:
        """Load the one-based case number selected with the slider."""
        return self._case_values(int(case_number) - 1)

    def _perform_search(self, query: str) -> tuple[Any, ...]:
        """Run one configured image search and populate the gallery."""
        try:
            candidates = self.searcher.search(query)
        except Exception as exc:
            raise gr.Error(f"Image search failed: {exc}") from exc

        if not candidates:
            raise gr.Error("No image results were returned.")

        serialized = [asdict(candidate) for candidate in candidates]
        gallery_items = [
            (candidate.thumbnail_url, candidate.caption(index))
            for index, candidate in enumerate(candidates)
        ]

        return (
            serialized,
            gallery_items,
            -1,
            None,
            "No candidate selected.",
            f"Found {len(candidates)} candidate images.",
        )

    @staticmethod
    def _choose_candidate(
        candidates_data: list[dict[str, Any]],
        event: gr.SelectData,
    ) -> tuple[int, str, str]:
        """Convert a gallery click into the active candidate selection."""
        if not candidates_data:
            raise gr.Error("Search for images first.")

        index = int(event.index)
        if index < 0 or index >= len(candidates_data):
            raise gr.Error("The selected gallery index is invalid.")

        candidate = ImageCandidate(**candidates_data[index])
        details = (
            f"**Selected candidate {index + 1}:** "
            f"{candidate.title or 'Untitled image'}  \n"
            f"**Source page:** {candidate.source_url or 'Unknown'}  \n"
            f"**Image URL:** {candidate.image_url}"
        )
        return index, candidate.image_url, details

    def _save_selection(
        self,
        case_index: int,
        selected_index: int,
        candidates_data: list[dict[str, Any]],
    ) -> tuple[str, str, str, str]:
        """Save the selected image and update the current working record."""
        if selected_index < 0:
            raise gr.Error("Select an image before saving.")
        if selected_index >= len(candidates_data):
            raise gr.Error("The selected result is no longer available.")

        case_index = max(0, min(int(case_index), len(self.records) - 1))
        record = self.records[case_index]
        candidate = ImageCandidate(**candidates_data[selected_index])

        try:
            saved = self.downloader.save_candidate(
                candidate,
                case_id=record["case_id"],
            )
            self.repository.update_image_path(
                record["case_id"],
                saved.project_relative_path,
            )
            self._update_provenance(record, candidate, saved)
        except Exception as exc:
            raise gr.Error(f"Could not save the selected image: {exc}") from exc

        status = (
            f"Saved `{saved.absolute_path}` and updated case "
            f"`{record['case_id']}` in the working JSONL."
        )

        return (
            self._case_markdown(record, case_index),
            self._progress_markdown(case_index),
            str(saved.absolute_path),
            status,
        )

    def _load_provenance(self) -> dict[str, Any]:
        """Load and validate the reviewer provenance manifest.

        The manifest is the authoritative record of cases completed through
        this UI.  Keeping that status separate from ``image_path`` matters
        because source evaluation files may already contain planned or
        placeholder image paths before review begins.
        """
        path = self.config.provenance_file
        if not path.exists():
            return {}

        try:
            provenance = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid provenance JSON: {path}") from exc

        if not isinstance(provenance, dict):
            raise ValueError("The provenance file must contain a JSON object.")

        return provenance

    def _completed_case_ids(self) -> set[str]:
        """Return case IDs with successful reviewer provenance entries."""
        return set(self._load_provenance())

    def _update_provenance(
        self,
        record: dict[str, Any],
        candidate: ImageCandidate,
        saved: SavedImage,
    ) -> None:
        """Store source details outside the evaluator's strict JSONL schema."""
        path = self.config.provenance_file
        provenance = self._load_provenance()

        provenance[record["case_id"]] = {
            "image_path": saved.project_relative_path,
            "image_url": candidate.image_url,
            "source_url": candidate.source_url,
            "search_result_title": candidate.title,
            "width": saved.width,
            "height": saved.height,
            "format": saved.image_format,
        }

        atomic_write_text(
            path,
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        )


    def _progress_markdown(self, index: int) -> str:
        """Render current-position and completion information.

        A case counts as completed when it has an entry in the provenance
        manifest.  ``image_path`` cannot be used for this purpose because the
        source evaluation file may already contain placeholder or planned
        image paths for every case.
        """
        completed = len(self._completed_case_ids())
        return (
            f"**Progress:** {completed} of {len(self.records)} completed "
            f"• Current case: {index + 1} of {len(self.records)}"
        )

    def _case_markdown(self, record: dict[str, Any], index: int) -> str:
        """Render a concise summary of the current evaluation case."""
        target_title = record.get("target_title") or "_None_"
        image_path = record.get("image_path") or "_None_"
        notes = record.get("notes") or "_None_"

        return (
            f"### Case {index + 1} of {len(self.records)}: "
            f"`{record['case_id']}`\n"
            f"**Query type:** {record['query_type']}  \n"
            f"**Category:** {record['category']}  \n"
            f"**Target title:** {target_title}  \n"
            f"**Current image path:** `{image_path}`  \n"
            f"**Notes:** {notes}"
        )
