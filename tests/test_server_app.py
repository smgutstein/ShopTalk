"""Compatibility note: the old Flask ``server.py`` wrapper has been removed.

The active web entry point is now ``server.gradio_app``. These tests keep a
small smoke check around the current UI-facing helpers instead of importing the
removed Flask module.
"""

from server import gradio_app


def test_format_user_display_handles_text_image_and_combined_inputs():
    assert gradio_app.format_user_display("red shoes", None) == "red shoes"
    assert gradio_app.format_user_display("", "query.jpg") == "[image uploaded]"
    assert (
        gradio_app.format_user_display("match this", "query.jpg")
        == "match this\n\n[image uploaded]"
    )
    assert gradio_app.format_user_display("", None) == ""
