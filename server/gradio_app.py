"""Gradio UI for the ShopTalk recommender.

This module intentionally lives beside the Flask app instead of replacing it.
Both UIs use the same ``ShopTalkRecommender`` backend, so the Gradio app can add
image-query support without changing the existing Flask route behavior.
"""

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run the ShopTalk Gradio application.")
    parser.add_argument("-p", "--personality", type=int, default=-1, help="Choose a personality")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument("-m", "--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--vector_db_output_dir",
        type=str,
        default="artifacts/vector_db",
        help="Base directory containing generated vector DB artifacts.",
    )
    parser.add_argument(
        "--vector_backend",
        type=str,
        default="faiss",
        choices=["faiss"],
        help="Vector backend to load for serving.",
    )
    parser.add_argument(
        "--product_blurbs",
        type=str,
        default="EDA/product_blurbs/combined_blurb_dict.json",
        help="Path to the product blurbs JSON file.",
    )
    parser.add_argument(
        "--images_csv",
        type=str,
        default="images.csv",
        help="Path to the image ID mapping CSV file.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default=None,
        help="Server host/interface for Gradio.",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=None,
        help="Server port for Gradio.",
    )
    return parser.parse_args()


def latest_ai_message(conversation):
    """Return the most recent assistant/AI message content from serialized chat."""
    for message in reversed(conversation or []):
        if message.get("type") == "AIMessage":
            return message.get("content", "")
    return ""


def format_user_display(user_text, image_path):
    """Create the user-side text shown in the Gradio chatbot."""
    clean_text = (user_text or "").strip()
    if clean_text and image_path is not None:
        return f"{clean_text}\n\n[image uploaded]"
    if clean_text:
        return clean_text
    if image_path is not None:
        return "[image uploaded]"
    return ""


def format_chosen_product(chosen_product):
    """Create a compact Markdown product summary for the side panel."""
    if not chosen_product:
        return "No product selected yet."

    lines = [f"**Chosen product:** {chosen_product.get('item_name', 'Unknown product')}"]
    if "score" in chosen_product:
        lines.append(f"**Vector score:** {chosen_product['score']:.4f}")
    if chosen_product.get("product_type"):
        lines.append(f"**Product type:** {chosen_product['product_type']}")
    if chosen_product.get("image_paths"):
        lines.append(f"**Images:** {len(chosen_product['image_paths'])}")
    return "\n\n".join(lines)


def format_diagnostics(result):
    """Return diagnostics as pretty JSON, even before rich diagnostics exist."""
    diagnostics = result.get("diagnostics") or {}
    return json.dumps(diagnostics, indent=2, sort_keys=True)


def handle_message(user_text, image_path, chat_history, recommender):
    """Handle one Gradio submit event.

    Args:
        user_text: Optional text query from the textbox.
        image_path: Optional local filepath returned by ``gr.Image(type='filepath')``.
        chat_history: Current Gradio chatbot history.
        recommender: Backend object exposing ``generate_reply``.

    Returns:
        Tuple of ``(chat_history, cleared_text, cleared_image, product_markdown,
        diagnostics_json)`` for Gradio outputs.
    """
    chat_history = list(chat_history or [])
    clean_text = (user_text or "").strip()
    text_arg = clean_text or None

    if not text_arg and image_path is None:
        return (
            chat_history,
            "",
            None,
            "Enter text, upload an image, or do both.",
            "{}",
        )

    result = recommender.generate_reply(
        user_input=text_arg,
        image_path=image_path,
    )
    assistant_text = latest_ai_message(result.get("conversation", []))
    user_display = format_user_display(text_arg, image_path)
    chat_history.append((user_display, assistant_text))

    return (
        chat_history,
        "",
        None,
        format_chosen_product(result.get("chosen_product")),
        format_diagnostics(result),
    )


def build_recommender_from_args(args):
    """Build the backend recommender using the Flask server's shared factory."""
    from server import build_recommender

    return build_recommender(args)


def configure_runtime_logging():
    """Configure logging using the shared Flask server helper."""
    from server import configure_logging

    configure_logging()


def create_gradio_interface(recommender):
    """Create the Gradio Blocks UI.

    Gradio is imported lazily so tests for helper functions do not require the
    package unless this UI is actually launched.
    """
    import gradio as gr

    with gr.Blocks(title="ShopTalk") as demo:
        gr.Markdown("# ShopTalk multimodal recommender")
        gr.Markdown("Search by text, image, or both. The backend uses the same recommender as the Flask app.")

        chatbot = gr.Chatbot(label="Conversation")
        with gr.Row():
            user_text = gr.Textbox(label="Text query", placeholder="What are you shopping for?")
            image_input = gr.Image(label="Optional image query", type="filepath")

        submit = gr.Button("Search")
        chosen_product = gr.Markdown(label="Chosen product")
        diagnostics = gr.Code(label="Diagnostics", language="json")

        submit.click(
            fn=lambda text, image, history: handle_message(text, image, history, recommender),
            inputs=[user_text, image_input, chatbot],
            outputs=[chatbot, user_text, image_input, chosen_product, diagnostics],
        )

    return demo


def main():
    args = parse_args()
    configure_runtime_logging()
    recommender = build_recommender_from_args(args)
    demo = create_gradio_interface(recommender)
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
