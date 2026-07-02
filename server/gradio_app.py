"""Gradio UI for the ShopTalk recommender."""

import argparse
import json
from pathlib import Path

from .gradio_images import (
    chosen_product_image_paths,
    top_product_image_paths,
)
from .recommender_core.config import RecommenderConfig
from .recommender_core.diagnostics import diagnostics_to_dict
from .shoptalk_paths import (
    COMBINED_BLURBS_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_VECTOR_BACKEND,
    IMAGES_CSV,
    VECTOR_DB_OUTPUT_DIR,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the ShopTalk Gradio application.")
    parser.add_argument("-p", "--personality", type=int, default=-1, help="Choose a personality")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ShopTalk config file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Override the LLM model name from the config file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the LLM temperature from the config file.",
    )
    parser.add_argument(
        "--vector_db_output_dir",
        type=str,
        default=str(VECTOR_DB_OUTPUT_DIR),
        help="Base directory containing generated vector DB artifacts.",
    )
    parser.add_argument(
        "--vector_backend",
        type=str,
        default=DEFAULT_VECTOR_BACKEND,
        choices=["faiss"],
        help="Vector backend to load for serving.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top options to keep when querying db.",
     )
    parser.add_argument(
        "--product_blurbs",
        type=str,
        default=str(COMBINED_BLURBS_PATH),
        help="Path to the product blurbs JSON file.",
    )
    parser.add_argument(
        "--images_csv",
        type=str,
        default=str(IMAGES_CSV),
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


def normalize_image_input(image_input):
    """Normalize Gradio image input into a local filepath or ``None``.

    With ``gr.Image(type="filepath")``, Gradio usually returns a string path.
    Some Gradio versions/components may hand back a ``pathlib.Path`` or a small
    metadata dict containing a path-like value. Keeping this normalization in one
    helper makes the callback easier to test and keeps the recommender interface
    stable.
    """
    if image_input is None:
        return None

    if isinstance(image_input, (str, Path)):
        image_path = str(image_input).strip()
        return image_path or None

    if isinstance(image_input, dict):
        for key in ("path", "name", "file", "filepath"):
            value = image_input.get(key)
            if isinstance(value, (str, Path)):
                image_path = str(value).strip()
                if image_path:
                    return image_path

    raise TypeError(
        "Unsupported Gradio image input type. Expected a filepath string, "
        "pathlib.Path, metadata dict, or None."
    )


def format_chosen_product(chosen_product):
    """Create a compact Markdown product summary for the side panel."""
    if not chosen_product:
        return "No product selected yet."

    lines = [f"**Chosen product:** {chosen_product.item_name}"]
    lines.append(f"**Vector score:** {chosen_product.score:.4f}")
    if chosen_product.product_type:
        lines.append(f"**Product type:** {chosen_product.product_type}")
    if chosen_product.image_paths:
        lines.append(f"**Images:** {len(chosen_product.image_paths)}")
    return "\n\n".join(lines)


def format_top_products(top_products):
    """Create a compact Markdown list of retrieved products."""
    if not top_products:
        return "No retrieved products reported."

    lines = []
    for rank, product in enumerate(top_products, start=1):
        product_id = product.get("product_id", "unknown")
        item_name = product.get("item_name", "Unknown product")
        score = product.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"{rank}. `{product_id}` — {item_name} — score: {score:.4f}")
        else:
            lines.append(f"{rank}. `{product_id}` — {item_name}")
    return "\n".join(lines)


def format_diagnostics_summary(result):
    """Create a readable Markdown diagnostics summary for the Gradio UI."""
    diagnostics = diagnostics_to_dict(result.get("diagnostics"))
    if not diagnostics:
        return "No diagnostics reported yet."

    timings = diagnostics.get("timings") or {}
    total_seconds = timings.get("total_seconds")

    lines = [
        f"**Embedding mode:** {diagnostics.get('embedding_mode', 'unknown')}",
        f"**Decision:** {diagnostics.get('decision', 'unknown')}",
    ]

    if diagnostics.get("chosen_pid"):
        lines.append(f"**Chosen product ID:** `{diagnostics['chosen_pid']}`")
    if diagnostics.get("llm_search_query"):
        lines.append(f"**LLM search query:** {diagnostics['llm_search_query']}")
    if isinstance(total_seconds, (int, float)):
        lines.append(f"**Total response time:** {total_seconds:.3f} seconds")
    if diagnostics.get("initial_llm_response"):
        lines.append(f"**Raw LLM control response:** `{diagnostics['initial_llm_response']}`")

    lines.append("\n**Top retrieved products:**")
    lines.append(format_top_products(diagnostics.get("top_products") or []))
    return "\n\n".join(lines)


def format_diagnostics(result):
    """Return diagnostics as pretty JSON for raw inspection."""
    diagnostics = diagnostics_to_dict(result.get("diagnostics"))
    return json.dumps(diagnostics, indent=2, sort_keys=True)


def initial_assistant_greeting(recommender):
    """Create the first assistant message shown when the UI opens.

    Personality selection belongs to recommender initialization. The UI should
    fail loudly if it receives a recommender without a resolved personality.
    """
    personality = recommender.personality
    if not isinstance(personality, str) or not personality.strip():
        raise ValueError("Recommender must have a non-empty personality string.")

    return (
        f"I'm your {personality.strip()} shopping assistant. "
        "What would you like to shop for today?"
    )


def initial_chat_history(recommender):
    """Create the initial Gradio chatbot history."""
    return [{"role": "assistant", "content": initial_assistant_greeting(recommender)}]


def handle_message(user_text, image_path, chat_history, recommender):
    """Handle one Gradio submit event.

    Args:
        user_text: Optional text query from the textbox.
        image_path: Optional local filepath returned by ``gr.Image(type='filepath')``.
        chat_history: Current Gradio chatbot history.
        recommender: Backend object exposing ``generate_reply``.

    Returns:
        Tuple of ``(chat_history, cleared_text, cleared_image, product_markdown,
        product_images, top_product_images, diagnostics_summary, diagnostics_json)``
        for Gradio outputs.
    """
    chat_history = list(chat_history or [])
    clean_text = (user_text or "").strip()
    text_arg = clean_text or None
    normalized_image_path = normalize_image_input(image_path)

    if not text_arg and normalized_image_path is None:
        return (
            chat_history,
            "",
            None,
            "Enter text, upload an image, or do both.",
            [],
            [],
            "No diagnostics reported yet.",
            "{}",
        )

    result = recommender.generate_reply(
        user_input=text_arg,
        image_path=normalized_image_path,
    )
    assistant_text = latest_ai_message(result.get("conversation", []))
    user_display = format_user_display(text_arg, normalized_image_path)
    chat_history.append({"role": "user", "content": user_display})
    chat_history.append({"role": "assistant", "content": assistant_text})
    chosen_product = result.get("chosen_product")

    return (
        chat_history,
        "",
        None,
        format_chosen_product(chosen_product),
        chosen_product_image_paths(chosen_product),
        top_product_image_paths(result),
        format_diagnostics_summary(result),
        format_diagnostics(result),
    )


def configure_runtime_logging():
    """Configure runtime logging for the Gradio app."""
    from .recommender_core.utils import configure_logging

    configure_logging()


def create_gradio_interface(recommender):
    """Create the Gradio Blocks UI.

    Gradio is imported lazily so tests for helper functions do not require the
    package unless this UI is actually launched.
    """
    import gradio as gr

    css = """
    #app-shell {
        max-width: 1400px;
        margin: 0 auto;
    }

    #app-header {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary);
        color: var(--body-text-color);
        margin-bottom: 1rem;
    }

    #app-header h1 {
        margin-bottom: 0.25rem;
        color: var(--body-text-color);
    }

    #app-header p {
        margin-top: 0;
        color: var(--body-text-color-subdued);
        font-size: 1rem;
    }

    #input-card,
    #result-card,
    #chat-card,
    #retrieval-card {
        border: 1px solid var(--border-color-primary);
        border-radius: 16px;
        padding: 1rem;
        background: var(--block-background-fill);
        color: var(--body-text-color);
        box-shadow: 0 1px 4px rgb(0 0 0 / 8%);
    }

    #input-card textarea,
    #input-card input {
        color: var(--body-text-color);
    }

    #input-card textarea::placeholder,
    #input-card input::placeholder {
        color: var(--body-text-color-subdued);
        opacity: 1;
    }

    #search-button {
        height: 48px;
        font-weight: 700;
    }
    """

    def submit_message(text, image, history):
        return handle_message(text, image, history, recommender)

    def reset_conversation():
        recommender.reset_conversation()
        return (
            initial_chat_history(recommender),
            "",
            None,
            "No product selected yet.",
            [],
            [],
            "No diagnostics reported yet.",
            "{}",
        )

    with gr.Blocks(title="ShopTalk", css=css) as demo:
        with gr.Column(elem_id="app-shell"):
            gr.HTML(
                """
                <div id="app-header">
                    <h1>ShopTalk</h1>
                    <p>Ask for product recommendations using text, an image, or both.</p>
                </div>
                """
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=7, elem_id="chat-card"):
                    chatbot = gr.Chatbot(
                        value=initial_chat_history(recommender),
                        label="Conversation",
                        height=520,
                    )

                with gr.Column(scale=5, elem_id="result-card"):
                    gr.Markdown("## Recommended product")
                    chosen_product = gr.Markdown(
                        value="No product selected yet.",
                        label="Chosen product",
                    )
                    chosen_product_images = gr.Gallery(
                        label="Chosen product images",
                        columns=3,
                        height=300,
                        object_fit="contain",
                    )

            with gr.Group(elem_id="input-card"):
                gr.Markdown("## Ask for a recommendation")
                with gr.Row(equal_height=True):
                    user_text = gr.Textbox(
                        label="Text query",
                        placeholder="",
                        lines=3,
                        scale=7,
                    )
                    image_input = gr.Image(
                        label="Optional image query",
                        type="filepath",
                        height=180,
                        scale=4,
                    )

                with gr.Row():
                    submit = gr.Button(
                        "Send",
                        variant="primary",
                        elem_id="search-button",
                    )
                    gr.ClearButton(
                        components=[user_text, image_input],
                        value="Clear input",
                    )
                    reset = gr.Button("Reset conversation")

            with gr.Group(elem_id="retrieval-card"):
                gr.Markdown("## Top retrieved products")
                top_product_images = gr.Gallery(
                    label="Top retrieved product images",
                    columns=5,
                    height=360,
                    object_fit="contain",
                )

            with gr.Accordion("Diagnostics", open=False):
                diagnostics_summary = gr.Markdown(
                    value="No diagnostics reported yet.",
                    label="Diagnostics summary",
                )
                diagnostics = gr.Code(label="Raw diagnostics", language="json")

            outputs = [
                chatbot,
                user_text,
                image_input,
                chosen_product,
                chosen_product_images,
                top_product_images,
                diagnostics_summary,
                diagnostics,
            ]
            inputs = [user_text, image_input, chatbot]

            submit.click(fn=submit_message, inputs=inputs, outputs=outputs)
            reset.click(fn=reset_conversation, inputs=[], outputs=outputs)

    return demo


def main():
    args = parse_args()
    configure_runtime_logging()
    from .recommender_core.recommender_factory import build_recommender

    config = RecommenderConfig.from_args(args)
    recommender = build_recommender(config)
    demo = create_gradio_interface(recommender)
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
