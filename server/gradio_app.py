"""Gradio UI for the ShopTalk recommender."""

import argparse
# Command-line parsing is kept in this UI module because these flags are
# specific to serving the Gradio app, not to the lower-level recommender logic.
import json
from pathlib import Path

# Image helper functions convert backend product objects/result payloads into
# filesystem paths that Gradio Gallery components know how to render.
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


# parse_args is intentionally small and declarative: it only translates CLI flags
# into an argparse Namespace. RecommenderConfig.from_args later interprets those
# values and resolves paths/model defaults.
def parse_args():
    parser = argparse.ArgumentParser(description="Run the ShopTalk Gradio application.")
    # Personality/debug/device flags are user-facing convenience knobs for local
    # demos. They are passed through to RecommenderConfig rather than handled here.
    parser.add_argument("-p", "--personality", type=int, default=-1, help="Choose a personality")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-c", "--cpu", action="store_true")
    # The config file supplies defaults for model and runtime settings; explicit
    # command-line flags below can still override selected fields.
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
    # Vector DB and product-artifact arguments make the UI launchable against
    # different generated artifact directories without changing Python code.
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
    # Gradio serving options are kept explicit so Docker/launcher scripts can
    # bind to 0.0.0.0 while normal local runs can bind to localhost.
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
    # Return the raw Namespace; downstream config construction decides how to
    # combine CLI values with defaults from shoptalk_paths/config files.
    return parser.parse_args()



# The recommender returns its internal LangChain conversation serialized as a
# list of message dictionaries. The Gradio chatbot only needs the latest AI text
# for the visible assistant bubble added after this submit event.
def latest_ai_message(conversation):
    """Return the most recent assistant/AI message content from serialized chat."""
    # Walk backward because the latest assistant response is usually the last AI
    # message, and this avoids depending on a fixed conversation length.
    for message in reversed(conversation or []):
        if message.get("type") == "AIMessage":
            return message.get("content", "")
    return ""



# The backend receives text/image inputs separately, but the visible chatbot has
# a single text bubble for the user. This helper summarizes the submitted input
# without trying to embed the uploaded image itself in the chat transcript.
def format_user_display(user_text, image_path):
    """Create the user-side text shown in the Gradio chatbot."""
    # Strip whitespace so an all-whitespace textbox is treated the same as no
    # text input. The original image_path value is enough to indicate image use.
    clean_text = (user_text or "").strip()
    if clean_text and image_path is not None:
        return f"{clean_text}\n\n[image uploaded]"
    if clean_text:
        return clean_text
    if image_path is not None:
        return "[image uploaded]"
    return ""



# Gradio image components have changed return shapes across versions. This
# function keeps that compatibility mess out of handle_message and keeps the
# recommender API stable: it receives either a real filepath or None.
def normalize_image_input(image_input):
    """Normalize Gradio image input into a local filepath or ``None``.

    With ``gr.Image(type="filepath")``, Gradio usually returns a string path.
    Some Gradio versions/components may hand back a ``pathlib.Path`` or a small
    metadata dict containing a path-like value. Keeping this normalization in one
    helper makes the callback easier to test and keeps the recommender interface
    stable.
    """
    # None is the normal value when the user did not upload an image or the
    # component has just been cleared.
    if image_input is None:
        return None

    # type="filepath" normally gives a string path. Path is accepted defensively
    # because some wrappers/tests may provide pathlib objects.
    if isinstance(image_input, (str, Path)):
        image_path = str(image_input).strip()
        return image_path or None

    # Some Gradio versions/components return a small metadata dict. Only known
    # path-like keys are accepted; unknown non-empty shapes should fail loudly
    # rather than silently dropping an uploaded image.
    if isinstance(image_input, dict):
        for key in ("path", "name", "file", "filepath"):
            value = image_input.get(key)
            if isinstance(value, (str, Path)):
                image_path = str(value).strip()
                if image_path:
                    return image_path

    # Reaching this point means Gradio gave us an object that this UI layer does
    # not know how to convert into a local image file path. Raising here is better
    # than letting multimodal search silently degrade into text-only search.
    raise TypeError(
        "Unsupported Gradio image input type. Expected a filepath string, "
        "pathlib.Path, metadata dict, or None."
    )



# The right-side "Recommended product" panel needs concise Markdown, not the
# full product evidence string sent to the LLM. Keep this display focused on
# identity, score, type, and image count.
def format_chosen_product(chosen_product):
    """Create a compact Markdown product summary for the side panel."""
    # No chosen product is valid for no-search turns, follow-up-question turns,
    # wrong-track decisions, and the initial UI state.
    if not chosen_product:
        return "No product selected yet."

    # Score is shown because this is a portfolio/debuggable recommender UI. It is
    # retrieval evidence, not necessarily the reason the LLM chose the product.
    lines = [f"**Chosen product:** {chosen_product.item_name}"]
    lines.append(f"**Vector score:** {chosen_product.score:.4f}")
    if chosen_product.product_type:
        lines.append(f"**Product type:** {chosen_product.product_type}")
    if chosen_product.image_paths:
        lines.append(f"**Images:** {len(chosen_product.image_paths)}")
    return "\n\n".join(lines)



# Diagnostics store top products as dictionaries so the UI can render them even
# after dataclass/model objects have been serialized. This function formats that
# serialized representation for human inspection.
def format_top_products(top_products):
    """Create a compact Markdown list of retrieved products."""
    if not top_products:
        return "No retrieved products reported."

    lines = []
    # enumerate(..., start=1) gives user-facing ranks rather than zero-based
    # Python indices.
    for rank, product in enumerate(top_products, start=1):
        product_id = product.get("product_id", "unknown")
        item_name = product.get("item_name", "Unknown product")
        score = product.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"{rank}. `{product_id}` — {item_name} — score: {score:.4f}")
        else:
            lines.append(f"{rank}. `{product_id}` — {item_name}")
    return "\n".join(lines)



# The diagnostics accordion has two views: this readable Markdown summary and a
# raw JSON dump below. The summary highlights the control-flow decisions that are
# most useful while testing retrieval and LLM policy behavior.
def format_diagnostics_summary(result):
    """Create a readable Markdown diagnostics summary for the Gradio UI."""
    # diagnostics_to_dict accepts the backend diagnostics object and normalizes it
    # to a plain dict so this UI code does not depend on the diagnostics class.
    diagnostics = diagnostics_to_dict(result.get("diagnostics"))
    if not diagnostics:
        return "No diagnostics reported yet."

    # Timings may be missing in error/test payloads, so use an empty dict rather
    # than assuming the full runtime diagnostics structure is always present.
    timings = diagnostics.get("timings") or {}
    total_seconds = timings.get("total_seconds")

    # Start with the two most important control-flow facts: which embedding path
    # was used and what final decision category was recorded.
    lines = [
        f"**Embedding mode:** {diagnostics.get('embedding_mode', 'unknown')}",
        f"**Decision:** {diagnostics.get('decision', 'unknown')}",
    ]

    # Optional fields are appended only when present so the summary stays readable
    # for no-search, dive-deeper, and wrong-track turns.
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



# The raw JSON diagnostics are intentionally verbose: they are meant for debugging
# and portfolio inspection, not for ordinary shopper-facing UX.
def format_diagnostics(result):
    """Return diagnostics as pretty JSON for raw inspection."""
    diagnostics = diagnostics_to_dict(result.get("diagnostics"))
    return json.dumps(diagnostics, indent=2, sort_keys=True)



# The visible Gradio chat starts with a UI greeting. The backend recommender also
# has its own internal conversation history; this function only builds the
# browser-visible initial assistant bubble.
def initial_assistant_greeting(recommender):
    """Create the first assistant message shown when the UI opens.

    Personality selection belongs to recommender initialization. The UI should
    fail loudly if it receives a recommender without a resolved personality.
    """
    # Personality should already be resolved by build_recommender. If it is not,
    # failing here exposes a construction bug immediately instead of showing a
    # bland or inconsistent UI greeting.
    personality = recommender.personality
    if not isinstance(personality, str) or not personality.strip():
        raise ValueError("Recommender must have a non-empty personality string.")

    return (
        f"I'm your {personality.strip()} shopping assistant. "
        "What would you like to shop for today?"
    )



# Gradio Chatbot expects OpenAI-style role/content dictionaries here. This is
# separate from the backend's LangChain message objects.
def initial_chat_history(recommender):
    """Create the initial Gradio chatbot history."""
    return [{"role": "assistant", "content": initial_assistant_greeting(recommender)}]



# This is the main UI-to-backend bridge for a single submit event. It translates
# Gradio component values into recommender inputs, then translates the backend
# result payload back into the exact tuple of Gradio component updates.
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
    # Copy the incoming chat history before appending to avoid mutating Gradio's
    # object in place. This makes the callback easier to reason about and test.
    chat_history = list(chat_history or [])
    clean_text = (user_text or "").strip()
    text_arg = clean_text or None
    normalized_image_path = normalize_image_input(image_path)

    # Empty submits are handled in the UI layer because generate_reply correctly
    # treats "no text and no image" as an invalid backend request.
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

    # The recommender owns the real control flow: search/no-search decision,
    # embedding, vector retrieval, LLM product decision, and final response text.
    result = recommender.generate_reply(
        user_input=text_arg,
        image_path=normalized_image_path,
    )
    # The backend returns the full serialized conversation. For the visible chat,
    # add only the current user turn plus the latest generated assistant reply.
    assistant_text = latest_ai_message(result.get("conversation", []))
    user_display = format_user_display(text_arg, normalized_image_path)
    chat_history.append({"role": "user", "content": user_display})
    chat_history.append({"role": "assistant", "content": assistant_text})
    chosen_product = result.get("chosen_product")

    # Output order must exactly match the outputs list in create_gradio_interface.
    # The textbox and image input are cleared after a successful submit.
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



# Logging setup is kept behind a function so tests can import this module without
# immediately changing global logging configuration.
def configure_runtime_logging():
    """Configure runtime logging for the Gradio app."""
    from .recommender_core.utils import configure_logging

    configure_logging()



# Build the visible Gradio app around an already-constructed recommender. This
# function should not load models, open databases, or make policy decisions; it
# only wires UI components to callback functions.
def create_gradio_interface(recommender):
    """Create the Gradio Blocks UI.

    Gradio is imported lazily so tests for helper functions do not require the
    package unless this UI is actually launched.
    """
    # Lazy import avoids requiring Gradio when unit tests only exercise helper
    # functions such as normalize_image_input or handle_message.
    import gradio as gr

    # CSS is scoped through component elem_id values. It customizes layout and
    # contrast without changing callback behavior.
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

    # Gradio callbacks receive only component values. This closure captures the
    # already-built recommender without exposing it as a hidden UI input.
    def submit_message(text, image, history):
        return handle_message(text, image, history, recommender)


    # Reset must clear both the backend conversation state and all visible UI
    # outputs. Clearing only the textbox/image input would leave the recommender
    # remembering earlier turns.
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

    # Blocks provides explicit layout control: header, chat/result row, input
    # card, retrieval gallery, and diagnostics accordion.
    with gr.Blocks(title="ShopTalk", css=css) as demo:
        with gr.Column(elem_id="app-shell"):
            # Static app header. The id ties this block to the CSS above; it is
            # not a Gradio-specific concept.
            gr.HTML(
                """
                <div id="app-header">
                    <h1>ShopTalk</h1>
                    <p>Ask for product recommendations using text, an image, or both.</p>
                </div>
                """
            )

            # Main results row: visible conversation on the left, selected-product
            # summary/images on the right.
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

            # Input card accepts text, an image, or both. The backend decides the
            # embedding mode after normalization.
            with gr.Group(elem_id="input-card"):
                gr.Markdown("## Ask for a recommendation")
                with gr.Row(equal_height=True):
                    # Multiline textbox favors richer product requests. Plain
                    # Enter usually inserts a newline, so submission is handled by
                    # the Send button rather than textbox.submit.
                    user_text = gr.Textbox(
                        label="Text query",
                        placeholder="",
                        lines=3,
                        scale=7,
                    )
                    # type="filepath" is important: QueryEmbedder expects a file
                    # path that it can pass into the image preprocessing pipeline.
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
                    # Clear input affects only the current draft query/image. It is
                    # intentionally different from Reset conversation below.
                    gr.ClearButton(
                        components=[user_text, image_input],
                        value="Clear input",
                    )
                    reset = gr.Button("Reset conversation")

            # The retrieved-products gallery exposes what vector search returned,
            # even when the LLM chooses to ask a follow-up or reject the results.
            with gr.Group(elem_id="retrieval-card"):
                gr.Markdown("## Top retrieved products")
                top_product_images = gr.Gallery(
                    label="Top retrieved product images",
                    columns=5,
                    height=360,
                    object_fit="contain",
                )

            # Diagnostics are hidden by default but available for development and
            # portfolio inspection of the control-flow decisions.
            with gr.Accordion("Diagnostics", open=False):
                diagnostics_summary = gr.Markdown(
                    value="No diagnostics reported yet.",
                    label="Diagnostics summary",
                )
                diagnostics = gr.Code(label="Raw diagnostics", language="json")

            # Gradio assigns callback return values positionally to this list. If
            # this order changes, handle_message/reset_conversation returns must
            # change in lockstep.
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
            # The callback needs the draft text, uploaded image path, and current
            # visible chat history. The backend's internal history remains inside
            # the recommender object captured by submit_message.
            inputs = [user_text, image_input, chatbot]

            # Both callbacks update the same output components. Submit runs a new
            # recommendation turn; reset restores the initial UI/backend state.
            submit.click(fn=submit_message, inputs=inputs, outputs=outputs)
            reset.click(fn=reset_conversation, inputs=[], outputs=outputs)

    return demo



# Script entry point used by `python -m server.gradio_app` and the launcher
# script. Startup order matters: parse CLI, configure logging, build the
# recommender and its heavy dependencies, then launch Gradio.
def main():
    args = parse_args()
    configure_runtime_logging()
    from .recommender_core.recommender_factory import build_recommender

    # Build the runtime config from CLI/config-file inputs, then delegate all
    # heavyweight construction to the recommender factory.
    config = RecommenderConfig.from_args(args)
    recommender = build_recommender(config)
    demo = create_gradio_interface(recommender)
    # server_name/server_port are supplied by args or launcher scripts. In Docker,
    # server_name is typically 0.0.0.0 so the host port mapping can reach Gradio.
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )



# Allows direct module execution while keeping imports side-effect-light for tests.
if __name__ == "__main__":
    main()
