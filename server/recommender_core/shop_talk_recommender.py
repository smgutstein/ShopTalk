import logging

from datetime import datetime
from pathlib import Path

from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage

from .config import RecommenderConfig
from .diagnostics import build_recommendation_diagnostics
from .llm_prompts import format_source_knowledge
from .parsing import determine_embedding_mode
from .reply_types import (
    ProductSearchResult,
    RecommendationDecision,
    ReplyRequest,
    ReplyTiming,
)
from .utils import (
    elapsed_time_string,
    serialize_convo,
)
from .vector_query import combine_query_embeddings

from ..shoptalk_paths import DEBUG_FILE


class ShopTalkRecommender:
    # Main runtime coordinator for a single ShopTalk recommender instance.
    #
    # This class does not directly implement the lower-level mechanics of
    # vector search, embedding generation, or LLM prompting. Those details are
    # injected as dependencies:
    #
    # - product_store: searches the vector database and returns product candidates.
    # - query_embedder: converts text and/or image input into vector embeddings.
    # - conversation_policy: makes LLM-driven search and recommendation decisions.
    #
    # The class's job is to connect those pieces into one turn-level control flow:
    # receive user input, update conversation state, optionally retrieve products,
    # ask the policy layer what to do next, generate a final response, record
    # diagnostics/debug information, and return a UI-friendly payload.
    def __init__(
        self,
        *,
        config: RecommenderConfig | None = None,
        product_store,
        query_embedder,
        conversation_policy,
        image_id_to_path: dict,
        personality: str,
        db_load_time: str,
        embed_load_time: str,
        image_path_load_time: str,
    ):
        # Runtime configuration controls debug mode, top-k retrieval size, model
        # settings, artifact paths, and other recommender options. If no config is
        # supplied, use the default RecommenderConfig.
        self.config = config or RecommenderConfig()
        self.debug = self.config.debug

        # Dependency objects are provided by recommender_factory.build_recommender.
        # Keeping construction outside this class makes the runtime easier to test:
        # tests can pass fake stores, fake embedders, or fake policy objects.
        self.product_store = product_store
        self.top_k = self.config.top_k
        self.query_embedder = query_embedder
        self.conversation_policy = conversation_policy

        # Product candidates can reference image IDs. This mapping lets the vector
        # store convert those IDs into actual local image paths for the Gradio UI.
        self.image_id_to_path = image_id_to_path

        # A personality is required because it is baked into the initial system
        # prompt. Letting it be empty would produce an unclear assistant identity
        # and make the initial conversation state less predictable.
        if not isinstance(personality, str) or not personality.strip():
            raise ValueError("ShopTalkRecommender requires a non-empty personality string.")
        self.personality = personality.strip()
        self.chosen_personality = self.personality

        # These strings are not needed for recommendation logic. They are kept so
        # debug output can record how long the major startup artifact loads took.
        self.db_load_time = db_load_time
        self.embed_load_time = embed_load_time
        self.image_path_load_time = image_path_load_time

        # LangChain message history is the backend source of conversational truth.
        # The Gradio chatbot has its own visible transcript, but this list is what
        # the LLM decision and response calls actually receive.
        self.conversation_history = self._initial_conversation_history(self.personality)

        # In debug mode, start a fresh debug log for this process. This intentionally
        # removes any prior DEBUG_FILE so one run's trace does not mix with another.
        if self.debug:
            self._initialize_debug_file()

    def reset_conversation(self):
        """Reset the internal conversation state to the initial assistant context."""
        # Used by the UI reset button. This clears accumulated user/assistant turns
        # but preserves the same selected personality for the fresh conversation.
        self.conversation_history = self._initial_conversation_history(self.personality)

    @classmethod
    def from_args(cls, args):
        # Convenience constructor for command-line entry points. The actual assembly
        # work stays in recommender_factory so this class does not need to know how
        # to load FAISS, ImageBind, OpenAI models, product blurbs, or image maps.
        from .recommender_factory import build_recommender

        return build_recommender(RecommenderConfig.from_args(args))

    def _initial_conversation_history(self, personality):
        # The system message establishes the assistant persona and the key safety
        # rule for grounded recommendation: only recommend products that have been
        # provided later through retrieved product evidence.
        sys_msg_str = (
            f"You are a helpful shopping assistant with personality: {personality}. "
            "You are helping a user find a product, after gathering enough info to make a strong recommendation. "
            "Never recommend a product that isn't first provided to you by a system message, "
            "and do not ask the user for product IDs or information - it will be given to you automatically."
        )

        # Include an initial assistant turn so the LLM history and the displayed UI
        # conversation start from the same conceptual greeting.
        return [
            SystemMessage(content=sys_msg_str),
            AIMessage(content="What would you like to shop for today?"),
        ]

    def _initialize_debug_file(self):
        # Start each debug run with a clean file and write process-level startup
        # metadata before any individual user turns are appended.
        DEBUG_FILE.unlink(missing_ok=True)
        with open(DEBUG_FILE, "a") as f:
            f.write(f"Embedding Load Time: {self.embed_load_time}\n")
            f.write(f"DB Load Time: {self.db_load_time}\n")
            f.write(f"Image Path Load Time: {self.image_path_load_time}\n")
            f.write(f"Chosen Personality: {self.chosen_personality}\n")
            f.write("\n\n")

    def generate_reply(self, user_input=None, image_path=None):
        # Top-level method for one user turn. This is the main runtime path called
        # by the Gradio callback. It accepts text, an image path, or both.
        start_time = datetime.now()

        # Log the raw incoming inputs before normalization. This helps diagnose
        # whether the UI passed text, an image path, both, or neither.
        if user_input is not None:
            logging.info(f"\n\n\nUser input: {user_input}")
        else:
            logging.info(f"\n\n\nNo User input")

        if image_path is not None:
            logging.info(f"User image input: {image_path}")
        else:
            logging.info(f"\n\n\nNo User image input")


        # Normalize the incoming turn into a ReplyRequest and append the user's
        # message to backend conversation history.
        request = self._prepare_reply_request(
            user_input=user_input,
            image_path=image_path,
        )

        # Decide whether to search and, if so, perform vector retrieval. For image-
        # only turns this goes straight to retrieval; for text/text+image turns the
        # LLM policy first decides whether retrieval is warranted.
        search_result = self._search_for_products(request)

        if search_result.search_performed:
            # Post-retrieval decision step. The policy layer receives the current
            # conversation and retrieved product evidence, then chooses whether to
            # recommend, ask for more detail, or treat retrieval as wrong-track.
            decision = self.conversation_policy.decide_next_response(
                conversation_history=self.conversation_history,
                search_result=search_result,
            )

            # Final response generation is deliberately separate from the structured
            # decision step. The decision determines the control branch; this call
            # turns that branch into user-facing language.
            final_llm_response = self.conversation_policy.generate_final_response(
                conversation_history=self.conversation_history,
                decision=decision,
                source_knowledge=search_result.source_knowledge,
            )
        else:
            # No-search path for conversational turns such as thanks, clarification,
            # or other messages that should be answered without hitting the vector DB.
            decision = RecommendationDecision(
                initial_llm_response="<NO SEARCH>",
                chosen_pid=None,
                chosen_product=None,
                dive_deeper=False,
            )

            final_llm_response = self.conversation_policy.generate_response_without_search(
                conversation_history=self.conversation_history,
            )

        # Timing and diagnostics are gathered after the LLM/retrieval path completes
        # so the reported duration covers the whole turn.
        timing = self._measure_reply_time(start_time)

        diagnostics = self._build_diagnostics(
            request=request,
            search_result=search_result,
            decision=decision,
            timing=timing,
        )

        # Debug mode writes a more verbose trace to DEBUG_FILE. The normal return
        # payload still includes diagnostics for UI display.
        if self.debug:
            self._write_turn_debug_info(
                request=request,
                search_result=search_result,
                decision=decision,
                final_llm_response=final_llm_response,
                timing=timing,
                diagnostics=diagnostics
            )

        # Append the final assistant answer to conversation history and return the
        # serialized response object expected by the Gradio layer.
        return self._build_reply_payload(
            final_llm_response=final_llm_response,
            decision=decision,
            diagnostics=diagnostics,
        )
    
    def _prepare_reply_request(self, user_input=None, image_path=None):
        # At least one modality is required. Empty text plus no image is a caller
        # error; the UI should generally catch this before calling generate_reply.
        if not user_input and image_path is None:
            raise ValueError("generate_reply requires user_input, image_path, or both.")

        # Convert image paths to pathlib.Path early so downstream code has a stable
        # path representation regardless of whether Gradio supplied a string/path.
        image_path = Path(image_path) if image_path is not None else None

        # embedding_mode is diagnostic/control metadata such as text, image, or
        # multimodal. The embedding work itself happens later in _search_products.
        embedding_mode = determine_embedding_mode(user_input, image_path)

        # The LLM conversation history needs a text message even for image-only
        # turns. Use a synthetic human message so later policy calls know the user
        # asked for recommendations based on an uploaded image.
        message_content = (
            user_input
            or "The user uploaded an image and wants product recommendations based on it."
        )

        # For text+image turns, preserve the user's text while adding an explicit
        # note that image evidence also exists. The actual image is not embedded in
        # the LangChain text history; it is handled by the vector-search path.
        if user_input and image_path is not None:
            message_content = (
                f"{user_input}\n\n"
                "[The user also uploaded an image for the product search.]"
            )

        # This is the moment the current user turn becomes part of the backend LLM
        # context. Subsequent search and response policy calls see this message.
        self.conversation_history.append(HumanMessage(content=message_content))

        return ReplyRequest(
            user_input=user_input,
            image_path=image_path,
            embedding_mode=embedding_mode,
            message_content=message_content,
        )

    def _search_for_products(self, request):
        # Image-only input cannot be converted into a text search decision, so skip
        # the LLM search/no-search classifier and search directly with the image.
        if request.image_path is not None and not request.user_input:
            return self._run_product_search(
                search_query=None,
                image_path=request.image_path,
            )

        # For text or text+image input, ask the policy layer whether this turn needs
        # retrieval. If it does, the policy also provides the text query that should
        # be embedded for vector search.
        search_decision = self.conversation_policy.decide_search_action(
            self.conversation_history
        )

        # Some turns are conversational rather than product-search turns. Return an
        # dummy ProductSearchResult so the rest of the pipeline can follow a
        # no-search branch without checking for None.
        if search_decision.action == "answer_without_search":
            return ProductSearchResult(
                search_performed=False,
                llm_search_query=None,
                found_products={},
                source_knowledge="",
            )

        # Search with the policy-generated text query, optionally combined with the
        # user-provided image if this is a text+image turn.
        return self._run_product_search(
            search_query=search_decision.search_query,
            image_path=request.image_path,
        )
    
    def _run_product_search(self, search_query=None, image_path=None):
        # This wrapper turns raw product retrieval into the richer ProductSearchResult
        # object consumed by the decision and diagnostics layers.
        found_products = self._search_products(
            search_query=search_query,
            top_k=self.top_k,
            image_path=image_path,
        )

        # Convert retrieved products into a source-knowledge block for the LLM. This
        # is where product IDs, names, scores, types, and llm_str details become text
        # evidence for recommendation decision making.
        source_knowledge = format_source_knowledge(found_products)

        return ProductSearchResult(
            search_performed=True,
            llm_search_query=search_query,
            found_products=found_products,
            source_knowledge=source_knowledge,
        )

    def _measure_reply_time(self, start_time):
        # Capture elapsed wall-clock time for logging, diagnostics, and optional
        # debug output. elapsed_time_string preserves the existing human-readable
        # minutes/seconds formatting used elsewhere in the project.
        stop_time = datetime.now()
        minutes, seconds, _ = elapsed_time_string(start_time, stop_time)
        total_seconds = (stop_time - start_time).total_seconds()

        logging.info(
            f"Took {minutes} minutes, {seconds} seconds to prepare a response "
            "to the user's message."
        )

        return ReplyTiming(
            minutes=minutes,
            seconds=seconds,
            total_seconds=total_seconds,
        )

    def _build_diagnostics(self, request, search_result, decision, timing):
        # Diagnostics flatten the major branch decisions and retrieval results into
        # one object suitable for both the Gradio diagnostics panel and debug logs.
        return build_recommendation_diagnostics(
            embedding_mode=request.embedding_mode,
            search_performed=search_result.search_performed,
            llm_search_query=search_result.llm_search_query,
            found_products=search_result.found_products,
            initial_llm_response=decision.initial_llm_response,
            chosen_pid=decision.chosen_pid,
            dive_deeper=decision.dive_deeper,
            total_seconds=timing.total_seconds,
        )

    def _write_turn_debug_info(
        self,
        request,
        search_result,
        decision,
        final_llm_response,
        timing,
        diagnostics,
    ):
        # For debugging, record not only the chosen product but also the highest-
        # scoring retrieved product. Comparing these reveals cases where the LLM
        # deliberately chose a lower-scoring item or refused to choose any item.
        max_score_dict = self._best_scored_product(search_result.found_products)

        self._write_debug_info(
            user_input=request.user_input,
            chosen_product=decision.chosen_product,
            max_score_dict=max_score_dict,
            decision=decision,
            minutes=timing.minutes,
            seconds=timing.seconds,
            llm_response=final_llm_response,
            found_products=search_result.found_products,
            diagnostics=diagnostics,
        )

    def _best_scored_product(self, found_products):
        # Return None for no-search or empty-search cases. Otherwise, use the vector
        # similarity score to identify the top retrieval result before LLM reranking.
        if not found_products:
            return None

        return max(found_products.values(), key=lambda product: product.score)

    def _build_reply_payload(self, final_llm_response, decision, diagnostics):
        # Persist the assistant's final response in backend conversation history so
        # the next turn has access to what the assistant just told the user.
        ai_ans = AIMessage(content=final_llm_response)
        self.conversation_history.append(ai_ans)

        logging.info(f"Chosen pid: {decision.chosen_pid}")
        logging.info(f"Chosen product: {decision.chosen_product}")

        # The Gradio layer expects serialized conversation text, the selected
        # ProductCandidate if one exists, the personality label, and diagnostics.
        return {
            "conversation": serialize_convo(self.conversation_history),
            "chosen_product": decision.chosen_product,
            "personality": self.personality,
            "diagnostics": diagnostics,
        }

    def _search_products(self, search_query=None, top_k=10, image_path=None):
        # Build a list of modality-specific embeddings. Text-only produces one text
        # vector, image-only produces one image vector, and text+image produces both.
        query_embeddings = []
        if search_query:
            query_embeddings.append(self.query_embedder.embed_query(search_query).flatten())
        if image_path is not None:
            query_embeddings.append(self.query_embedder.embed_image(image_path).flatten())

        # combine_query_embeddings is the current multimodal fusion point. It turns
        # one or more query vectors into the single vector expected by the FAISS
        # product store.
        embedded_query = combine_query_embeddings(query_embeddings).tolist()

        # ProductVectorStore performs the actual vector lookup, maps vector rows back
        # to product IDs/blurbs, deduplicates product candidates, and attaches image
        # paths using image_id_to_path.
        found_products = self.product_store.search(
            embedded_query=embedded_query,
            top_k=top_k,
            image_id_to_path=self.image_id_to_path,
        )
        product_names = [info.item_name for info in found_products.values()]
        logging.info(f"VectorDB search results: {product_names}")
        return found_products


    def _write_debug_info(
        self,
        user_input,
        chosen_product,
        max_score_dict,
        decision,
        minutes,
        seconds,
        llm_response,
        found_products,
        diagnostics
    ):
        # Append a full turn trace to the debug file. This intentionally includes
        # product details and image paths, so it is useful for development but too
        # verbose for normal user-facing output.
        with open(DEBUG_FILE, "a") as f:
            f.write(f"User input: {user_input}\n")

            # First record the LLM-selected product, if any. This may differ from
            # the highest-scoring vector result because the LLM decision layer can
            # reject high-scoring but semantically unsuitable products.
            if chosen_product:
                f.write(f"  Chosen result: {chosen_product.item_name}\n")
                f.write(f"  Score: {chosen_product.score}\n")
            else:
                f.write("  Chosen result: None\n")
                f.write("  Score: N/A\n")
            f.write("\n")

            # Then record the best raw retrieval hit for comparison against the LLM
            # decision. This helps diagnose retrieval/decision disagreements.
            if max_score_dict:
                f.write(f"  Best Item: {max_score_dict.item_name}\n")
                f.write(f"  Best Score: {max_score_dict.score}\n\n")
            else:
                f.write("  Best Item: None\n")
                f.write("  Best Score: N/A\n\n")

            # Record high-level decision metadata before dumping the verbose product
            # list. These fields are the fastest way to inspect the control branch.
            f.write(f"  Embedding Mode: {diagnostics.embedding_mode}\n")
            f.write(f"  Search Performed: {diagnostics.search_performed}\n")
            f.write(f"  LLM Search Query: {diagnostics.llm_search_query}\n\n")
            f.write(f"  Initial LLM Response: {decision.initial_llm_response}\n")
            f.write(f"  Decision: {diagnostics.decision}\n")
            f.write(f"  Chosen PID: {decision.chosen_pid}\n")
            f.write(f"  Dive Deeper: {decision.dive_deeper}\n\n")
            f.write(f"  Response Time: {minutes} minutes, {seconds} seconds\n")
            f.write(f"  Response: {llm_response}\n")
            f.write("\n\n")

            # Dump every retrieved product, sorted by vector score. This makes the
            # debug file a complete snapshot of what evidence the LLM had available
            # when it chose recommend/dive_deeper/wrong_track.
            for pid, product in sorted(
                found_products.items(), key=lambda x: x[1].score, reverse=True
            ):
                f.write(f"  {product.item_name}: {product.score} \n")
                f.write(f"       {list(product.image_paths)} \n")
                f.write(f"       {product.llm_str} \n")
                f.write("\n")
            f.write("======================================================\n\n")
