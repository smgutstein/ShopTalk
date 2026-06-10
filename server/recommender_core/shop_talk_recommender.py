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
        self.config = config or RecommenderConfig()
        self.debug = self.config.debug
        self.product_store = product_store
        self.top_k = self.config.top_k
        self.query_embedder = query_embedder
        self.conversation_policy = conversation_policy
        self.image_id_to_path = image_id_to_path
        self.personality = personality
        self.chosen_personality = personality
        self.db_load_time = db_load_time
        self.embed_load_time = embed_load_time
        self.image_path_load_time = image_path_load_time

        self.conversation_history = self._initial_conversation_history(self.personality)

        if self.debug:
            self._initialize_debug_file()

    @classmethod
    def from_args(cls, args):
        from .recommender_factory import build_recommender

        return build_recommender(RecommenderConfig.from_args(args))

    def _initial_conversation_history(self, personality):
        sys_msg_str = (
            f"You are a helpful shopping assistant with personality: {personality}. "
            "You are helping a user find a product, after gathering enough info to make a strong recommendation. "
            "Never recommend a product that isn't first provided to you by a system message, "
            "and do not ask the user for product IDs or information - it will be given to you automatically."
        )
        return [
            SystemMessage(content=sys_msg_str),
            AIMessage(content="What would you like to shop for today?"),
        ]

    def _initialize_debug_file(self):
        DEBUG_FILE.unlink(missing_ok=True)
        with open(DEBUG_FILE, "a") as f:
            f.write(f"Embedding Load Time: {self.embed_load_time}\n")
            f.write(f"DB Load Time: {self.db_load_time}\n")
            f.write(f"Image Path Load Time: {self.image_path_load_time}\n")
            f.write(f"Chosen Personality: {self.chosen_personality}\n")
            f.write("\n\n")

    def generate_reply(self, user_input=None, image_path=None):
        start_time = datetime.now()

        logging.info(f"\n\n\nUser input: {user_input}")
        if image_path is not None:
            logging.info(f"User image input: {image_path}")

        request = self._prepare_reply_request(
            user_input=user_input,
            image_path=image_path,
        )

        search_result = self._search_for_products(request)

        decision = self.conversation_policy.decide_next_response(
            conversation_history=self.conversation_history,
            search_result=search_result,
        )

        final_llm_response = self.conversation_policy.generate_final_response(
            conversation_history=self.conversation_history,
            decision=decision,
            source_knowledge=search_result.source_knowledge,
        )

        timing = self._measure_reply_time(start_time)

        diagnostics = self._build_diagnostics(
            request=request,
            search_result=search_result,
            decision=decision,
            timing=timing,
        )

        if self.debug:
            self._write_turn_debug_info(
                request=request,
                search_result=search_result,
                decision=decision,
                final_llm_response=final_llm_response,
                timing=timing,
                diagnostics=diagnostics
            )

        return self._build_reply_payload(
            final_llm_response=final_llm_response,
            decision=decision,
            diagnostics=diagnostics,
        )
    
    def _prepare_reply_request(self, user_input=None, image_path=None):
        if not user_input and image_path is None:
            raise ValueError("generate_reply requires user_input, image_path, or both.")

        image_path = Path(image_path) if image_path is not None else None
        embedding_mode = determine_embedding_mode(user_input, image_path)

        message_content = (
            user_input
            or "The user uploaded an image and wants product recommendations based on it."
        )

        if user_input and image_path is not None:
            message_content = (
                f"{user_input}\n\n"
                "[The user also uploaded an image for the product search.]"
            )

        self.conversation_history.append(HumanMessage(content=message_content))

        return ReplyRequest(
            user_input=user_input,
            image_path=image_path,
            embedding_mode=embedding_mode,
            message_content=message_content,
        )

    def _search_for_products(self, request):
        llm_search_query = (
                            self.conversation_policy.build_search_query(self.conversation_history)
                            if request.user_input
                            else None
                           )

        found_products = self._search_products(
            search_query=llm_search_query,
            top_k=self.top_k,
            image_path=request.image_path,
        )

        source_knowledge = format_source_knowledge(found_products)

        return ProductSearchResult(
            llm_search_query=llm_search_query,
            found_products=found_products,
            source_knowledge=source_knowledge,
        )

    def _measure_reply_time(self, start_time):
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
        return build_recommendation_diagnostics(
            embedding_mode=request.embedding_mode,
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
        if not found_products:
            return None

        return max(found_products.values(), key=lambda product: product.score)

    def _build_reply_payload(self, final_llm_response, decision, diagnostics):
        ai_ans = AIMessage(content=final_llm_response)
        self.conversation_history.append(ai_ans)

        logging.info(f"Chosen pid: {decision.chosen_pid}")
        logging.info(f"Chosen product: {decision.chosen_product}")

        return {
            "conversation": serialize_convo(self.conversation_history),
            "chosen_product": decision.chosen_product,
            "personality": self.personality,
            "diagnostics": diagnostics,
        }

    def _search_products(self, search_query=None, top_k=10, image_path=None):
        query_embeddings = []
        if search_query:
            query_embeddings.append(self.query_embedder.embed_query(search_query).flatten())
        if image_path is not None:
            query_embeddings.append(self.query_embedder.embed_image(image_path).flatten())

        embedded_query = combine_query_embeddings(query_embeddings).tolist()
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
        with open(DEBUG_FILE, "a") as f:
            f.write(f"User input: {user_input}\n")
            if chosen_product:
                f.write(f"  Chosen result: {chosen_product.item_name}\n")
                f.write(f"  Score: {chosen_product.score}\n")
            else:
                f.write("  Chosen result: None\n")
                f.write("  Score: N/A\n")
            f.write("\n")

            if max_score_dict:
                f.write(f"  Best Item: {max_score_dict.item_name}\n")
                f.write(f"  Best Score: {max_score_dict.score}\n\n")
            else:
                f.write("  Best Item: None\n")
                f.write("  Best Score: N/A\n\n")

            f.write(f"  Embedding Mode: {diagnostics.embedding_mode}\n")
            f.write(f"  LLM Search Query: {diagnostics.llm_search_query}\n\n")
            f.write(f"  Initial LLM Response: {decision.initial_llm_response}\n")
            f.write(f"  Decision: {diagnostics.decision}\n")
            f.write(f"  Chosen PID: {decision.chosen_pid}\n")
            f.write(f"  Dive Deeper: {decision.dive_deeper}\n\n")
            f.write(f"  Response Time: {minutes} minutes, {seconds} seconds\n")
            f.write(f"  Response: {llm_response}\n")
            f.write("\n\n")
            for pid, product in sorted(
                found_products.items(), key=lambda x: x[1].score, reverse=True
            ):
                f.write(f"  {product.item_name}: {product.score} \n")
                f.write(f"       {list(product.image_paths)} \n")
                f.write(f"       {product.llm_str} \n")
                f.write("\n")
            f.write("======================================================\n\n")
