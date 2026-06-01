import logging
import os
import random
import torch

from datetime import datetime
from pathlib import Path

from imagebind.models.imagebind_model import imagebind_huge

from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .diagnostics import build_recommendation_diagnostics
from .llm_prompts import (
    build_augmented_prompt,
    build_no_product_reprompt,
    build_search_query_prompt,
    format_source_knowledge,
)
from .parsing import (
    determine_embedding_mode,
    extract_bracketed_choice,
    parse_product_choice,
)
from .product_images import load_image_paths_csv
from .product_vector_store import ProductVectorStore
from .query_embedder import QueryEmbedder
from .utils import (
    PERSONALITIES,
    elapsed_time_string,
    load_openai_api_key,
    serialize_convo,
)
from .vector_query import combine_query_embeddings


class ShopTalkRecommender:
    def __init__(
        self,
        personality_index=-1,
        debug=False,
        force_cpu=False,
        model_name="gpt-4o",
        vector_db_output_dir="artifacts/vector_db",
        vector_backend="faiss",
        index_path=None,
        blurbs_path="EDA/product_blurbs/combined_blurb_dict.json",
        product_ids_path=None,
        images_csv_path="images.csv",
    ):
        self.debug = debug

        self.device = "cpu" if force_cpu else "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        self.api_key = load_openai_api_key()
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        self.product_store, self.db_load_time = self._load_database(
            vector_db_output_dir=vector_db_output_dir,
            vector_backend=vector_backend,
            index_path=index_path,
            blurbs_path=blurbs_path,
            product_ids_path=product_ids_path,
        )

        self.ibind_model, self.embed_load_time = self._load_imagebind_model()
        self.query_embedder = QueryEmbedder(self.ibind_model, self.device)

        self.personality = self._choose_personality(personality_index)
        self.chosen_personality = self.personality
        logging.info(f" {self.personality}")

        self.chat_openai = ChatOpenAI(
            api_key=self.api_key,
            model=model_name,
            temperature=0.1,
        )

        self.conversation_history = self._initial_conversation_history(self.personality)
        self.image_id_to_path, self.image_path_load_time = self._load_image_paths(
            images_csv_path
        )

        if self.debug and Path("debug.txt").exists():
            self._initialize_debug_file()

    def _load_imagebind_model(self):
        logging.info("Loading ImageBind model...")
        start_time = datetime.now()
        ibind_model = imagebind_huge(pretrained=True)
        ibind_model.eval()
        ibind_model.to(self.device)
        stop_time = datetime.now()
        minutes, seconds, load_time = elapsed_time_string(start_time, stop_time)
        logging.info(f"{minutes} minutes, {seconds} seconds")
        return ibind_model, load_time

    def _load_database(
        self,
        vector_db_output_dir,
        vector_backend,
        index_path,
        blurbs_path,
        product_ids_path,
    ):
        logging.info("Loading Database...")
        start_time = datetime.now()

        if vector_backend != "faiss":
            raise ValueError(
                "ShopTalkRecommender currently supports only the FAISS vector backend. "
                f"Got vector_backend={vector_backend!r}."
            )

        backend_dir = Path(vector_db_output_dir) / vector_backend
        index_path = index_path or backend_dir / "embeddings.faiss"
        product_ids_path = product_ids_path or backend_dir / "product_ids.json"

        product_store = ProductVectorStore.from_paths(
            index_path=index_path,
            blurbs_path=blurbs_path,
            product_ids_path=product_ids_path,
        )
        stop_time = datetime.now()
        minutes, seconds, load_time = elapsed_time_string(start_time, stop_time)
        logging.info(f"{minutes} minutes, {seconds} seconds")
        return product_store, load_time

    def _choose_personality(self, personality_index):
        if personality_index == -1 or personality_index >= len(PERSONALITIES):
            personality = random.choice(PERSONALITIES)
            logging.info(f"Random Personality: {personality}")
        else:
            personality = PERSONALITIES[personality_index]
        return personality

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

    def _load_image_paths(self, images_csv_path):
        logging.info("Loading Image paths...")
        start_time = datetime.now()
        image_id_to_path = load_image_paths_csv(images_csv_path)
        stop_time = datetime.now()
        minutes, seconds, load_time = elapsed_time_string(start_time, stop_time)
        logging.info(f"{minutes} minutes, {seconds} seconds")
        return image_id_to_path, load_time

    def _initialize_debug_file(self):
        Path("debug.txt").unlink()
        with open("debug.txt", "a") as f:
            f.write(f"Embedding Load Time: {self.embed_load_time}\n")
            f.write(f"DB Load Time: {self.db_load_time}\n")
            f.write(f"Image Path Load Time: {self.image_path_load_time}\n")
            f.write(f"Chosen Personality: {self.chosen_personality}\n")
            f.write("\n\n")

    def generate_reply(self, user_input=None, image_path=None):
        if not user_input and image_path is None:
            raise ValueError("generate_reply requires user_input, image_path, or both.")

        logging.info(f"\n\n\nUser input: {user_input}")
        if image_path is not None:
            logging.info(f"User image input: {image_path}")
        start_time = datetime.now()
        embedding_mode = determine_embedding_mode(user_input, image_path)

        message_content = user_input or "The user uploaded an image and wants product recommendations based on it."
        if user_input and image_path is not None:
            message_content = f"{user_input}\n\n[The user also uploaded an image for the product search.]"
        self.conversation_history.append(HumanMessage(content=message_content))

        llm_search_query = self._build_search_query() if user_input else None
        found_products = self._search_products(
            llm_search_query,
            top_k=10,
            image_path=image_path,
        )
        source_knowledge = self._format_source_knowledge(found_products)

        initial_llm_response = self._choose_product_or_next_action(
            found_products=found_products,
            source_knowledge=source_knowledge,
        )
        chosen_pid, chosen_product, dive_deeper = self._parse_product_choice(
            llm_response=initial_llm_response,
            found_products=found_products,
        )

        final_llm_response = self._build_final_response(
            initial_llm_response=initial_llm_response,
            chosen_pid=chosen_pid,
            chosen_product=chosen_product,
            dive_deeper=dive_deeper,
            source_knowledge=source_knowledge,
        )
        ai_ans = AIMessage(content=final_llm_response)
        self.conversation_history += [ai_ans]

        logging.info(f"Chosen pid: {chosen_pid}")
        logging.info(f"Chosen product: {chosen_product}")

        stop_time = datetime.now()
        minutes, seconds, _ = elapsed_time_string(start_time, stop_time)
        total_seconds = (stop_time - start_time).total_seconds()
        logging.info(
            f"Took {minutes} minutes, {seconds} seconds to prepare a response to the user's message."
        )

        max_score_dict = max(found_products.values(), key=lambda x: x["score"])
        diagnostics = build_recommendation_diagnostics(
            embedding_mode=embedding_mode,
            llm_search_query=llm_search_query,
            found_products=found_products,
            initial_llm_response=initial_llm_response,
            chosen_pid=chosen_pid,
            dive_deeper=dive_deeper,
            total_seconds=total_seconds,
        )

        if self.debug:
            self._write_debug_info(
                user_input=user_input,
                chosen_product=chosen_product,
                max_score_dict=max_score_dict,
                minutes=minutes,
                seconds=seconds,
                llm_response=final_llm_response,
                found_products=found_products,
            )

        return {
            "conversation": serialize_convo(self.conversation_history),
            "chosen_product": chosen_product,
            "personality": self.personality,
            "diagnostics": diagnostics,
        }

    def _build_search_query(self):
        search_query_prompt = build_search_query_prompt()
        llm_search_query = self.chat_openai.invoke(
            self.conversation_history + [SystemMessage(content=search_query_prompt)]
        ).content
        logging.info(f"LLM's suggested search query: {llm_search_query}")
        return llm_search_query

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
        product_names = [info["item_name"] for info in found_products.values()]
        logging.info(f"VectorDB search results: {product_names}")
        return found_products

    def _choose_product_or_next_action(self, found_products, source_knowledge):
        augmented_prompt = self._build_augmented_prompt(source_knowledge)
        self.conversation_history.append(SystemMessage(augmented_prompt))

        logging.info(f"conversation_history: {self.conversation_history}\n\n")
        llm_response = self.chat_openai.invoke(self.conversation_history).content
        logging.info(f"Initial LLM Response: {llm_response}")

        # Delete search result info from conversation.
        self.conversation_history = self.conversation_history[:-1]
        return llm_response

    def _parse_product_choice(self, llm_response, found_products):
        return parse_product_choice(llm_response, found_products)

    @staticmethod
    def _extract_bracketed_choice(llm_response):
        return extract_bracketed_choice(llm_response)

    def _build_final_response(
        self,
        initial_llm_response,
        chosen_pid,
        chosen_product,
        dive_deeper,
        source_knowledge,
    ):
        ai_ans = AIMessage(content=initial_llm_response)

        if chosen_pid:
            reprompt_str = (
                "Let's continue the conversation while recommending the following product "
                "(you don't need to describe every detail of the product, just whatever seems relevant "
                "for the buyer based on this conversation): "
            )
            reprompt_str += chosen_product["llm_str"]
            log_message = "No-PID LLM Response"
        else:
            reprompt_str, log_message = self._build_no_product_reprompt(
                dive_deeper=dive_deeper,
                source_knowledge=source_knowledge,
            )

        reprompt = SystemMessage(content=reprompt_str)
        temp_history = self.conversation_history + [ai_ans, reprompt]
        final_llm_response = self.chat_openai.invoke(temp_history).content
        logging.info(f"{log_message}: {final_llm_response}")
        return final_llm_response

    def _build_no_product_reprompt(self, dive_deeper, source_knowledge):
        return build_no_product_reprompt(dive_deeper, source_knowledge)

    def _format_source_knowledge(self, found_products):
        return format_source_knowledge(found_products)

    def _build_augmented_prompt(self, source_knowledge):
        return build_augmented_prompt(source_knowledge)

    def _write_debug_info(
        self,
        user_input,
        chosen_product,
        max_score_dict,
        minutes,
        seconds,
        llm_response,
        found_products,
    ):
        with open("debug.txt", "a") as f:
            f.write(f"User input: {user_input}\n")
            if chosen_product:
                f.write(f"  Chosen result: {chosen_product['item_name']}\n")
                f.write(f"  Score: {chosen_product['score']}\n")
            else:
                f.write("  Chosen result: None\n")
                f.write("  Score: N/A\n")
            f.write("\n")
            f.write(f"  Best Item: {max_score_dict['item_name']}\n")
            f.write(f"  Best Score: {max_score_dict['score']}\n\n")
            f.write(f"  Response Time: {minutes} minutes, {seconds} seconds\n")
            f.write(f"  Response: {llm_response}\n")
            f.write("\n\n")
            for pid, product in sorted(
                found_products.items(), key=lambda x: x[1]["score"], reverse=True
            ):
                f.write(f"  {product['item_name']}: {product['score']} \n")
                f.write(f"       {product['image_paths']} \n")
                f.write(f"       {product['llm_str']} \n")
                f.write("\n")
            f.write("======================================================\n\n")
