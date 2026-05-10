import csv
import json
import logging
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from imagebind import data
from imagebind.models.imagebind_model import ModalityType, imagebind_huge
from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import ChatOpenAI

# Suppress warnings from torchvision.transforms._functional_video (ImageBind import's fault.)
warnings.filterwarnings("ignore")


PERSONALITIES = [
    "Mae West",
    "Pentecostal preacher",
    "1920s flapper",
    "1950s beatnik",
    "1960s hippie",
    "Puritan preacher",
    "Damon Runyon character",
    "Shakespearean character",
    "Dickensian character",
    "1920s gangster",
    "1950s greaser",
    "Edward Bulwer-Lytton character",
]


def elapsed_time_string(start_time, stop_time):
    delta_time = stop_time - start_time
    minutes, seconds = divmod(delta_time.seconds, 60)
    return minutes, seconds, f"{minutes} minutes, {seconds} seconds"


def normalize(vectors):
    norms = torch.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def load_vector_db(index_path, blurbs_path, product_ids_path):
    """Load vector-search artifacts and product metadata.

    The vector DB generator writes a FAISS index plus a row-aligned
    ``product_ids.json`` file. Row ``i`` in the FAISS index corresponds to
    ``product_ids[i]``.
    """
    index_path = Path(index_path)
    blurbs_path = Path(blurbs_path)
    product_ids_path = Path(product_ids_path)

    if not index_path.is_file():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not product_ids_path.is_file():
        raise FileNotFoundError(f"Product IDs file not found: {product_ids_path}")
    if not blurbs_path.is_file():
        raise FileNotFoundError(f"Product blurbs file not found: {blurbs_path}")

    index = faiss.read_index(str(index_path))
    logging.info(f"Vector DB info: # of vectors = {index.ntotal}, dims = {index.d}")

    with open(product_ids_path, "r", encoding="utf-8") as f:
        product_ids = json.load(f)
        logging.info(f"Vector DB product ID count: {len(product_ids)}")

    if index.ntotal != len(product_ids):
        raise ValueError(
            f"Vector DB artifact mismatch: FAISS index contains {index.ntotal} "
            f"vectors, but product_ids.json contains {len(product_ids)} product IDs."
        )

    with open(blurbs_path, "r", encoding="utf-8") as f:
        logging.info(f"Reading {blurbs_path}")
        blurbs = json.load(f)
        logging.info(f"Blurbs loaded for {len(blurbs)} products.")

    return index, product_ids, blurbs


# Warning: I don't recommend trying to simplify this code.
def all_img_ids(blurb):
    return (
        [blurb.get("main_image_id")]
        if isinstance(blurb.get("main_image_id"), str)
        else (blurb.get("main_image_id") or [])
    ) + (
        [blurb.get("other_image_id")]
        if isinstance(blurb.get("other_image_id"), str)
        else (blurb.get("other_image_id") or [])
    )


def all_img_paths(blurb, image_id_to_path):
    image_paths = []
    for img_id in all_img_ids(blurb):
        if img_id not in image_id_to_path:
            logging.warning(
                "Skipping image_id %s because it was not found in images.csv",
                img_id,
            )
            continue
        image_paths.append(image_id_to_path[img_id])
    return image_paths


def serialize_convo(conversation_history):
    return [
        {"type": msg.__class__.__name__, "content": msg.content}
        for msg in conversation_history
    ]


class ProductVectorStore:
    """Search row-aligned product vectors and return product metadata."""

    def __init__(self, faiss_index, product_ids, blurbs):
        self.faiss_index = faiss_index
        self.product_ids = product_ids
        self.blurbs = blurbs

    @classmethod
    def from_paths(cls, index_path, blurbs_path, product_ids_path):
        faiss_index, product_ids, blurbs = load_vector_db(
            index_path=index_path,
            blurbs_path=blurbs_path,
            product_ids_path=product_ids_path,
        )
        return cls(faiss_index=faiss_index, product_ids=product_ids, blurbs=blurbs)

    def search(self, embedded_query, top_k, image_id_to_path):
        distances, indices = self.faiss_index.search(
            np.array([embedded_query]).astype(np.float32),
            k=top_k,
        )
        return self._deduplicate_products(distances, indices, image_id_to_path)

    def _deduplicate_products(self, distances, indices, image_id_to_path):
        found_products = {}
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            if idx >= len(self.product_ids):
                raise IndexError(
                    f"FAISS returned row {idx}, but only "
                    f"{len(self.product_ids)} product IDs are loaded."
                )

            pid = self.product_ids[int(idx)]
            blurb = self.blurbs[pid]
            image_paths = all_img_paths(blurb, image_id_to_path)

            if pid not in found_products:
                found_products[pid] = {
                    "item_name": blurb["item_name"],
                    "score": float(score),
                    "image_paths": image_paths,
                    "product_type": blurb["feature_fields"]["product_type"],
                    "llm_str": blurb["llm_str"],
                }
            else:
                found_products[pid]["image_paths"] = found_products[pid]["image_paths"] + [
                    new_img
                    for new_img in image_paths
                    if new_img not in found_products[pid]["image_paths"]
                ]
        return found_products


class QueryEmbedder(Embeddings):
    def __init__(self, ibind_model, device):
        self.ibind_model = ibind_model
        self.device = device

    def embed_documents(self, texts):
        logging.info(f"searching multiple strings: {texts}")
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        logging.info(f"embedding one string: {text}")
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([text], self.device)
        }
        with torch.no_grad():
            outputs = self.ibind_model(inputs)
            normalized = normalize(outputs[ModalityType.TEXT])
            logging.info(f"text embedding's shape: {outputs[ModalityType.TEXT].shape}")
            logging.info(f"after normalization.  : {normalized.shape}")
        return normalized.cpu().numpy().flatten()


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

        self.product_store, self.db_load_time = self._load_database(
            vector_db_output_dir=vector_db_output_dir,
            vector_backend=vector_backend,
            index_path=index_path,
            blurbs_path=blurbs_path,
            product_ids_path=product_ids_path,
        )

        self.ibind_model, self.embed_load_time = self._load_imagebind_model()
        self.query_embedder = QueryEmbedder(self.ibind_model, self.device)

        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        image_id_to_path = {}
        with open(images_csv_path, mode="r") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                image_id_to_path[row["image_id"]] = row["path"]
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

    def generate_reply(self, user_input):
        logging.info(f"\n\n\nUser input: {user_input}")
        start_time = datetime.now()

        self.conversation_history.append(HumanMessage(content=user_input))

        llm_search_query = self._build_search_query()
        found_products = self._search_products(llm_search_query, top_k=10)
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
        logging.info(
            f"Took {minutes} minutes, {seconds} seconds to prepare a response to the user's message."
        )

        max_score_dict = max(found_products.values(), key=lambda x: x["score"])

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
        }

    def _build_search_query(self):
        caption_please = (
            "Based on the current conversation, what sort of product should we search for? "
            "Please ignore your personality and limit your answer to a maximum of 10 words - "
            "words an automated search system would find useful."
        )
        llm_search_query = self.chat_openai.invoke(
            self.conversation_history + [SystemMessage(content=caption_please)]
        ).content
        logging.info(f"LLM's suggested search query: {llm_search_query}")
        return llm_search_query

    def _search_products(self, search_query, top_k=10):
        embedded_query = self.query_embedder.embed_query(search_query).flatten().tolist()
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
        dive_deeper = "DIVE DEEPER" in llm_response
        if (
            not dive_deeper
            and "WRONG TRACK" not in llm_response
            and "<" in llm_response
            and ">" in llm_response
        ):
            chosen_pid = llm_response.split("<")[1].split(">") [0]
            chosen_product = found_products.get(chosen_pid, None)
        else:
            chosen_pid = None
            chosen_product = {}

        return chosen_pid, chosen_product, dive_deeper

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
        if dive_deeper:
            reprompt_str = (
                "Let's continue the conversation so we can find better product matches. "
                "Don't recommend any specific products - we're trying to learn more so we can make better recommendations. "
                "For context, here are the latest top search results, which we find promising and want to be able to dive deeper into:\n"
                f"{source_knowledge}"
            )
            logging.info("Asking the user for more details")
        else:
            reprompt_str = (
                "Let's continue the conversation to see if we can find a search area that's better served by our stock. "
                "Don't recommend any specific products - we're trying to learn more so we can see if we have anything that suits the buyer. "
                "You may want to apologize to them, since we're not finding any relevant products in our searches so far. "
                "For context, here are the latest top search results, which we're finding lacking:\n"
                f"{source_knowledge}"
            )
            logging.info("Redirect the user to another search area")

        return reprompt_str, "No-rec LLM Response"

    def _format_source_knowledge(self, found_products):
        return "\n\n;\n\n".join(
            [
                f"product_id: {pid}, item_name: {info['item_name']}"
                for pid, info in found_products.items()
            ]
        )

    def _build_augmented_prompt(self, source_knowledge):
        return (
            "Your next output must be surrounded by <> symbols, filled according to 1 of the following 3 options, with no trailing period:\n"
            "A. If any of the listed >>Suggested Products<< below are relevant and suggestible, "
            "please output its product ID (NOT it's product NAME!), like: <B071K17SWD>.\n"
            "B. If you think that refining search results won't lead to better results, please output: <WRONG TRACK>.\n"
            "C. If you think that search results are promising and there's room to ask the user for more specificity, please output: <DIVE DEEPER>.\n"
            ">>Suggested Products<<:"
            f"{source_knowledge}"
        )

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
