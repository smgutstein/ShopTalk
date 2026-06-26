"""Factory functions for constructing the ShopTalk recommender runtime."""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

from .config import RecommenderConfig
from .conversation_policy import ConversationPolicy
from .product_images import load_image_paths_csv
from .shop_talk_recommender import ShopTalkRecommender
from .utils import PERSONALITIES, elapsed_time_string, load_openai_api_key


def build_recommender(config: RecommenderConfig | None = None) -> ShopTalkRecommender:
    """Build a fully configured ``ShopTalkRecommender`` runtime."""
    config = config or RecommenderConfig()
    device = choose_device(config)
    api_key = configure_runtime_environment()

    product_store, db_load_time = load_product_store(config)
    query_embedder, embed_load_time = build_query_embedder(device)
    personality = choose_personality(config.personality_index)
    conversation_policy = build_conversation_policy(
        api_key=api_key,
        model_name=config.model_name,
        temperature=config.temperature,
    )
    image_id_to_path, image_path_load_time = load_image_paths(config.images_csv_path)

    return ShopTalkRecommender(
        config=config,
        product_store=product_store,
        query_embedder=query_embedder,
        conversation_policy=conversation_policy,
        image_id_to_path=image_id_to_path,
        personality=personality,
        db_load_time=db_load_time,
        embed_load_time=embed_load_time,
        image_path_load_time=image_path_load_time,
    )


def choose_device(config: RecommenderConfig) -> str:
    """Choose the torch device for embedding queries."""
    import torch

    device = "cpu" if config.force_cpu else "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    return device


def configure_runtime_environment() -> str:
    """Configure process-level environment needed by runtime dependencies."""
    api_key = load_openai_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return api_key


def load_product_store(config: RecommenderConfig):
    """Load the product vector store and return it with a timing string."""
    logging.info("Loading Database...")
    start_time = datetime.now()

    if config.vector_backend != "faiss":
        raise ValueError(
            "ShopTalkRecommender currently supports only the FAISS vector backend. "
            f"Got vector_backend={config.vector_backend!r}."
        )

    backend_dir = Path(config.vector_db_output_dir) / config.vector_backend
    index_path = config.index_path or backend_dir / "embeddings.faiss"
    product_ids_path = config.product_ids_path or backend_dir / "product_ids.json"

    from .product_vector_store import ProductVectorStore

    product_store = ProductVectorStore.from_paths(
        index_path=index_path,
        blurbs_path=config.blurbs_path,
        product_ids_path=product_ids_path,
    )
    stop_time = datetime.now()
    minutes, seconds, load_time = elapsed_time_string(start_time, stop_time)
    logging.info(f"{minutes} minutes, {seconds} seconds")
    return product_store, load_time


def build_query_embedder(device: str):
    """Load ImageBind and wrap it in a query embedder."""
    from imagebind.models.imagebind_model import imagebind_huge

    from .query_embedder import QueryEmbedder

    logging.info("Loading ImageBind model...")
    start_time = datetime.now()
    ibind_model = imagebind_huge(pretrained=True)
    ibind_model.eval()
    ibind_model.to(device)
    stop_time = datetime.now()
    minutes, seconds, load_time = elapsed_time_string(start_time, stop_time)
    logging.info(f"{minutes} minutes, {seconds} seconds")
    return QueryEmbedder(ibind_model, device), load_time


def build_conversation_policy(api_key: str, model_name: str, temperature: float) -> ConversationPolicy:
    """Build the LLM-backed conversation policy."""
    from langchain_openai import ChatOpenAI

    chat_openai = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )
    return ConversationPolicy(chat_openai)


def load_image_paths(images_csv_path):
    """Load image ID to path mappings and return them with a timing string."""
    logging.info("Loading Image paths...")
    start_time = datetime.now()
    image_id_to_path = load_image_paths_csv(images_csv_path)
    stop_time = datetime.now()
    minutes, seconds, load_time = elapsed_time_string(start_time, stop_time)
    logging.info(f"{minutes} minutes, {seconds} seconds")
    return image_id_to_path, load_time


def choose_personality(personality_index: int) -> str:
    """Choose the configured personality, randomizing when requested."""
    if personality_index == -1 or personality_index >= len(PERSONALITIES):
        personality = random.choice(PERSONALITIES)
        logging.info(f"Random Personality: {personality}")
    else:
        personality = PERSONALITIES[personality_index]
    logging.info(f" {personality}")
    return personality
