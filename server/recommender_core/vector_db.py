
import faiss
import json
import logging

from pathlib import Path


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