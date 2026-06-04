import numpy as np

from .product_candidate import ProductCandidate
from .product_images import all_img_paths
from .vector_db import load_vector_db

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
                found_products[pid] = ProductCandidate.from_blurb(
                    product_id=pid,
                    blurb=blurb,
                    score=score,
                    image_paths=image_paths,
                )
            else:
                found_products[pid] = found_products[pid].with_additional_image_paths(
                    image_paths
                )
        return found_products