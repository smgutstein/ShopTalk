import numpy as np

def combine_query_embeddings(embeddings):
    """Average one or more query embeddings and return a normalized vector."""
    if not embeddings:
        raise ValueError("At least one query embedding is required.")

    embedding_matrix = np.vstack([np.asarray(embedding, dtype=np.float32) 
                                  for embedding in embeddings])
    combined = embedding_matrix.mean(axis=0)
    norm = np.linalg.norm(combined)
    if norm == 0:
        raise ValueError("Cannot combine query embeddings with zero vector norm.")
    return combined / norm