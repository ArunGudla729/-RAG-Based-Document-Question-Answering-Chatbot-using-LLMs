"""Dense embedding generation via sentence-transformers.

Wraps the ``all-MiniLM-L6-v2`` model (384-dim, fast, strong baseline for
semantic search). The model is loaded lazily and cached so that repeated
calls within a session do not re-download or re-instantiate it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimensionality of all-MiniLM-L6-v2


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load and cache a SentenceTransformer model by name."""
    from sentence_transformers import SentenceTransformer  # lazy import

    return SentenceTransformer(model_name)


class Embedder:
    """Encodes text into L2-normalised dense vectors.

    Vectors are normalised so that an inner-product FAISS index is
    equivalent to cosine similarity search.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = _load_model(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into a (n, dim) float32 array."""
        if not texts:
            return np.empty((0, self.dim), dtype="float32")
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string into a (1, dim) float32 array."""
        return self.encode([text])
