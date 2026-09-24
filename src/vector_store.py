"""FAISS-backed vector store for semantic similarity retrieval.

Uses an inner-product index (``IndexFlatIP``) over L2-normalised vectors,
which is mathematically equivalent to cosine similarity. Each vector is
paired with its source :class:`~src.pdf_processor.Chunk` so retrieval can
return both the score and full provenance.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .pdf_processor import Chunk


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search, with its score."""

    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


class VectorStore:
    """A thin wrapper around a FAISS inner-product index plus chunk metadata."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        """Add embeddings and their corresponding chunks to the index."""
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of dim {self.dim}, got {vectors.shape[1]}"
            )
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[RetrievedChunk]:
        """Return the top-k most similar chunks to a query vector."""
        if self.is_empty:
            return []
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(query_vector, k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than k exist
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results

    def sources(self) -> List[str]:
        """Distinct source document names currently indexed."""
        return sorted({c.source for c in self.chunks})

    # --- persistence -----------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist the index and chunk metadata to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as fh:
            pickle.dump({"dim": self.dim, "chunks": self.chunks}, fh)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a previously saved vector store from a directory."""
        directory = Path(directory)
        with open(directory / "chunks.pkl", "rb") as fh:
            payload = pickle.load(fh)
        store = cls(dim=payload["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.chunks = payload["chunks"]
        return store
