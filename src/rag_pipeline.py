"""End-to-end RAG orchestration.

Ties together PDF processing, embedding, FAISS retrieval, prompt strategies,
and the LLM client. The pipeline keeps an in-memory vector store that can be
incrementally populated from multiple PDFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .embeddings import Embedder
from .llm_client import LLMClient
from .pdf_processor import Chunk, process_pdf
from .prompts import (
    DEFAULT_STRATEGY,
    INSUFFICIENT_CONTEXT_REPLY,
    format_context,
    get_strategy,
)
from .vector_store import RetrievedChunk, VectorStore


@dataclass
class Answer:
    """The result of a query against the indexed documents."""

    text: str
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    grounded: bool = True  # False when retrieval found nothing relevant

    @property
    def citations(self) -> List[str]:
        return [r.chunk.citation() for r in self.retrieved]


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline over user-supplied PDFs.

    Args:
        llm: A configured chat client.
        embedder: Optional pre-built embedder (defaults to all-MiniLM-L6-v2).
        chunk_size: Target chunk size in characters.
        overlap: Chunk overlap in characters.
        top_k: Number of chunks to retrieve per query.
        min_score: Minimum cosine similarity for a chunk to count as relevant.
                   If the best chunk is below this, the pipeline short-circuits
                   to an "I don't know" answer without calling the LLM.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Optional[Embedder] = None,
        chunk_size: int = 1000,
        overlap: int = 150,
        top_k: int = 4,
        min_score: float = 0.25,
    ) -> None:
        self.llm = llm
        self.embedder = embedder or Embedder()
        self.store = VectorStore(dim=self.embedder.dim)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.min_score = min_score

    # --- indexing --------------------------------------------------------

    def add_pdf(self, file_bytes: bytes, source_name: str) -> int:
        """Process and index a single PDF. Returns the number of chunks added."""
        chunks: List[Chunk] = process_pdf(
            file_bytes, source_name, self.chunk_size, self.overlap
        )
        if not chunks:
            return 0
        vectors = self.embedder.encode([c.text for c in chunks])
        self.store.add(vectors, chunks)
        return len(chunks)

    @property
    def num_chunks(self) -> int:
        return len(self.store)

    @property
    def sources(self) -> List[str]:
        return self.store.sources()

    # --- querying --------------------------------------------------------

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for a question."""
        query_vec = self.embedder.encode_one(question)
        return self.store.search(query_vec, k=self.top_k)

    def answer(self, question: str, strategy: str = DEFAULT_STRATEGY) -> Answer:
        """Answer a question with retrieval-augmented generation.

        Args:
            question: The user's question.
            strategy: Name of the prompt strategy to use.
        """
        retrieved = self.retrieve(question)

        # Guardrail: if nothing is relevant enough, refuse without the LLM.
        if not retrieved or retrieved[0].score < self.min_score:
            return Answer(
                text=INSUFFICIENT_CONTEXT_REPLY,
                retrieved=retrieved,
                grounded=False,
            )

        strat = get_strategy(strategy)
        context = format_context(retrieved)
        user_content = strat.build_user(question, context)
        response = self.llm.complete(strat.system_prompt, user_content)

        return Answer(text=response, retrieved=retrieved, grounded=True)
