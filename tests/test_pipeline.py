"""Tests for the RAG pipeline components.

These tests avoid network access and heavy model downloads by using a fake
embedder and a fake LLM, so they run fast in CI. Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np

from src.pdf_processor import Chunk, chunk_text
from src.prompts import INSUFFICIENT_CONTEXT_REPLY, format_context, get_strategy
from src.rag_pipeline import RAGPipeline
from src.vector_store import RetrievedChunk, VectorStore


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic bag-of-words embedder over a tiny fixed vocabulary."""

    VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]

    def __init__(self) -> None:
        self.dim = len(self.VOCAB)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for i, word in enumerate(self.VOCAB):
            if word in text.lower():
                v[i] = 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v

    def encode(self, texts):
        return np.vstack([self._vec(t) for t in texts]).astype("float32")

    def encode_one(self, text):
        return self.encode([text])


class FakeLLM:
    """Echoes the prompt so we can assert what was sent."""

    def __init__(self) -> None:
        self.last_system = None
        self.last_user = None

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.last_system = system
        self.last_user = user
        return "ANSWER [Source 1]"


# --------------------------------------------------------------------------
# pdf_processor
# --------------------------------------------------------------------------
def test_chunk_text_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)  # size + small slack
    # consecutive chunks should share some words (overlap)
    first_tail = set(chunks[0].split()[-2:])
    second_head = set(chunks[1].split()[:5])
    assert first_tail & second_head


def test_chunk_text_single_chunk_when_short():
    assert chunk_text("hello world", chunk_size=1000) == ["hello world"]


def test_chunk_citation_label():
    c = Chunk(text="x", source="report.pdf", page=3, chunk_id=0)
    assert c.citation() == "report.pdf p.3"


# --------------------------------------------------------------------------
# vector_store
# --------------------------------------------------------------------------
def test_vector_store_search_ranks_by_similarity():
    emb = FakeEmbedder()
    store = VectorStore(dim=emb.dim)
    chunks = [
        Chunk(text="alpha beta", source="d", page=1, chunk_id=0),
        Chunk(text="gamma delta", source="d", page=1, chunk_id=1),
    ]
    store.add(emb.encode([c.text for c in chunks]), chunks)
    results = store.search(emb.encode_one("alpha"), k=2)
    assert results[0].chunk.text == "alpha beta"
    assert results[0].score >= results[1].score


def test_vector_store_save_load(tmp_path):
    emb = FakeEmbedder()
    store = VectorStore(dim=emb.dim)
    chunks = [Chunk(text="alpha", source="d", page=1, chunk_id=0)]
    store.add(emb.encode(["alpha"]), chunks)
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 1
    assert loaded.chunks[0].text == "alpha"


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
def test_format_context_numbers_sources():
    chunks = [
        RetrievedChunk(Chunk("text one", "a.pdf", 1, 0), 0.9),
        RetrievedChunk(Chunk("text two", "b.pdf", 2, 1), 0.8),
    ]
    ctx = format_context(chunks)
    assert "[Source 1 | a.pdf p.1]" in ctx
    assert "[Source 2 | b.pdf p.2]" in ctx


def test_strategies_contain_refusal_clause():
    for name in ("strict_grounded", "cite_first", "cot_grounded"):
        strat = get_strategy(name)
        assert INSUFFICIENT_CONTEXT_REPLY in strat.system_prompt


# --------------------------------------------------------------------------
# rag_pipeline
# --------------------------------------------------------------------------
def _pipeline_with_data() -> tuple[RAGPipeline, FakeLLM]:
    llm = FakeLLM()
    pipe = RAGPipeline(llm=llm, embedder=FakeEmbedder(), min_score=0.1, top_k=2)
    chunks = [
        Chunk("alpha beta context", "doc.pdf", 1, 0),
        Chunk("gamma delta context", "doc.pdf", 2, 1),
    ]
    pipe.store.add(pipe.embedder.encode([c.text for c in chunks]), chunks)
    return pipe, llm


def test_pipeline_answers_when_relevant():
    pipe, llm = _pipeline_with_data()
    ans = pipe.answer("alpha", strategy="strict_grounded")
    assert ans.grounded is True
    assert ans.text == "ANSWER [Source 1]"
    assert "alpha beta context" in llm.last_user  # context was injected


def test_pipeline_refuses_when_irrelevant():
    pipe, llm = _pipeline_with_data()
    ans = pipe.answer("epsilon", strategy="strict_grounded")  # no overlap
    assert ans.grounded is False
    assert ans.text == INSUFFICIENT_CONTEXT_REPLY
    assert llm.last_user is None  # LLM was never called
