"""Prompt-engineering strategies for grounded, low-hallucination answers.

This module defines several *system prompt strategies* that constrain the
LLM to answer only from retrieved context. They differ in how strictly they
enforce grounding and how they format citations, and can be A/B compared to
study their effect on answer quality and hallucination rate.

Strategies
----------
strict_grounded   : refuse unless the answer is fully supported by context;
                    inline [Source N] citations. (default)
cite_first        : require an explicit citation before every claim.
cot_grounded      : brief chain-of-thought over the context before answering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .vector_store import RetrievedChunk

# Returned verbatim when no context clears the relevance threshold.
INSUFFICIENT_CONTEXT_REPLY = (
    "I don't know based on the provided documents."
)


def format_context(retrieved: List[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    blocks = []
    for i, item in enumerate(retrieved, start=1):
        label = item.chunk.citation()
        blocks.append(f"[Source {i} | {label}]\n{item.chunk.text}")
    return "\n\n".join(blocks)


@dataclass
class PromptStrategy:
    """A named system prompt plus a builder for the user-turn content."""

    name: str
    description: str
    system_prompt: str
    build_user: Callable[[str, str], str]


def _default_user(question: str, context: str) -> str:
    return (
        f"Context documents:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


_STRICT_GROUNDED = PromptStrategy(
    name="strict_grounded",
    description=(
        "Strict grounding: answer only from context, cite sources inline, "
        "and decline when context is insufficient."
    ),
    system_prompt=(
        "You are a meticulous document question-answering assistant. "
        "Answer the user's question using ONLY the information in the provided "
        "context documents. Follow these rules without exception:\n"
        "1. If the context does not contain enough information to answer, reply "
        f'exactly: "{INSUFFICIENT_CONTEXT_REPLY}"\n'
        "2. Never use outside knowledge or make assumptions beyond the context.\n"
        "3. Support each claim with an inline citation in the form [Source N], "
        "matching the numbered sources you used.\n"
        "4. Be concise and precise. Do not speculate."
    ),
    build_user=_default_user,
)


_CITE_FIRST = PromptStrategy(
    name="cite_first",
    description=(
        "Citation-first: every sentence must begin with the source it draws "
        "from, maximising traceability."
    ),
    system_prompt=(
        "You are a citation-first document assistant. Answer ONLY from the "
        "provided context. For every sentence in your answer, begin with the "
        "citation [Source N] that supports it, then state the fact. If no "
        "source supports an answer to the question, reply exactly: "
        f'"{INSUFFICIENT_CONTEXT_REPLY}" '
        "Do not introduce any information that is not present in the sources."
    ),
    build_user=_default_user,
)


def _cot_user(question: str, context: str) -> str:
    return (
        f"Context documents:\n{context}\n\n"
        f"Question: {question}\n\n"
        "First, in one or two sentences, reason about which sources are "
        "relevant. Then give a final answer prefixed with 'Answer:' that uses "
        "only the context and cites sources as [Source N]."
    )


_COT_GROUNDED = PromptStrategy(
    name="cot_grounded",
    description=(
        "Chain-of-thought grounding: briefly reason over the context before "
        "committing to a cited answer."
    ),
    system_prompt=(
        "You are an analytical document assistant. Think step by step about the "
        "provided context, but base every conclusion strictly on it. If the "
        "context is insufficient, your final answer must be exactly: "
        f'"{INSUFFICIENT_CONTEXT_REPLY}" '
        "Cite supporting sources as [Source N]. Never rely on outside knowledge."
    ),
    build_user=_cot_user,
)


STRATEGIES: Dict[str, PromptStrategy] = {
    s.name: s for s in (_STRICT_GROUNDED, _CITE_FIRST, _COT_GROUNDED)
}

DEFAULT_STRATEGY = "strict_grounded"


def get_strategy(name: str) -> PromptStrategy:
    """Look up a strategy by name, raising a helpful error if unknown."""
    try:
        return STRATEGIES[name]
    except KeyError:
        valid = ", ".join(STRATEGIES)
        raise KeyError(f"Unknown strategy '{name}'. Choose from: {valid}")
