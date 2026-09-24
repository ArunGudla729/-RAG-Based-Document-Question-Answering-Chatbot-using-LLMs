"""Streamlit front-end for the RAG Document Q&A system.

Run locally:
    streamlit run app.py

Features:
    * Multi-PDF upload and indexing
    * Conversational querying with chat history
    * Chunk-level source attribution (expandable)
    * Selectable LLM provider/model and prompt strategy
"""

from __future__ import annotations

import os

import streamlit as st

from src.llm_client import ANTHROPIC_MODELS, OPENAI_MODELS, build_client
from src.prompts import STRATEGIES
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Sidebar: configuration
# --------------------------------------------------------------------------
def sidebar_config() -> dict:
    st.sidebar.header("⚙️ Configuration")

    provider = st.sidebar.selectbox("LLM provider", ["anthropic", "openai"])
    if provider == "anthropic":
        model = st.sidebar.selectbox(
            "Model", list(ANTHROPIC_MODELS), format_func=lambda m: f"{m}"
        )
        env_key = "ANTHROPIC_API_KEY"
    else:
        model = st.sidebar.selectbox("Model", list(OPENAI_MODELS))
        env_key = "OPENAI_API_KEY"

    key_present = bool(os.getenv(env_key))
    api_key = st.sidebar.text_input(
        f"{env_key}"
        + (" (loaded from env ✓)" if key_present else ""),
        type="password",
        help="Leave blank to use the environment variable.",
    )

    strategy = st.sidebar.selectbox(
        "Prompt strategy",
        list(STRATEGIES),
        help="\n".join(f"{n}: {s.description}" for n, s in STRATEGIES.items()),
    )
    top_k = st.sidebar.slider("Chunks to retrieve (k)", 1, 10, 4)
    min_score = st.sidebar.slider(
        "Relevance threshold", 0.0, 1.0, 0.25, 0.05,
        help="Below this cosine similarity, the assistant answers 'I don't know'.",
    )

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key or None,
        "strategy": strategy,
        "top_k": top_k,
        "min_score": min_score,
    }


# --------------------------------------------------------------------------
# Pipeline lifecycle
# --------------------------------------------------------------------------
def ensure_pipeline(cfg: dict) -> RAGPipeline | None:
    """Build the pipeline once and keep it in session state."""
    signature = (cfg["provider"], cfg["model"], bool(cfg["api_key"]))
    if (
        "pipeline" in st.session_state
        and st.session_state.get("pipeline_sig") == signature
    ):
        pipe = st.session_state.pipeline
        pipe.top_k = cfg["top_k"]
        pipe.min_score = cfg["min_score"]
        return pipe

    try:
        with st.spinner("Loading embedding model and LLM client…"):
            llm = build_client(cfg["provider"], cfg["model"], cfg["api_key"])
            pipe = RAGPipeline(llm=llm, top_k=cfg["top_k"], min_score=cfg["min_score"])
    except Exception as exc:  # noqa: BLE001 — surface any setup error to the user
        st.sidebar.error(f"Setup failed: {exc}")
        return None

    st.session_state.pipeline = pipe
    st.session_state.pipeline_sig = signature
    st.session_state.indexed_files = set()
    st.session_state.messages = []
    return pipe


def index_uploads(pipe: RAGPipeline, uploaded_files) -> None:
    new_files = [
        f for f in uploaded_files if f.name not in st.session_state.indexed_files
    ]
    if not new_files:
        return
    for f in new_files:
        with st.spinner(f"Indexing {f.name}…"):
            added = pipe.add_pdf(f.getvalue(), f.name)
            st.session_state.indexed_files.add(f.name)
        st.sidebar.success(f"{f.name}: {added} chunks indexed")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("📄 AI-Powered Document Q&A")
    st.caption(
        "Retrieval-Augmented Generation over your PDFs — grounded answers with "
        "chunk-level source attribution."
    )

    cfg = sidebar_config()
    pipe = ensure_pipeline(cfg)
    if pipe is None:
        st.info("Resolve the configuration error in the sidebar to begin.")
        return

    uploaded = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True
    )
    if uploaded:
        index_uploads(pipe, uploaded)

    if pipe.num_chunks:
        st.success(
            f"Indexed {pipe.num_chunks} chunks from {len(pipe.sources)} "
            f"document(s): {', '.join(pipe.sources)}"
        )
    else:
        st.info("Upload at least one PDF to start asking questions.")
        return

    # Replay chat history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**[Source {i}] {src['label']}** "
                                    f"(similarity {src['score']:.2f})")
                        st.caption(src["text"])

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                answer = pipe.answer(question, strategy=cfg["strategy"])
            except Exception as exc:  # noqa: BLE001
                err = f"Error while generating an answer: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )
                return

        st.markdown(answer.text)
        sources = [
            {
                "label": r.chunk.citation(),
                "score": r.score,
                "text": r.chunk.text[:500] + ("…" if len(r.chunk.text) > 500 else ""),
            }
            for r in answer.retrieved
        ]
        if sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**[Source {i}] {src['label']}** "
                                f"(similarity {src['score']:.2f})")
                    st.caption(src["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "sources": sources}
    )


if __name__ == "__main__":
    main()
