---
title: AI Document Q&A (RAG)
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
license: mit
---

# 📄 AI-Powered Document Q&A System (RAG)

An end-to-end **Retrieval-Augmented Generation** pipeline that lets you upload
PDFs and ask questions about them in natural language. Answers are grounded in
the source documents, cite the exact chunks they came from, and return
*"I don't know"* when the documents don't contain the answer — instead of
hallucinating.

> Built with Python, PyMuPDF, Sentence-Transformers, FAISS, an LLM API
> (Anthropic Claude or OpenAI), and a Streamlit UI.

---

## ✨ Features

- **End-to-end RAG pipeline** — PDF text extraction → chunking → dense
  embeddings → FAISS semantic retrieval → grounded LLM generation.
- **Multi-PDF support** — index several documents at once across diverse types
  (research papers, invoices, policy documents).
- **Chunk-level source attribution** — every answer is paired with the exact
  source chunks (filename + page + similarity score).
- **Anti-hallucination guardrails** — a relevance threshold short-circuits to
  *"I don't know"* before the LLM is even called when retrieval finds nothing
  relevant.
- **Prompt-engineering strategies** — three swappable system-prompt strategies
  (`strict_grounded`, `cite_first`, `cot_grounded`) you can A/B compare.
- **Provider-agnostic** — switch between Anthropic and OpenAI from the sidebar.
- **Streamlit UI + CLI** — use the interactive app or the headless `cli.py`.
- **Tested** — fast unit tests with fakes (no network/model downloads needed).

---

## 🏗️ Architecture

```
                ┌──────────────┐
   PDF upload ─►│ pdf_processor│  extract (PyMuPDF) + clean + chunk (overlap)
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │  embeddings  │  all-MiniLM-L6-v2 → 384-d normalised vectors
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │ vector_store │  FAISS IndexFlatIP (cosine) + chunk metadata
                └──────┬───────┘
                       ▼
   question ─► retrieve top-k ─► relevance gate ─► ┌──────────┐
                                                   │ prompts  │ grounded system prompt
                                                   └────┬─────┘
                                                        ▼
                                                  ┌───────────┐
                                                  │ llm_client│ Claude / OpenAI
                                                  └────┬──────┘
                                                       ▼
                                          grounded answer + citations
```

The orchestration lives in [`src/rag_pipeline.py`](src/rag_pipeline.py).

---

## 🚀 Quickstart

### 1. Install

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add an API key

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

You can also paste the key directly into the app's sidebar at runtime.

### 3. Run the app

```bash
streamlit run app.py
```

Upload PDFs, pick a model and prompt strategy in the sidebar, and start asking
questions.

### CLI alternative

```bash
python cli.py --pdf report.pdf --pdf invoice.pdf -q "What is the total due?"
# or interactive:
python cli.py --pdf paper.pdf
```

---

## 🧠 Prompt-engineering strategies

| Strategy          | What it does |
|-------------------|--------------|
| `strict_grounded` | Answers only from context, inline `[Source N]` citations, refuses when context is insufficient. **(default)** |
| `cite_first`      | Every sentence begins with the source it draws from — maximum traceability. |
| `cot_grounded`    | Briefly reasons over the context before committing to a cited answer. |

Defined in [`src/prompts.py`](src/prompts.py). All three enforce the same
refusal contract so you can compare grounding behaviour fairly.

---

## ⚙️ How grounding works

1. The question is embedded and matched against the FAISS index.
2. If the **top chunk's cosine similarity is below the threshold** (default
   `0.25`), the pipeline returns *"I don't know based on the provided
   documents."* without calling the LLM — cheap and hallucination-proof.
3. Otherwise the top-k chunks are formatted into a numbered, citable context
   block and passed to the LLM under a strict system prompt.

Tune `top_k` and the relevance threshold from the sidebar.

---

## 🧪 Tests

```bash
pip install pytest
pytest -q
```

Tests use a fake embedder and fake LLM, so they run offline in milliseconds.

---

## 📁 Project structure

```
rag-document-qa/
├── app.py                 # Streamlit UI
├── cli.py                 # Headless CLI
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
├── src/
│   ├── pdf_processor.py   # PyMuPDF extraction + chunking
│   ├── embeddings.py      # sentence-transformers (all-MiniLM-L6-v2)
│   ├── vector_store.py    # FAISS index + persistence
│   ├── prompts.py         # prompt-engineering strategies
│   ├── llm_client.py      # Anthropic / OpenAI clients
│   └── rag_pipeline.py    # orchestration
├── tests/test_pipeline.py
└── examples/sample_questions.md
```

---

## 🛣️ Possible extensions

- Re-ranking retrieved chunks with a cross-encoder.
- Hybrid (BM25 + dense) retrieval.
- Conversation-aware retrieval (rewrite follow-up questions using history).
- Persisting the FAISS index to disk between sessions (`VectorStore.save/load`).

---

## 📄 License

MIT — see [LICENSE](LICENSE).
