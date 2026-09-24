"""RAG Document Q&A — core package.

Modules:
    pdf_processor  -- PDF text extraction and chunking (PyMuPDF)
    embeddings     -- dense embeddings via sentence-transformers
    vector_store   -- FAISS index wrapper for semantic retrieval
    prompts        -- prompt-engineering strategies
    llm_client     -- Anthropic / OpenAI chat completion clients
    rag_pipeline   -- orchestration tying everything together
"""

__version__ = "1.0.0"
