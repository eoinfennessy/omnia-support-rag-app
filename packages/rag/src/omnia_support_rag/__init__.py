"""RAG pipeline for Omnia Support."""

from omnia_support_rag.rag import (
    CompletionsClient,
    generate_answer,
    render_messages,
    retrieve_context,
    run_rag_pipeline,
)

__all__ = [
    "CompletionsClient",
    "generate_answer",
    "render_messages",
    "retrieve_context",
    "run_rag_pipeline",
]
