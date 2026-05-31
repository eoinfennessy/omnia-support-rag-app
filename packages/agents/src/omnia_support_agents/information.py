"""Information agent: answers queries via the RAG pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

from omnia_support_rag.query_interpretation import PassthroughQueryInterpreter, QueryInterpreter
from omnia_support_rag.rag import run_rag_pipeline

from .models import InformationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai.types.chat import ChatCompletionMessageParam

    from omnia_support_rag.context_processing import ContextProcessor
    from omnia_support_rag.rag import CompletionsClient, Rescorer, Retriever


class InformationAgent:
    """Handles information queries by delegating to the RAG pipeline."""

    def __init__(
        self,
        *,
        completions_client: CompletionsClient,
        retriever: Retriever,
        prompt_uri: str,
        n_results: int,
        model: str,
        query_interpreter: QueryInterpreter = PassthroughQueryInterpreter(),  # noqa: B008 — stateless
        rescorers: Sequence[Rescorer] | None = None,
        context_processors: Sequence[ContextProcessor] | None = None,
        use_metadata_filter: bool = True,
    ) -> None:
        """Create an information agent."""
        self._completions_client = completions_client
        self._retriever = retriever
        self._prompt_uri = prompt_uri
        self._n_results = n_results
        self._model = model
        self._query_interpreter = query_interpreter
        self._rescorers = rescorers
        self._context_processors = context_processors
        self._use_metadata_filter = use_metadata_filter

    @mlflow.trace(name="information")
    def handle(
        self,
        question: str,
        history: list[ChatCompletionMessageParam] | None = None,
    ) -> InformationResult:
        """Answer an information query using the RAG pipeline."""
        rag_result = run_rag_pipeline(
            question,
            completions_client=self._completions_client,
            retriever=self._retriever,
            prompt_uri=self._prompt_uri,
            n_results=self._n_results,
            model=self._model,
            history=history,
            query_interpreter=self._query_interpreter,
            rescorers=self._rescorers,
            context_processors=self._context_processors,
            use_metadata_filter=self._use_metadata_filter,
        )

        return InformationResult(
            query=question,
            answer=rag_result.answer,
            context=rag_result.retrieved_chunks,
        )
