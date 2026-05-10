"""Tests for the RAG pipeline module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from omnia_support_rag.rag import generate_answer, render_messages, retrieve_context, run_rag_pipeline
from omnia_support_vectorstore import VectorStore
from omnia_support_vectorstore.models import ScoredVectorRecord, VectorRecordMetadata


def _make_scored_record(
    *,
    document_id: str = "doc-1",
    chunk: str = "Some text",
    chunk_index: int = 0,
    total_chunks: int = 1,
    distance: float = 0.1,
) -> ScoredVectorRecord:
    metadata: VectorRecordMetadata = {
        "document_id": document_id,
        "type": "faq",
        "category": "General",
        "date": "2025-01-01",
        "product_category": "laptops",
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    return ScoredVectorRecord(
        id=f"{document_id}-chunk-{chunk_index}",
        chunk=chunk,
        metadata=metadata,
        distance=distance,
    )


def _make_prompt(template: list) -> MagicMock:
    prompt = MagicMock()
    prompt.template = template
    return prompt


def _make_completion(content: str | None = "The answer") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestRetrieveContext:
    def test_returns_page_content_and_metadata(self):
        store = MagicMock(spec=VectorStore)
        record = _make_scored_record()
        store.query.return_value = [record]

        result = retrieve_context(store, "test query", 3)

        assert len(result) == 1
        assert result[0]["page_content"] == "Some text"
        assert result[0]["metadata"]["document_id"] == "doc-1"
        assert result[0]["distance"] == 0.1
        store.query.assert_called_once_with("test query", 3)

    def test_multiple_results_preserves_order(self):
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [
            _make_scored_record(chunk="A", document_id="d1", distance=0.1),
            _make_scored_record(chunk="B", document_id="d2", distance=0.5),
            _make_scored_record(chunk="C", document_id="d3", distance=0.9),
        ]

        result = retrieve_context(store, "query", 3)

        assert [r["page_content"] for r in result] == ["A", "B", "C"]
        assert [r["distance"] for r in result] == [0.1, 0.5, 0.9]

    def test_empty_results(self):
        store = MagicMock(spec=VectorStore)
        store.query.return_value = []

        result = retrieve_context(store, "query", 5)

        assert result == []

    def test_metadata_passed_through(self):
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [
            _make_scored_record(document_id="policy-001", chunk_index=2, total_chunks=5),
        ]

        result = retrieve_context(store, "query", 1)

        meta = result[0]["metadata"]
        assert meta["document_id"] == "policy-001"
        assert meta["chunk_index"] == 2
        assert meta["total_chunks"] == 5
        assert meta["type"] == "faq"
        assert result[0]["distance"] == 0.1


class TestRenderMessages:
    def test_single_variable_substitution(self):
        prompt = _make_prompt([{"role": "user", "content": "Hello {{name}}"}])

        result = render_messages(prompt, name="World")

        assert result == [{"role": "user", "content": "Hello World"}]

    def test_multiple_messages_and_variables(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "You are a helper."},
                {"role": "developer", "content": "Context: {{context}}"},
                {"role": "user", "content": "{{question}}"},
            ]
        )

        result = render_messages(prompt, context="some context", question="What?")

        assert len(result) == 3
        assert result[0]["content"] == "You are a helper."
        assert result[1]["content"] == "Context: some context"
        assert result[2]["content"] == "What?"

    def test_preserves_role_field(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ]
        )

        result = render_messages(prompt)

        assert [m["role"] for m in result] == ["system", "user"]

    def test_no_variables_passes_through_unchanged(self):
        prompt = _make_prompt([{"role": "system", "content": "Static text"}])

        result = render_messages(prompt)

        assert result[0]["content"] == "Static text"

    def test_non_dict_entries_filtered(self):
        prompt = _make_prompt([{"role": "user", "content": "ok"}, "not a dict"])

        result = render_messages(prompt)

        assert len(result) == 1
        assert result[0]["content"] == "ok"

    def test_history_spliced_between_preamble_and_current_turn(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "developer", "content": "Context: {{context}}"},
                {"role": "user", "content": "{{question}}"},
            ]
        )
        history = [
            {"role": "developer", "content": "Context: old context"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]

        result = render_messages(prompt, history=history, context="new context", question="new question")  # ty: ignore[invalid-argument-type]

        assert len(result) == 6
        assert result[0] == {"role": "system", "content": "System prompt"}
        assert result[1] == {"role": "developer", "content": "Context: old context"}
        assert result[2] == {"role": "user", "content": "old question"}
        assert result[3] == {"role": "assistant", "content": "old answer"}
        assert result[4] == {"role": "developer", "content": "Context: new context"}
        assert result[5] == {"role": "user", "content": "new question"}

    def test_history_filters_out_system_messages(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "{{question}}"},
            ]
        )
        history = [
            {"role": "system", "content": "Should be filtered"},
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ]

        result = render_messages(prompt, history=history, question="new question")  # ty: ignore[invalid-argument-type]

        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_none_history_same_as_no_history(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "{{question}}"},
            ]
        )

        with_none = render_messages(prompt, history=None, question="q")
        without = render_messages(prompt, question="q")

        assert with_none == without


class TestGenerateAnswer:
    def test_returns_response_content(self):
        client = MagicMock()
        client.create.return_value = _make_completion("The answer")
        messages = [{"role": "user", "content": "Hi"}]

        result = generate_answer(client, messages, "gpt-4")  # ty: ignore[invalid-argument-type]

        assert result == "The answer"

    def test_returns_empty_string_when_content_is_none(self):
        client = MagicMock()
        client.create.return_value = _make_completion(None)

        result = generate_answer(client, [], "gpt-4")

        assert result == ""

    def test_passes_correct_parameters(self):
        client = MagicMock()
        client.create.return_value = _make_completion()
        messages = [{"role": "user", "content": "test"}]

        generate_answer(client, messages, "test-model")  # ty: ignore[invalid-argument-type]

        client.create.assert_called_once_with(
            model="test-model",
            messages=messages,
            temperature=0,
        )


@patch("omnia_support_rag.rag.mlflow.genai.load_prompt")
class TestRunRagPipeline:
    def test_orchestrates_retrieve_render_generate(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [
            _make_scored_record(chunk="chunk A"),
            _make_scored_record(chunk="chunk B", document_id="doc-2"),
        ]
        client = MagicMock()
        client.create.return_value = _make_completion("answer text")

        result = run_rag_pipeline(
            "my question",
            completions_client=client,
            store=store,
            prompt_uri="prompts:/test@prod",
            n_results=5,
            model="gpt-4",
        )

        assert result["answer"] == "answer text"
        assert result["retrieved_context"] == "chunk A\n---\nchunk B"
        assert len(result["retrieved_records"]) == 2

    def test_calls_load_prompt_with_uri(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        store = MagicMock(spec=VectorStore)
        store.query.return_value = []
        client = MagicMock()
        client.create.return_value = _make_completion()

        run_rag_pipeline(
            "q",
            completions_client=client,
            store=store,
            prompt_uri="prompts:/my_prompt@prod",
            n_results=1,
            model="gpt-4",
        )

        mock_load_prompt.assert_called_once_with("prompts:/my_prompt@prod")

    def test_passes_history_to_render_messages(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "system", "content": "System"},
                {"role": "developer", "content": "{{context}}"},
                {"role": "user", "content": "{{question}}"},
            ]
        )
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [_make_scored_record(chunk="new chunk")]
        client = MagicMock()
        client.create.return_value = _make_completion("answer")
        history = [
            {"role": "developer", "content": "old context"},
            {"role": "user", "content": "old q"},
            {"role": "assistant", "content": "old a"},
        ]

        run_rag_pipeline(
            "new q",
            completions_client=client,
            store=store,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
            history=history,  # ty: ignore[invalid-argument-type]
        )

        messages = client.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "System"}
        assert messages[1] == {"role": "developer", "content": "old context"}
        assert messages[2] == {"role": "user", "content": "old q"}
        assert messages[3] == {"role": "assistant", "content": "old a"}
        assert messages[4] == {"role": "developer", "content": "new chunk"}
        assert messages[5] == {"role": "user", "content": "new q"}

    @patch("omnia_support_rag.rag.reformulate_query")
    def test_reformulates_question_when_prompt_uri_provided(self, mock_reformulate, mock_load_prompt):
        mock_reformulate.return_value = "standalone question"
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [_make_scored_record(chunk="chunk")]
        client = MagicMock()
        client.create.return_value = _make_completion("answer")
        history = [
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "reply"},
        ]

        run_rag_pipeline(
            "follow-up",
            completions_client=client,
            store=store,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
            history=history,  # ty: ignore[invalid-argument-type]
            reformulate_prompt_uri="prompts:/reformulate@prod",
        )

        mock_reformulate.assert_called_once_with(
            history,
            "follow-up",
            completions_client=client,
            model="gpt-4",
            prompt_uri="prompts:/reformulate@prod",
        )
        store.query.assert_called_once_with("standalone question", 1)

    def test_skips_reformulation_without_prompt_uri(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [_make_scored_record(chunk="chunk")]
        client = MagicMock()
        client.create.return_value = _make_completion("answer")

        run_rag_pipeline(
            "my question",
            completions_client=client,
            store=store,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
        )

        store.query.assert_called_once_with("my question", 1)

    def test_context_joined_with_separator(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}}"},
            ]
        )
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [
            _make_scored_record(chunk="first"),
            _make_scored_record(chunk="second", document_id="doc-2"),
            _make_scored_record(chunk="third", document_id="doc-3"),
        ]
        client = MagicMock()
        client.create.return_value = _make_completion()

        result = run_rag_pipeline(
            "q",
            completions_client=client,
            store=store,
            prompt_uri="prompts:/p@prod",
            n_results=3,
            model="gpt-4",
        )

        assert result["retrieved_context"] == "first\n---\nsecond\n---\nthird"
