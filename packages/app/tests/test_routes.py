"""Tests for API routes."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omnia_support_app.dependencies import AppConfig, AppState
from omnia_support_app.routes import router
from omnia_support_app.session import InMemorySessionStore
from omnia_support_models import (
    Dataset,
    DatasetMetadata,
    Document,
    DocumentType,
    DocumentTypes,
    Priority,
    ProductCategory,
    ResolutionStatus,
)
from omnia_support_rag import PassthroughQueryInterpreter, RagResult
from omnia_support_store.models import ChunkMetadata, ScoredChunk

_TEST_DATASET = Dataset(
    dataset_metadata=DatasetMetadata(
        name="test",
        version="1.0",
        created=datetime.date(2024, 1, 1),
        description="test dataset",
        total_documents=3,
        document_types=DocumentTypes(support_tickets=1, policies=1, faqs=1),
        product_categories=[ProductCategory.ALL],
        geographic_context="test",
        currency="EUR",
        regulatory_context=[],
    ),
    documents=[
        Document(
            document_id="policy_001",
            type=DocumentType.POLICY,
            category="returns",
            date=datetime.date(2024, 1, 1),
            resolution_status=None,
            customer_satisfaction=None,
            product_category=ProductCategory.ALL,
            priority=None,
            content="Return policy content",
        ),
        Document(
            document_id="faq_001",
            type=DocumentType.FAQ,
            category="general",
            date=datetime.date(2024, 1, 1),
            resolution_status=None,
            customer_satisfaction=None,
            product_category=ProductCategory.ALL,
            priority=None,
            content="FAQ content",
        ),
        Document(
            document_id="ticket_001",
            type=DocumentType.SUPPORT_TICKET,
            category="delivery_issue",
            date=datetime.date(2024, 1, 1),
            resolution_status=ResolutionStatus.RESOLVED,
            customer_satisfaction=4.5,
            product_category=ProductCategory.LAPTOPS,
            priority=Priority.MEDIUM,
            content="Ticket content",
        ),
    ],
    test_queries=[],
)


def _create_test_app() -> tuple[FastAPI, AppState]:
    app = FastAPI()
    app.include_router(router)

    state = AppState(
        config=AppConfig(
            prompt_uri="prompts:/test@prod",
            n_results=3,
            llm_model="gpt-4",
        ),
        retriever=MagicMock(),
        rescorers=[],
        context_processors=[],
        completions_client=MagicMock(),
        dataset=_TEST_DATASET,
        query_interpreter=PassthroughQueryInterpreter(),
        sessions=InMemorySessionStore(),
    )
    app.state.app_state = state
    return app, state


def _make_record(
    *,
    document_id: str,
    chunk: str,
    doc_type: str,
    category: str,
    product_category: str = "all",
    chunk_index: int = 0,
) -> ScoredChunk:
    metadata: ChunkMetadata = {
        "document_id": document_id,
        "type": doc_type,
        "category": category,
        "date": "2024-01-01",
        "product_category": product_category,
        "chunk_index": chunk_index,
        "total_chunks": 1,
    }
    return ScoredChunk(
        id=f"{document_id}-chunk-{chunk_index}",
        chunk=chunk,
        metadata=metadata,
        score=0.5,
    )


_POLICY_RECORD = _make_record(
    document_id="policy_001",
    chunk="Return policy content",
    doc_type="policy",
    category="returns",
)

_FAQ_RECORD = _make_record(
    document_id="faq_001",
    chunk="FAQ content",
    doc_type="faq",
    category="general",
)

_TICKET_RECORD = _make_record(
    document_id="ticket_001",
    chunk="Ticket content",
    doc_type="support_ticket",
    category="delivery_issue",
    product_category="laptops",
)


@patch("omnia_support_app.routes.run_rag_pipeline")
class TestChatEndpoint:
    def test_returns_answer(self, mock_pipeline):
        mock_pipeline.return_value = RagResult(
            answer="Our return policy is...",
            retrieved_chunks=[_POLICY_RECORD],
        )
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "What is your return policy?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Our return policy is..."
        assert data["session_id"] == "s1"

    def test_passes_query_interpreter_to_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = RagResult(
            answer="answer",
            retrieved_chunks=[_POLICY_RECORD],
        )
        app, _ = _create_test_app()
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        assert isinstance(mock_pipeline.call_args.kwargs["query_interpreter"], PassthroughQueryInterpreter)

    def test_second_message_passes_history(self, mock_pipeline):
        mock_pipeline.return_value = RagResult(
            answer="answer",
            retrieved_chunks=[_POLICY_RECORD],
        )
        app, _state = _create_test_app()
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "first question"},
        )
        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "follow-up"},
        )

        second_pipeline_call = mock_pipeline.call_args_list[1]
        history = second_pipeline_call.kwargs["history"]
        assert len(history) == 3
        assert history[0]["role"] == "developer"
        assert history[1]["role"] == "user"
        assert history[2]["role"] == "assistant"

    def test_stores_developer_user_assistant_triple(self, mock_pipeline):
        mock_pipeline.return_value = RagResult(
            answer="the answer",
            retrieved_chunks=[_POLICY_RECORD],
        )
        app, state = _create_test_app()
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "my question"},
        )

        history = state.sessions.get_history("s1")
        assert len(history) == 3
        assert history[0] == {"role": "developer", "content": "Return policy content"}
        assert history[1] == {"role": "user", "content": "my question"}
        assert history[2] == {"role": "assistant", "content": "the answer"}

    def test_returns_deduplicated_sources(self, mock_pipeline):
        duplicate_record = _make_record(
            document_id="policy_001",
            chunk="Another chunk from same policy",
            doc_type="policy",
            category="returns",
            chunk_index=1,
        )
        mock_pipeline.return_value = RagResult(
            answer="answer",
            retrieved_chunks=[_POLICY_RECORD, duplicate_record, _FAQ_RECORD],
        )
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        sources = response.json()["sources"]
        assert len(sources) == 2
        assert sources[0] == {"url": "/policies/policy_001", "title": "Returns", "type": "policy"}
        assert sources[1] == {"url": "/faqs/faq_001", "title": "faq_001 - General", "type": "faq"}

    def test_excludes_support_tickets_from_sources(self, mock_pipeline):
        mock_pipeline.return_value = RagResult(
            answer="answer",
            retrieved_chunks=[_TICKET_RECORD, _POLICY_RECORD],
        )
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        sources = response.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["type"] == "policy"


@patch("omnia_support_app.routes.run_rag_pipeline")
class TestClearSessionEndpoint:
    def test_clears_session(self, mock_pipeline):
        mock_pipeline.return_value = RagResult(
            answer="a",
            retrieved_chunks=[_POLICY_RECORD],
        )
        app, state = _create_test_app()
        client = TestClient(app)

        client.post("/api/chat", json={"session_id": "s1", "message": "hi"})
        assert len(state.sessions.get_history("s1")) == 3

        response = client.delete("/api/chat/s1")

        assert response.status_code == 204
        assert state.sessions.get_history("s1") == []

    def test_clear_nonexistent_session_returns_204(self, mock_pipeline):  # noqa: ARG002
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.delete("/api/chat/nonexistent")

        assert response.status_code == 204


class TestPoliciesEndpoint:
    def test_returns_only_policies(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/policies")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["document_id"] == "policy_001"
        assert data[0]["type"] == "policy"

    def test_excludes_other_document_types(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/policies")

        types = {doc["type"] for doc in response.json()}
        assert types == {"policy"}

    def test_get_single_policy(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/policies/policy_001")

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "policy_001"
        assert data["type"] == "policy"

    def test_get_nonexistent_policy_returns_404(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/policies/policy_999")

        assert response.status_code == 404


class TestFaqsEndpoint:
    def test_returns_only_faqs(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/faqs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["document_id"] == "faq_001"
        assert data[0]["type"] == "faq"

    def test_excludes_other_document_types(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/faqs")

        types = {doc["type"] for doc in response.json()}
        assert types == {"faq"}

    def test_get_single_faq(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/faqs/faq_001")

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "faq_001"
        assert data["type"] == "faq"

    def test_get_nonexistent_faq_returns_404(self):
        app, _ = _create_test_app()
        client = TestClient(app)

        response = client.get("/api/faqs/faq_999")

        assert response.status_code == 404
