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
            reformulate_prompt_uri="prompts:/reformulate@prod",
            n_results=3,
            llm_model="gpt-4",
        ),
        store=MagicMock(),
        completions_client=MagicMock(),
        dataset=_TEST_DATASET,
        sessions=InMemorySessionStore(),
    )
    app.state.app_state = state
    return app, state


_POLICY_RECORD = {
    "page_content": "Return policy content",
    "metadata": {
        "document_id": "policy_001",
        "type": "policy",
        "category": "returns",
        "date": "2024-01-01",
        "product_category": "all",
        "chunk_index": 0,
        "total_chunks": 1,
    },
}

_FAQ_RECORD = {
    "page_content": "FAQ content",
    "metadata": {
        "document_id": "faq_001",
        "type": "faq",
        "category": "general",
        "date": "2024-01-01",
        "product_category": "all",
        "chunk_index": 0,
        "total_chunks": 1,
    },
}

_TICKET_RECORD = {
    "page_content": "Ticket content",
    "metadata": {
        "document_id": "ticket_001",
        "type": "support_ticket",
        "category": "delivery_issue",
        "date": "2024-01-01",
        "product_category": "laptops",
        "chunk_index": 0,
        "total_chunks": 1,
    },
}


@patch("omnia_support_app.routes.run_rag_pipeline")
class TestChatEndpoint:
    def test_returns_answer(self, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "Our return policy is...",
            "retrieved_context": "policy content",
            "retrieved_records": [_POLICY_RECORD],
        }
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

    def test_passes_reformulate_prompt_uri_to_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "answer",
            "retrieved_context": "context",
            "retrieved_records": [_POLICY_RECORD],
        }
        app, _ = _create_test_app()
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        assert mock_pipeline.call_args.kwargs["reformulate_prompt_uri"] == "prompts:/reformulate@prod"

    def test_second_message_passes_history(self, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "answer",
            "retrieved_context": "context",
            "retrieved_records": [_POLICY_RECORD],
        }
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
        mock_pipeline.return_value = {
            "answer": "the answer",
            "retrieved_context": "the context",
            "retrieved_records": [_POLICY_RECORD],
        }
        app, state = _create_test_app()
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "my question"},
        )

        history = state.sessions.get_history("s1")
        assert len(history) == 3
        assert history[0] == {"role": "developer", "content": "the context"}
        assert history[1] == {"role": "user", "content": "my question"}
        assert history[2] == {"role": "assistant", "content": "the answer"}

    def test_returns_deduplicated_sources(self, mock_pipeline):
        duplicate_record = {
            "page_content": "Another chunk from same policy",
            "metadata": {
                "document_id": "policy_001",
                "type": "policy",
                "category": "returns",
                "date": "2024-01-01",
                "product_category": "all",
                "chunk_index": 1,
                "total_chunks": 1,
            },
        }
        mock_pipeline.return_value = {
            "answer": "answer",
            "retrieved_context": "context",
            "retrieved_records": [_POLICY_RECORD, duplicate_record, _FAQ_RECORD],
        }
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
        mock_pipeline.return_value = {
            "answer": "answer",
            "retrieved_context": "context",
            "retrieved_records": [_TICKET_RECORD, _POLICY_RECORD],
        }
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
        mock_pipeline.return_value = {
            "answer": "a",
            "retrieved_context": "c",
            "retrieved_records": [_POLICY_RECORD],
        }
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
