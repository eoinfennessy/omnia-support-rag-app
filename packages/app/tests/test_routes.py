"""Tests for API routes."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omnia_support_agents import ActionResult, AgentResult, InformationResult, Urgency
from omnia_support_agents.models import ToolCall, ToolCallResultOk
from omnia_support_app.dependencies import AppState
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
        dataset=_TEST_DATASET,
        orchestrator=MagicMock(handle=AsyncMock()),
        mcp_server=MagicMock(),
        sessions=InMemorySessionStore(),
    )
    app.state.app_state = state
    return app, state


def _make_record(
    *,
    document_id: str,
    page_content: str,
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
        page_content=page_content,
        metadata=metadata,
        score=0.5,
    )


_POLICY_RECORD = _make_record(
    document_id="policy_001",
    page_content="Return policy content",
    doc_type="policy",
    category="returns",
)

_FAQ_RECORD = _make_record(
    document_id="faq_001",
    page_content="FAQ content",
    doc_type="faq",
    category="general",
)

_TICKET_RECORD = _make_record(
    document_id="ticket_001",
    page_content="Ticket content",
    doc_type="support_ticket",
    category="delivery_issue",
    product_category="laptops",
)


def _info_result(
    query: str = "question",
    answer: str = "answer",
    context: list[ScoredChunk] | None = None,
) -> InformationResult:
    return InformationResult(query=query, answer=answer, context=context or [])


def _action_result(
    query: str = "question",
    answer: str = "action answer",
    tool_calls: list[ToolCall] | None = None,
) -> ActionResult:
    return ActionResult(query=query, answer=answer, tool_calls=tool_calls or [])


def _agent_result(
    answer: str = "answer",
    urgency: Urgency = Urgency.LOW,
    sub_query_results: list[InformationResult | ActionResult] | None = None,
) -> AgentResult:
    return AgentResult(
        answer=answer,
        urgency=urgency,
        sub_query_results=sub_query_results or [],
    )


class TestChatEndpoint:
    def test_returns_answer(self):
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = _agent_result(  # ty: ignore[unresolved-attribute]
            answer="Our return policy is...",
            sub_query_results=[_info_result(answer="Our return policy is...", context=[_POLICY_RECORD])],
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "What is your return policy?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Our return policy is..."
        assert data["session_id"] == "s1"

    def test_returns_urgency(self):
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = _agent_result(urgency=Urgency.HIGH)  # ty: ignore[unresolved-attribute]
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        assert response.json()["urgency"] == "high"

    def test_returns_sub_query_results(self):
        app, state = _create_test_app()
        info = _info_result(query="info q", answer="info a", context=[_POLICY_RECORD])
        action = _action_result(
            query="action q",
            answer="action a",
            tool_calls=[ToolCall(tool_name="get_order", arguments={"id": "ORD-1"}, result=ToolCallResultOk(data={}))],
        )
        state.orchestrator.handle.return_value = _agent_result(  # ty: ignore[unresolved-attribute]
            answer="combined",
            sub_query_results=[info, action],
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        data = response.json()
        assert len(data["sub_query_results"]) == 2

    def test_second_message_passes_history(self):
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = _agent_result()  # ty: ignore[unresolved-attribute]
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "first question"},
        )
        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "follow-up"},
        )

        second_call = state.orchestrator.handle.call_args_list[1]  # ty: ignore[unresolved-attribute]
        history = second_call.args[1]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_stores_user_assistant_pair(self):
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = _agent_result(answer="the answer")  # ty: ignore[unresolved-attribute]
        client = TestClient(app)

        client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "my question"},
        )

        history = state.sessions.get_history("s1")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "my question"}
        assert history[1] == {"role": "assistant", "content": "the answer"}

    def test_returns_deduplicated_sources(self):
        duplicate_record = _make_record(
            document_id="policy_001",
            page_content="Another chunk from same policy",
            doc_type="policy",
            category="returns",
            chunk_index=1,
        )
        state_orchestrator_result = _agent_result(
            sub_query_results=[
                _info_result(context=[_POLICY_RECORD, duplicate_record, _FAQ_RECORD]),
            ],
        )
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = state_orchestrator_result  # ty: ignore[unresolved-attribute]
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        sources = response.json()["sources"]
        assert len(sources) == 2
        assert sources[0] == {"url": "/policies/policy_001", "title": "Returns", "type": "policy"}
        assert sources[1] == {"url": "/faqs/faq_001", "title": "faq_001 - General", "type": "faq"}

    def test_excludes_support_tickets_from_sources(self):
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = _agent_result(  # ty: ignore[unresolved-attribute]
            sub_query_results=[_info_result(context=[_TICKET_RECORD, _POLICY_RECORD])],
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "question"},
        )

        sources = response.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["type"] == "policy"

    def test_action_only_response_has_no_sources(self):
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = _agent_result(  # ty: ignore[unresolved-attribute]
            sub_query_results=[
                _action_result(
                    tool_calls=[
                        ToolCall(
                            tool_name="get_order",
                            arguments={"order_id": "ORD-1"},
                            result=ToolCallResultOk(data={"status": "shipped"}),
                        ),
                    ],
                ),
            ],
        )
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"session_id": "s1", "message": "Where is my order?"},
        )

        data = response.json()
        assert data["sources"] == []


class TestClearSessionEndpoint:
    def test_clears_session(self):
        app, state = _create_test_app()
        state.orchestrator.handle.return_value = _agent_result()  # ty: ignore[unresolved-attribute]
        client = TestClient(app)

        client.post("/api/chat", json={"session_id": "s1", "message": "hi"})
        assert len(state.sessions.get_history("s1")) == 2

        response = client.delete("/api/chat/s1")

        assert response.status_code == 204
        assert state.sessions.get_history("s1") == []

    def test_clear_nonexistent_session_returns_204(self):
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
