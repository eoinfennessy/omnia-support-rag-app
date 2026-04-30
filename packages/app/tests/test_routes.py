"""Tests for API routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omnia_support_app.dependencies import AppConfig, AppState
from omnia_support_app.routes import router
from omnia_support_app.session import InMemorySessionStore


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
        sessions=InMemorySessionStore(),
    )
    app.state.app_state = state
    return app, state


@patch("omnia_support_app.routes.run_rag_pipeline")
class TestChatEndpoint:
    def test_returns_answer(self, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "Our return policy is...",
            "retrieved_context": "policy content",
            "retrieved_records": [],
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
            "retrieved_records": [],
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
            "retrieved_records": [],
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
            "retrieved_records": [],
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


@patch("omnia_support_app.routes.run_rag_pipeline")
class TestClearSessionEndpoint:
    def test_clears_session(self, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "a",
            "retrieved_context": "c",
            "retrieved_records": [],
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
