"""API route handlers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel

from omnia_support_rag import run_rag_pipeline

from .dependencies import AppState  # noqa: TC001 — needed at runtime for FastAPI Depends

router = APIRouter(prefix="/api")


def _get_state(request: Request) -> AppState:
    return request.app.state.app_state


class ChatRequest(BaseModel):
    """Incoming chat message."""

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response returned to the client."""

    session_id: str
    answer: str


@router.post("/chat")
def chat(
    body: ChatRequest,
    state: Annotated[AppState, Depends(_get_state)],
) -> ChatResponse:
    """Handle a chat message and return the assistant's response."""
    history = state.sessions.get_history(body.session_id)
    cfg = state.config

    result = run_rag_pipeline(
        body.message,
        completions_client=state.completions_client,
        store=state.store,
        prompt_uri=cfg.prompt_uri,
        n_results=cfg.n_results,
        model=cfg.llm_model,
        history=history,
        reformulate_prompt_uri=cfg.reformulate_prompt_uri,
    )

    state.sessions.add_messages(
        body.session_id,
        [
            {"role": "developer", "content": result["retrieved_context"]},
            {"role": "user", "content": body.message},
            {"role": "assistant", "content": result["answer"]},
        ],
    )

    return ChatResponse(session_id=body.session_id, answer=result["answer"])


@router.delete("/chat/{session_id}", status_code=204)
def clear_session(
    session_id: str,
    state: Annotated[AppState, Depends(_get_state)],
) -> Response:
    """Clear a session's conversation history."""
    state.sessions.clear(session_id)
    return Response(status_code=204)
