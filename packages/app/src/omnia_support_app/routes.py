"""API route handlers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from omnia_support_agents import ActionResult, InformationResult
from omnia_support_models import Document, DocumentType

from .dependencies import AppState  # noqa: TC001 — needed at runtime for FastAPI Depends

router = APIRouter(prefix="/api")


def _to_title_case(snake: str) -> str:
    return " ".join(word.capitalize() for word in snake.split("_"))


def _get_state(request: Request) -> AppState:
    return request.app.state.app_state


class ChatRequest(BaseModel):
    """Incoming chat message."""

    session_id: str
    message: str


class Source(BaseModel):
    """A source document referenced by the assistant's answer."""

    url: str
    title: str
    type: str


class ChatResponse(BaseModel):
    """Chat response returned to the client."""

    session_id: str
    answer: str
    sources: list[Source]
    urgency: str | None = None
    sub_query_results: list[InformationResult | ActionResult] = Field(default_factory=list)


@router.post("/chat")
async def chat(
    body: ChatRequest,
    state: Annotated[AppState, Depends(_get_state)],
) -> ChatResponse:
    """Handle a chat message and return the assistant's response."""
    history = state.sessions.get_history(body.session_id)

    result = await state.orchestrator.handle(body.message, history)

    state.sessions.add_messages(
        body.session_id,
        [
            {"role": "user", "content": body.message},
            {"role": "assistant", "content": result.answer},
        ],
    )

    seen: set[str] = set()
    sources: list[Source] = []
    for sub_result in result.sub_query_results:
        if not isinstance(sub_result, InformationResult):
            continue
        for record in sub_result.context:
            doc_id = record.metadata["document_id"]
            doc_type = record.metadata["type"]
            if doc_id not in seen and doc_type in ("policy", "faq"):
                seen.add(doc_id)
                category = record.metadata["category"]
                title = _to_title_case(category)
                if doc_type == "faq":
                    title = f"{doc_id} - {title}"
                sources.append(
                    Source(
                        url=f"/{doc_type}s/{doc_id}" if doc_type == "faq" else f"/policies/{doc_id}",
                        title=title,
                        type=doc_type,
                    )
                )

    return ChatResponse(
        session_id=body.session_id,
        answer=result.answer,
        sources=sources,
        urgency=result.urgency,
        sub_query_results=result.sub_query_results,
    )


@router.delete("/chat/{session_id}", status_code=204)
def clear_session(
    session_id: str,
    state: Annotated[AppState, Depends(_get_state)],
) -> Response:
    """Clear a session's conversation history."""
    state.sessions.clear(session_id)
    return Response(status_code=204)


def _documents_by_type(state: AppState, doc_type: DocumentType) -> list[Document]:
    return [doc for doc in state.dataset.documents if doc.type == doc_type]


def _find_document(state: AppState, doc_type: DocumentType, document_id: str) -> Document:
    for doc in state.dataset.documents:
        if doc.type == doc_type and doc.document_id == document_id:
            return doc
    raise HTTPException(status_code=404, detail="Document not found")


@router.get("/policies")
def list_policies(
    state: Annotated[AppState, Depends(_get_state)],
) -> list[Document]:
    """Return all policy documents."""
    return _documents_by_type(state, DocumentType.POLICY)


@router.get("/policies/{document_id}")
def get_policy(
    document_id: str,
    state: Annotated[AppState, Depends(_get_state)],
) -> Document:
    """Return a single policy document by ID."""
    return _find_document(state, DocumentType.POLICY, document_id)


@router.get("/faqs")
def list_faqs(
    state: Annotated[AppState, Depends(_get_state)],
) -> list[Document]:
    """Return all FAQ documents."""
    return _documents_by_type(state, DocumentType.FAQ)


@router.get("/faqs/{document_id}")
def get_faq(
    document_id: str,
    state: Annotated[AppState, Depends(_get_state)],
) -> Document:
    """Return a single FAQ document by ID."""
    return _find_document(state, DocumentType.FAQ, document_id)
