"""FastAPI application for Omnia Support chatbot."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .dependencies import create_app_state
from .routes import router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[4] / "frontend" / "build"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared application state at startup."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    app.state.app_state = create_app_state()
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="Omnia Support", lifespan=lifespan)
    app.include_router(router)

    static_dir = Path(os.environ.get("STATIC_DIR", str(_DEFAULT_STATIC_DIR)))
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    uvicorn.run(
        "omnia_support_app.main:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104
        port=8080,
    )
