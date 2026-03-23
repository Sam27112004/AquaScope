"""
FastAPI application factory for the Aquascope web backend.

Start the server::

    uvicorn api.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aquascope.api.routes.inference_routes import router as inference_router
from aquascope.api.routes.dataset_routes import router as dataset_router
from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    logger.info("Aquascope API starting up...")
    yield
    logger.info("Aquascope API shutting down.")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Aquascope API",
        description="AI-based underwater inspection analysis service.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten to specific origins in production.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(inference_router, prefix="/inference", tags=["Inference"])
    application.include_router(dataset_router, prefix="/datasets", tags=["Datasets"])

    @application.get("/health", tags=["Health"])
    async def health_check() -> dict:
        return {"status": "ok", "service": "aquascope"}

    return application


app = create_app()
