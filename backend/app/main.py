"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import settings


def create_app() -> FastAPI:
    """Create the application without loading data or ML artifacts at startup."""
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="APIs for the completed AI-commerce analytics workflows.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()
