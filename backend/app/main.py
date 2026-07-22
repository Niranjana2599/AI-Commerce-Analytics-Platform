"""FastAPI application entry point."""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.monitoring.metrics import ACTIVE_REQUESTS, HTTP_ERRORS, HTTP_REQUEST_DURATION, HTTP_REQUESTS


def _endpoint_label(request: Request) -> str:
    """Use route templates rather than user-controlled URLs as metric labels."""
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


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

    @app.middleware("http")
    async def collect_http_metrics(request: Request, call_next):
        """Record API requests while excluding Prometheus scrapes from API traffic metrics."""
        if request.url.path.startswith("/metrics"):
            return await call_next(request)

        started = time.perf_counter()
        ACTIVE_REQUESTS.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            endpoint = _endpoint_label(request)
            method = request.method
            HTTP_REQUESTS.labels(method, endpoint, str(status_code)).inc()
            HTTP_REQUEST_DURATION.labels(method, endpoint).observe(duration)
            if status_code >= 500:
                HTTP_ERRORS.labels(method, endpoint).inc()
            ACTIVE_REQUESTS.dec()

    app.include_router(router, prefix=settings.api_prefix)
    # Prometheus scrapes this endpoint; it also includes Python process metrics on Linux.
    app.mount("/metrics", make_asgi_app())
    return app


app = create_app()
