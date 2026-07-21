"""Shared configuration for the Streamlit dashboard."""

import os


# Override this value with the FASTAPI_BASE_URL environment variable in Docker or production.
API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000/api/v1")
PAGE_TITLE = "AI Commerce Analytics"
