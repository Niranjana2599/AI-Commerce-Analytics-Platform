"""Shared configuration for the Streamlit dashboard."""

import os


# FASTAPI_BASE_URL works locally/Docker. Render provides FASTAPI_HOSTPORT for private networking.
_internal_api = os.getenv("FASTAPI_HOSTPORT")
API_BASE_URL = os.getenv("FASTAPI_BASE_URL") or (f"http://{_internal_api}/api/v1" if _internal_api else "http://127.0.0.1:8000/api/v1")
PAGE_TITLE = "AI Commerce Analytics"
