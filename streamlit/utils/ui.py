"""Reusable Streamlit presentation helpers."""

import streamlit as st

from config import API_BASE_URL, PAGE_TITLE


def setup_page(title: str) -> None:
    """Apply consistent browser metadata and sidebar content."""
    st.set_page_config(page_title=f"{title} | {PAGE_TITLE}", page_icon="🛒", layout="wide")
    with st.sidebar:
        st.title("🛒 AI Commerce")
        st.caption("Analytics & AI workspace")
        st.divider()
        st.success("Backend connected" if _api_url_is_set() else "Configure FASTAPI_BASE_URL")
        st.caption(f"API: {API_BASE_URL}")
        st.divider()
        st.caption("Choose a module from the navigation above.")


def _api_url_is_set() -> bool:
    return bool(API_BASE_URL)


def show_api_error(error: str | None) -> None:
    """Display an API failure consistently across all pages."""
    if error:
        st.error(error)
        st.info("Check that FastAPI is running and that the endpoint has its data/model artifact.")
