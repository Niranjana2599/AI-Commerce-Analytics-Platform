"""Reusable Streamlit presentation helpers."""

import streamlit as st

from config import API_BASE_URL, PAGE_TITLE


def setup_page(title: str) -> None:
    """Apply consistent browser metadata, styling, and sidebar content."""
    st.set_page_config(
        page_title=f"{title} | {PAGE_TITLE}",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    with st.sidebar:
        st.markdown("### 🛒 AI Commerce")
        st.caption("Analytics & AI workspace")
        st.divider()
        if _api_url_is_set():
            st.markdown('<div class="sidebar-status"><span></span> API configured</div>', unsafe_allow_html=True)
        else:
            st.warning("Configure FASTAPI_BASE_URL")
        with st.expander("Connection details"):
            st.code(API_BASE_URL, language=None)
        st.divider()
        st.caption("Select a module from the navigation menu.")


def _apply_theme() -> None:
    """Inject lightweight, responsive styling shared by every page."""
    st.markdown(
        """
        <style>
        :root {
            --ac-primary: #6366f1;
            --ac-success: #10b981;
            --ac-border: rgba(148, 163, 184, 0.22);
            --ac-surface: rgba(30, 41, 59, 0.38);
        }
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1480px;}
        h1, h2, h3 {letter-spacing: -0.025em;}
        [data-testid="stMetric"] {
            background: var(--ac-surface); border: 1px solid var(--ac-border);
            border-radius: 16px; padding: 1rem 1.1rem; min-height: 118px;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.08);
        }
        [data-testid="stMetricLabel"] {font-weight: 600; color: #a5b4fc;}
        [data-testid="stMetricValue"] {font-size: clamp(1.45rem, 2.4vw, 2rem);}
        [data-testid="stForm"] {border: 1px solid var(--ac-border); border-radius: 16px; padding: 1.2rem;}
        .hero {
            padding: clamp(1.4rem, 4vw, 2.4rem); border: 1px solid var(--ac-border);
            border-radius: 22px; background: linear-gradient(135deg, rgba(99,102,241,.18), rgba(14,165,233,.08));
            margin-bottom: 1.4rem;
        }
        .eyebrow {color: #a5b4fc; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;}
        .hero h1 {margin: .35rem 0 .55rem; font-size: clamp(2rem, 4vw, 3.35rem);}
        .hero p {max-width: 760px; color: #cbd5e1; font-size: 1.02rem; margin: 0;}
        .section-heading {margin: 1.5rem 0 .8rem;}
        .section-heading h2 {font-size: 1.35rem; margin: 0;}
        .section-heading p {color: #94a3b8; margin: .25rem 0 0;}
        .sidebar-status {display:flex; align-items:center; gap:.55rem; font-size:.88rem; color:#cbd5e1;}
        .sidebar-status span {width:.55rem; height:.55rem; border-radius:50%; background:var(--ac-success); box-shadow:0 0 0 4px rgba(16,185,129,.12);}
        .module-card {
            border: 1px solid var(--ac-border); border-radius: 14px; padding: 1rem;
            min-height: 126px; background: rgba(15,23,42,.2); margin-bottom: .7rem;
        }
        .module-card h3 {font-size: 1rem; margin: .35rem 0;}
        .module-card p {color:#94a3b8; font-size:.88rem; margin:0;}
        @media (max-width: 640px) {
            .block-container {padding: 1rem .8rem 2rem;}
            [data-testid="stMetric"] {min-height: 100px;}
            .hero {border-radius: 16px;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_hero(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent page introduction."""
    st.markdown(
        f'<section class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{description}</p></section>',
        unsafe_allow_html=True,
    )


def section_heading(title: str, description: str) -> None:
    """Render a compact section heading and supporting copy."""
    st.markdown(
        f'<div class="section-heading"><h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def _api_url_is_set() -> bool:
    return bool(API_BASE_URL)


def show_api_error(error: str | None) -> None:
    """Display an API failure consistently across all pages."""
    if error:
        st.error(error)
        st.info("Check that FastAPI is running and that the endpoint has its data/model artifact.")
