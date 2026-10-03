"""Natural-language business entry point for the existing agent API."""

import streamlit as st

from utils.agent_entry import render_agent_entry
from utils.ui import page_hero, setup_page


setup_page("Agentic Analyst")
page_hero(
    "One supervisor · existing commerce tools",
    "Ask the Business Analyst",
    "Ask a business question. The supervisor selects the relevant existing customer, analytics, forecasting, recommendation, or knowledge tools.",
)

render_agent_entry("agentic_analyst", show_optional_context=True)

with st.expander("Example questions"):
    st.markdown(
        "- Who are our highest-value customers?\n"
        "- Which customers are likely to churn?\n"
        "- Which products should we recommend?\n"
        "- Forecast demand for next week.\n"
        "- What are our primary KPIs?\n"
        "- A festival is coming. How can we increase revenue?\n"
        "- How do we compare with competitors?"
    )

with st.expander("How this differs from the RAG Chatbot"):
    st.markdown(
        "The Agentic Analyst routes a question across structured tools and may use knowledge retrieval when relevant. "
        "The RAG Chatbot remains a direct workspace for questions specifically about indexed documents. "
        "Customer, Analytics, and Knowledge are capability groupings; this application runs one supervisor, not three autonomous agents."
    )
