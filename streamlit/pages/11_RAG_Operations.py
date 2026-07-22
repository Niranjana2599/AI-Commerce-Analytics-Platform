"""Operational dashboard for RAG quality, latency, and version monitoring."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("RAG Operations")
st.title("RAG Operations Dashboard")
st.caption("Privacy-safe monitoring: query text and generated answers are not stored here.")

if st.button("Refresh metrics"):
    st.rerun()

with st.spinner("Loading RAG operational metrics..."):
    metrics, error = api_request("GET", "/rag/metrics")
show_api_error(error)

if metrics:
    query, retrieval, prompt = st.columns(3)
    query.metric("Logged queries", metrics["query_count"])
    retrieval.metric("Mean retrieval latency", f"{metrics['latency_ms'].get('retrieval_mean', 0):.1f} ms")
    prompt.metric("Mean prompt latency", f"{metrics['latency_ms'].get('prompt_mean', 0):.1f} ms")

    st.subheader("RAG evaluation signals")
    evaluations = pd.DataFrame({"Metric": list(metrics["averages"].keys()), "Score": list(metrics["averages"].values())})
    if evaluations.empty:
        st.info("No RAG queries have been logged yet. Use the RAG Chatbot page to create the first event.")
    else:
        st.plotly_chart(px.bar(evaluations, x="Metric", y="Score", range_y=[0, 1], title="Average evaluation scores"), use_container_width=True)

    st.subheader("Latency percentiles")
    latency = pd.DataFrame({"Metric": list(metrics["latency_ms"].keys()), "Milliseconds": list(metrics["latency_ms"].values())})
    st.plotly_chart(px.bar(latency, x="Metric", y="Milliseconds", title="Retrieval and prompt latency"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Prompt versions")
        st.dataframe(pd.DataFrame(metrics["prompt_versions"].items(), columns=["Version", "Queries"]), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Knowledge-base versions")
        st.dataframe(pd.DataFrame(metrics["knowledge_base_versions"].items(), columns=["Version", "Queries"]), hide_index=True, use_container_width=True)
