"""Operational dashboard for aggregate, privacy-safe RAG telemetry."""

from datetime import datetime
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("RAG Operations")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
SERVICE_LINKS = {
    "Prometheus": "http://localhost:9090",
    "Grafana": "http://localhost:3000",
    "MLflow": "http://localhost:5000",
    "LangSmith": "https://smith.langchain.com",
}
SCORE_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_relevance": "Context relevance",
}
LATENCY_LABELS = {
    "retrieval": "Retrieval",
    "prompt": "Prompt / generation",
    "total": "Total execution",
}


def style_figure(figure: go.Figure, *, height: int = 370) -> go.Figure:
    """Apply consistent, accessible chart styling."""
    figure.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=58, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        title_font=dict(size=17, color="#f8fafc"),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
        legend_title_text="",
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    figure.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    return figure


def metric_value(mapping: dict, key: str) -> float:
    try:
        return float(mapping.get(key, 0))
    except (TypeError, ValueError):
        return 0.0


page_hero(
    "LLMOps observability",
    "RAG operations dashboard",
    "Monitor privacy-safe RAG quality, latency, and artifact-version usage without storing questions or generated answers here.",
)

refresh_column, snapshot_column, timestamp_column = st.columns([1, 1, 3])
with refresh_column:
    if st.button("Refresh metrics", type="primary", width="stretch"):
        st.rerun()

with st.spinner("Loading RAG operational metrics..."):
    metrics, error = api_request("GET", "/rag/metrics")
show_api_error(error)

if metrics:
    with snapshot_column:
        st.download_button(
            "Download snapshot",
            data=json.dumps(metrics, indent=2).encode("utf-8"),
            file_name=f"rag_metrics_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json",
            width="stretch",
        )
    with timestamp_column:
        st.caption(f"Last refreshed in this browser: {datetime.now():%d %b %Y, %H:%M:%S}")

    query_count = int(metrics.get("query_count", 0) or 0)
    averages = metrics.get("averages") or {}
    latency = metrics.get("latency_ms") or {}
    prompt_versions = metrics.get("prompt_versions") or {}
    knowledge_versions = metrics.get("knowledge_base_versions") or {}

    score_values = [metric_value(averages, key) for key in SCORE_LABELS]
    overall_quality = sum(score_values) / len(score_values) if score_values else 0.0

    section_heading("Operational summary", "Aggregate measurements from logged RAG events.")
    kpis = st.columns(6)
    kpis[0].metric("Logged queries", f"{query_count:,}")
    kpis[1].metric("Overall quality", f"{overall_quality:.0%}" if averages else "No data")
    kpis[2].metric("Mean total latency", f"{metric_value(latency, 'total_mean'):,.1f} ms")
    kpis[3].metric("P95 total latency", f"{metric_value(latency, 'total_p95'):,.1f} ms")
    kpis[4].metric("Prompt versions", len(prompt_versions))
    kpis[5].metric("Knowledge versions", len(knowledge_versions))

    if query_count == 0:
        st.info("No RAG events are available yet. Submit a question on the RAG Chatbot page, then refresh this dashboard.")
    else:
        if overall_quality >= 0.8:
            st.success("Aggregate RAG quality signals are currently strong. Continue reviewing individual answers and sources.")
        elif overall_quality >= 0.6:
            st.warning("Aggregate RAG quality is moderate. Review retrieval relevance and answer grounding.")
        else:
            st.error("Aggregate RAG quality signals are low. Inspect retrieval data, prompts, and evaluation design.")

    if averages:
        section_heading("Quality signals", "Automated evaluation averages across logged RAG requests.")
        gauge_column, bar_column = st.columns([1, 1.6])
        with gauge_column:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_quality * 100,
                    number={"suffix": "%", "font": {"color": "#f8fafc"}},
                    title={"text": "Overall evaluation average", "font": {"color": "#cbd5e1"}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": "#6366f1"},
                        "bgcolor": "rgba(15,23,42,.25)",
                        "steps": [
                            {"range": [0, 60], "color": "rgba(239,68,68,.22)"},
                            {"range": [60, 80], "color": "rgba(245,158,11,.22)"},
                            {"range": [80, 100], "color": "rgba(16,185,129,.22)"},
                        ],
                    },
                )
            )
            st.plotly_chart(style_figure(gauge, height=340), width="stretch", config=PLOT_CONFIG)
        with bar_column:
            evaluation_frame = pd.DataFrame(
                {
                    "Metric": [SCORE_LABELS[key] for key in SCORE_LABELS],
                    "Score": [metric_value(averages, key) for key in SCORE_LABELS],
                }
            )
            evaluation_chart = px.bar(
                evaluation_frame,
                x="Score",
                y="Metric",
                orientation="h",
                range_x=[0, 1],
                text_auto=".0%",
                title="Average evaluation scores",
                color="Score",
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            )
            evaluation_chart.update_layout(coloraxis_showscale=False)
            evaluation_chart.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_figure(evaluation_chart, height=340), width="stretch", config=PLOT_CONFIG)

    if latency:
        section_heading("Latency profile", "Compare mean and 95th-percentile execution times by RAG stage.")
        latency_rows = []
        for prefix, label in LATENCY_LABELS.items():
            latency_rows.extend(
                [
                    {"Stage": label, "Statistic": "Mean", "Milliseconds": metric_value(latency, f"{prefix}_mean")},
                    {"Stage": label, "Statistic": "P95", "Milliseconds": metric_value(latency, f"{prefix}_p95")},
                ]
            )
        latency_frame = pd.DataFrame(latency_rows)
        latency_chart = px.bar(
            latency_frame,
            x="Stage",
            y="Milliseconds",
            color="Statistic",
            barmode="group",
            title="RAG latency by stage",
            color_discrete_map={"Mean": "#0ea5e9", "P95": "#a855f7"},
            text_auto=".1f",
        )
        st.plotly_chart(style_figure(latency_chart, height=400), width="stretch", config=PLOT_CONFIG)

        bottleneck = max(
            ((label, metric_value(latency, f"{prefix}_mean")) for prefix, label in LATENCY_LABELS.items()),
            key=lambda item: item[1],
        )
        st.info(f"Highest mean latency: **{bottleneck[0]}** at **{bottleneck[1]:,.1f} ms**.")

    section_heading("Artifact versions", "Track which prompt and knowledge-base versions produced logged responses.")
    prompt_column, knowledge_column = st.columns(2)
    version_sets = [
        (prompt_column, prompt_versions, "Prompt version usage", "Prompt version"),
        (knowledge_column, knowledge_versions, "Knowledge-base version usage", "Knowledge-base version"),
    ]
    for column, versions, title, label in version_sets:
        with column:
            if not versions:
                st.info(f"No {label.lower()} data is available.")
            else:
                version_frame = pd.DataFrame(versions.items(), columns=[label, "Queries"])
                version_chart = px.pie(
                    version_frame,
                    names=label,
                    values="Queries",
                    hole=0.56,
                    title=title,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                version_chart.update_traces(textinfo="percent+label")
                st.plotly_chart(style_figure(version_chart, height=350), width="stretch", config=PLOT_CONFIG)
                st.dataframe(version_frame.sort_values("Queries", ascending=False), hide_index=True, width="stretch")
else:
    st.warning("RAG operational metrics could not be loaded.")

section_heading("Observability services", "Use each tool for the telemetry it is designed to manage.")
service_columns = st.columns(4)
service_descriptions = {
    "Prometheus": "Raw application and infrastructure metrics",
    "Grafana": "Provisioned dashboards and operational trends",
    "MLflow": "Training experiments, parameters, metrics, and artifacts",
    "LangSmith": "Optional RAG traces, debugging, and evaluations",
}
for column, (name, url) in zip(service_columns, SERVICE_LINKS.items()):
    with column:
        st.markdown(
            f'<div class="module-card"><h3>{name}</h3><p>{service_descriptions[name]}</p></div>',
            unsafe_allow_html=True,
        )
        st.link_button(f"Open {name}", url, width="stretch")

with st.expander("Data coverage and monitoring boundaries"):
    st.markdown(
        """
        - This page uses the existing `/rag/metrics` endpoint and displays aggregate, privacy-safe event data.
        - Query text and generated answers are not included in this operational response.
        - The endpoint does not return a timestamped request series or error count, so this page does not create request-rate or error-rate charts.
        - Use Grafana and Prometheus for API request rate, response time, errors, CPU, memory, and container monitoring.
        - MLflow tracks training experiments; LangSmith is separate and traces the optional RAG execution path when configured.
        """
    )
