"""Unified operational monitoring backed by Prometheus and service health APIs."""

from datetime import datetime, timedelta, timezone
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from config import API_BASE_URL
from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page


setup_page("Monitoring")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
IN_DOCKER = "backend:8000" in API_BASE_URL
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090" if IN_DOCKER else "http://127.0.0.1:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000" if IN_DOCKER else "http://127.0.0.1:3000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000" if IN_DOCKER else "http://127.0.0.1:5000")


def style_figure(figure, *, height: int = 350):
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


@st.cache_data(ttl=15, show_spinner=False)
def service_health() -> dict[str, dict[str, str | bool]]:
    """Probe only non-sensitive health endpoints."""
    services = {}
    health, error = api_request("GET", "/health", timeout=5)
    services["FastAPI"] = {
        "healthy": bool(health and not error),
        "detail": health.get("environment", "Healthy") if health else (error or "Unavailable"),
    }
    probes = {
        "Prometheus": (f"{PROMETHEUS_URL}/-/healthy", "Prometheus metrics store"),
        "Grafana": (f"{GRAFANA_URL}/api/health", "Grafana dashboard service"),
        "MLflow": (f"{MLFLOW_URL}/health", "MLflow tracking server"),
    }
    for name, (url, description) in probes.items():
        try:
            response = requests.get(url, timeout=4)
            services[name] = {"healthy": response.ok, "detail": description if response.ok else f"HTTP {response.status_code}"}
        except requests.RequestException:
            services[name] = {"healthy": False, "detail": "Unavailable from Streamlit"}
    return services


@st.cache_data(ttl=15, show_spinner=False)
def prometheus_instant(query: str) -> tuple[float | None, str | None]:
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=8)
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])
        if not results:
            return None, None
        return float(results[0]["value"][1]), None
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        return None, str(exc)


@st.cache_data(ttl=15, show_spinner=False)
def prometheus_range(query: str, hours: int) -> tuple[pd.DataFrame, str | None]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    params = {"query": query, "start": start.timestamp(), "end": end.timestamp(), "step": "60s"}
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params, timeout=10)
        response.raise_for_status()
        series = response.json().get("data", {}).get("result", [])
        rows = []
        for item in series:
            label = item.get("metric", {}).get("model_name") or item.get("metric", {}).get("endpoint") or "Total"
            for timestamp, value in item.get("values", []):
                rows.append({"Time": pd.to_datetime(timestamp, unit="s", utc=True), "Value": float(value), "Series": label})
        return pd.DataFrame(rows), None
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        return pd.DataFrame(columns=["Time", "Value", "Series"]), str(exc)


page_hero(
    "Platform observability",
    "System monitoring",
    "Track live service health, API traffic, prediction activity, latency, errors, and container resource usage from one workspace.",
)

control_column, timestamp_column = st.columns([1, 4])
with control_column:
    if st.button("Refresh now", type="primary", width="stretch"):
        service_health.clear()
        prometheus_instant.clear()
        prometheus_range.clear()
        st.rerun()
with timestamp_column:
    st.caption(f"Metrics refresh automatically from a 15-second cache · Viewed {datetime.now():%d %b %Y, %H:%M:%S}")

with st.spinner("Checking platform services..."):
    services = service_health()

section_heading("Service health", "Reachability from the running Streamlit process.")
service_columns = st.columns(5)
for column, (name, status) in zip(service_columns[:4], services.items()):
    with column:
        state = "Online" if status["healthy"] else "Unavailable"
        st.metric(name, state)
        st.caption(str(status["detail"]))
with service_columns[4]:
    st.metric("LangSmith", "External")
    st.caption("Optional RAG tracing; configured separately")

failed_services = [name for name, status in services.items() if not status["healthy"]]
if failed_services:
    st.warning("Unavailable from Streamlit: " + ", ".join(failed_services) + ". Check `docker compose ps` and service logs.")
else:
    st.success("FastAPI, Prometheus, Grafana, and MLflow are reachable.")

instant_queries = {
    "active_models": "sum(loaded_models)",
    "prediction_count": "sum(prediction_requests_total)",
    "active_requests": "sum(active_requests)",
    "api_requests": "sum(http_requests_total)",
    "cpu_percent": "sum(rate(container_cpu_usage_seconds_total{name=~\"ai-commerce-.*\"}[5m])) * 100",
    "memory_bytes": "sum(container_memory_working_set_bytes{name=~\"ai-commerce-.*\"})",
}
instant_values = {name: prometheus_instant(query)[0] for name, query in instant_queries.items()}

section_heading("Live platform KPIs", "Current cumulative activity and resource signals from Prometheus.")
kpis = st.columns(6)
kpis[0].metric("Active models", "—" if instant_values["active_models"] is None else f"{instant_values['active_models']:.0f}")
kpis[1].metric("Predictions", "—" if instant_values["prediction_count"] is None else f"{instant_values['prediction_count']:,.0f}")
kpis[2].metric("API requests", "—" if instant_values["api_requests"] is None else f"{instant_values['api_requests']:,.0f}")
kpis[3].metric("Active requests", "—" if instant_values["active_requests"] is None else f"{instant_values['active_requests']:.0f}")
kpis[4].metric("Container CPU", "—" if instant_values["cpu_percent"] is None else f"{instant_values['cpu_percent']:.1f}%")
kpis[5].metric(
    "Container memory",
    "—" if instant_values["memory_bytes"] is None else f"{instant_values['memory_bytes'] / (1024 ** 2):,.0f} MiB",
)

range_column, link_column = st.columns([1, 3])
with range_column:
    hours = st.selectbox("Chart window", options=[1, 3, 6, 12, 24], index=2, format_func=lambda value: f"Last {value} hour{'s' if value != 1 else ''}")
with link_column:
    st.caption("Charts use one-minute Prometheus query steps. For alerting and deeper drill-down, use the provisioned Grafana dashboards.")

chart_queries = {
    "API request rate": ("sum(rate(http_requests_total[5m]))", "Requests / second", "#0ea5e9"),
    "Prediction request rate": ("sum(rate(prediction_requests_total[5m]))", "Predictions / second", "#6366f1"),
    "P95 API response time": (
        "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
        "Seconds",
        "#a855f7",
    ),
    "API error rate": ("sum(rate(http_errors_total[5m]))", "Errors / second", "#ef4444"),
}

section_heading("Application traffic and reliability", "Real time-series data queried from Prometheus.")
chart_columns = st.columns(2)
for index, (title, (query, y_label, color)) in enumerate(chart_queries.items()):
    frame, query_error = prometheus_range(query, hours)
    with chart_columns[index % 2]:
        if query_error:
            st.error(f"{title} could not be loaded from Prometheus.")
        elif frame.empty:
            st.info(f"No samples are available yet for {title.lower()}.")
        else:
            figure = px.line(frame, x="Time", y="Value", color="Series", title=title, labels={"Value": y_label})
            if frame["Series"].nunique() == 1:
                figure.update_traces(line=dict(color=color, width=3), fill="tozeroy", fillcolor=f"{color}22")
                figure.update_layout(showlegend=False)
            st.plotly_chart(style_figure(figure), width="stretch", config=PLOT_CONFIG)

section_heading("Monitoring tools", "Open the dedicated interface for deeper investigation.")
tool_columns = st.columns(4)
tools = [
    ("Prometheus", "Build PromQL queries and inspect scrape targets.", "http://localhost:9090"),
    ("Grafana", "View dashboards, alerts, and infrastructure trends.", "http://localhost:3000"),
    ("MLflow", "Inspect experiments, metrics, parameters, and artifacts.", "http://localhost:5000"),
    ("LangSmith", "Inspect optional RAG traces and evaluations.", "https://smith.langchain.com"),
]
for column, (name, description, url) in zip(tool_columns, tools):
    with column:
        st.markdown(f'<div class="module-card"><h3>{name}</h3><p>{description}</p></div>', unsafe_allow_html=True)
        st.link_button(f"Open {name}", url, width="stretch")

with st.expander("Metric definitions and limitations"):
    st.markdown(
        """
        - API and prediction totals are process-lifetime Prometheus counters and reset when the backend container restarts.
        - Active models counts artifacts loaded in the current FastAPI process; models load when their endpoints are used.
        - CPU and memory depend on cAdvisor being successfully scraped by Prometheus.
        - LangSmith is a separate hosted RAG tracing service and is not managed by Prometheus.
        - MLflow records explicitly instrumented training runs; ordinary Streamlit interactions do not automatically create MLflow runs.
        """
    )
