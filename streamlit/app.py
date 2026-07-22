"""Production landing page for the AI Commerce Analytics workspace."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page


setup_page("Welcome")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
MODULES = [
    ("Understand", "Dashboard", "Live commerce KPIs and platform readiness", "pages/01_Home.py", "📊"),
    ("Understand", "Customer Analytics", "Explore RFM, value, spending, and geography", "pages/02_Customer_Analytics.py", "👥"),
    ("Predict", "Churn Prediction", "Estimate retention risk and review drivers", "pages/03_Churn_Prediction.py", "⚠️"),
    ("Predict", "CLV Prediction", "Estimate customer lifetime value", "pages/04_CLV_Prediction.py", "💎"),
    ("Predict", "Delivery Delay", "Assess delivery-delay risk", "pages/05_Delivery_Delay.py", "🚚"),
    ("Personalize", "Product Recommendation", "Generate relevant product suggestions", "pages/06_Product_Recommendation.py", "🎯"),
    ("Understand", "Sentiment Analysis", "Classify and explore review language", "pages/07_Sentiment_Analysis.py", "💬"),
    ("Plan", "Demand Forecasting", "Estimate near-term order demand", "pages/08_Demand_Forecasting.py", "📈"),
    ("Ask", "RAG Chatbot", "Ask grounded commerce questions", "pages/09_RAG_Chatbot.py", "🤖"),
    ("Operate", "RAG Operations", "Monitor RAG quality and latency", "pages/11_RAG_Operations.py", "🧭"),
    ("Operate", "System Monitoring", "Inspect live service and platform telemetry", "pages/12_Monitoring.py", "🖥️"),
]


@st.cache_data(ttl=30, show_spinner=False)
def load_readiness():
    """Cache the read-only API readiness check to avoid repeated calls."""
    return api_request("GET", "/health", timeout=5)


page_hero(
    "Commerce intelligence workspace",
    "AI Commerce Analytics Platform",
    "Move from business exploration to predictive models, demand planning, grounded AI answers, and production observability in one responsive application.",
)

with st.spinner("Checking platform readiness..."):
    health, health_error = load_readiness()
api_online = bool(health and health.get("status") == "ok")

status_columns = st.columns([1, 1, 1, 1, 2])
status_columns[0].metric("API status", "Online" if api_online else "Unavailable")
status_columns[1].metric("Modules", len(MODULES))
status_columns[2].metric("Frontend", "Streamlit")
status_columns[3].metric("Visualizations", "Plotly")
with status_columns[4]:
    st.markdown("#### Ready to explore?")
    st.page_link("pages/01_Home.py", label="Open executive dashboard", icon="📊", width="stretch")

if api_online:
    st.success(f"FastAPI is healthy in the **{health.get('environment', 'configured')}** environment.")
else:
    st.warning("FastAPI is not currently reachable. Informational pages will work, but live analytics and predictions require the backend.")
    if health_error:
        st.caption(health_error)

section_heading("Choose your workflow", "Start with a business question, then open the most relevant workspace.")
tabs = st.tabs(["Understand", "Predict", "Personalize", "Plan", "Ask", "Operate"])
for tab, workflow in zip(tabs, ["Understand", "Predict", "Personalize", "Plan", "Ask", "Operate"]):
    workflow_modules = [module for module in MODULES if module[0] == workflow]
    with tab:
        columns = st.columns(min(3, len(workflow_modules)))
        for column, (_, name, description, page, icon) in zip(columns, workflow_modules):
            with column:
                st.markdown(
                    f'<div class="module-card"><div style="font-size:1.4rem">{icon}</div><h3>{name}</h3><p>{description}</p></div>',
                    unsafe_allow_html=True,
                )
                st.page_link(page, label=f"Open {name}", icon=icon, width="stretch")

section_heading("Platform coverage", "See how the application balances analytics, prediction, AI assistance, planning, and operations.")
coverage = pd.DataFrame(MODULES, columns=["Workflow", "Module", "Description", "Page", "Icon"])
coverage_counts = coverage.groupby("Workflow", as_index=False).size().rename(columns={"size": "Modules"})
chart_column, guide_column = st.columns([1.35, 1])
with chart_column:
    coverage_chart = px.sunburst(
        coverage,
        path=[px.Constant("AI Commerce"), "Workflow", "Module"],
        values=[1] * len(coverage),
        color="Workflow",
        hover_data={"Description": True},
        color_discrete_sequence=["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#a855f7", "#64748b"],
        title="Application capability map",
    )
    coverage_chart.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        title_font=dict(size=17),
    )
    st.plotly_chart(coverage_chart, width="stretch", config=PLOT_CONFIG)
with guide_column:
    st.markdown("#### A practical starting path")
    st.markdown(
        """
        1. Open **Dashboard** to confirm business and API readiness.
        2. Use **Customer Analytics** to understand the available population.
        3. Choose a prediction or planning workflow for a focused decision.
        4. Use the **RAG Chatbot** for grounded knowledge retrieval.
        5. Check **System Monitoring**, **Grafana**, and **MLflow** for operations and experiments.
        """
    )
    st.dataframe(coverage_counts, hide_index=True, width="stretch")

section_heading("Platform services", "Open the supporting tools used for APIs, experiments, monitoring, and tracing.")
service_columns = st.columns(5)
services = [
    ("FastAPI docs", "http://localhost:8000/docs", "API contracts"),
    ("MLflow", "http://localhost:5000", "Training runs"),
    ("Prometheus", "http://localhost:9090", "Raw metrics"),
    ("Grafana", "http://localhost:3000", "Dashboards"),
    ("LangSmith", "https://smith.langchain.com", "RAG traces"),
]
for column, (name, url, purpose) in zip(service_columns, services):
    with column:
        st.markdown(f'<div class="module-card"><h3>{name}</h3><p>{purpose}</p></div>', unsafe_allow_html=True)
        st.link_button(f"Open {name}", url, width="stretch")

with st.expander("Local startup and platform boundaries"):
    st.code("docker compose up -d --build", language="powershell")
    st.markdown(
        """
        - Predictions depend on the mounted data and model artifacts.
        - MLflow records explicitly instrumented training runs; browsing Streamlit does not automatically create experiments.
        - Prometheus and Grafana handle application and infrastructure telemetry.
        - LangSmith is optional and applies specifically to traced RAG execution.
        - Outputs support analysis and demonstration; important business decisions still require human review.
        """
    )
