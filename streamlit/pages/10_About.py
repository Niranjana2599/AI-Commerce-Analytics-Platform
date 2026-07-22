"""Platform guide, architecture overview, and local service directory."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page


setup_page("About")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
MODULES = [
    ("Analytics", "Dashboard", "Business KPIs and commercial trends", "Descriptive"),
    ("Analytics", "Customer Analytics", "RFM, segments, spend, and geography", "Descriptive"),
    ("Machine Learning", "Churn Prediction", "Estimate customer attrition risk", "Predictive"),
    ("Machine Learning", "CLV Prediction", "Estimate future customer value", "Predictive"),
    ("Machine Learning", "Delivery Delay", "Assess delivery-delay risk", "Predictive"),
    ("Machine Learning", "Recommendations", "Retrieve relevant products", "Personalization"),
    ("Machine Learning", "Sentiment Analysis", "Classify customer review language", "NLP"),
    ("Machine Learning", "Demand Forecasting", "Estimate near-term order demand", "Forecasting"),
    ("Generative AI", "RAG Chatbot", "Retrieve grounded commerce answers", "RAG"),
    ("Operations", "RAG Operations", "Inspect RAG quality and runtime metrics", "Observability"),
]
SERVICES = [
    ("Streamlit", "Application UI", "http://localhost:8501", "8501"),
    ("FastAPI", "API and Swagger", "http://localhost:8000/docs", "8000"),
    ("MLflow", "Experiment tracking", "http://localhost:5000", "5000"),
    ("Prometheus", "Metrics and queries", "http://localhost:9090", "9090"),
    ("Grafana", "Monitoring dashboards", "http://localhost:3000", "3000"),
]


page_hero(
    "Platform guide",
    "About AI Commerce Analytics",
    "A modular analytics workspace combining ecommerce data, predictive machine learning, retrieval-augmented answers, and operational observability.",
)

overview_columns = st.columns(4)
overview_columns[0].metric("Application modules", len(MODULES))
overview_columns[1].metric("Local web services", len(SERVICES))
overview_columns[2].metric("API architecture", "FastAPI")
overview_columns[3].metric("Deployment", "Docker Compose")

section_heading("Platform capabilities", "Explore how the application modules fit into the analytics lifecycle.")
module_frame = pd.DataFrame(MODULES, columns=["Domain", "Module", "Purpose", "Capability"])
treemap = px.treemap(
    module_frame,
    path=[px.Constant("AI Commerce"), "Domain", "Module"],
    values=[1] * len(module_frame),
    color="Domain",
    hover_data={"Purpose": True, "Capability": True},
    color_discrete_map={
        "Analytics": "#0ea5e9",
        "Machine Learning": "#6366f1",
        "Generative AI": "#a855f7",
        "Operations": "#10b981",
        "(?)": "#64748b",
    },
    title="Module capability map",
)
treemap.update_traces(root_color="rgba(15,23,42,.45)")
treemap.update_layout(
    height=430,
    margin=dict(l=20, r=20, t=55, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8fafc"),
    title_font=dict(size=17),
)
st.plotly_chart(treemap, width="stretch", config=PLOT_CONFIG)

with st.expander("View all modules", expanded=False):
    st.dataframe(module_frame, hide_index=True, width="stretch")

section_heading("System architecture", "Each service has a focused responsibility and communicates through stable interfaces.")
st.markdown(
    """
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.8rem;align-items:stretch">
      <div class="module-card"><div class="eyebrow">Experience</div><h3>Streamlit</h3><p>Forms, dashboards, Plotly charts, and conversational UI.</p></div>
      <div class="module-card"><div class="eyebrow">Application API</div><h3>FastAPI</h3><p>Validated requests, analytics services, and prediction endpoints.</p></div>
      <div class="module-card"><div class="eyebrow">Intelligence</div><h3>Models + RAG</h3><p>Persisted ML artifacts, prepared data, and knowledge retrieval.</p></div>
      <div class="module-card"><div class="eyebrow">Operations</div><h3>Observability</h3><p>MLflow, Prometheus, Grafana, and optional LangSmith tracing.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Request flow: Browser → Streamlit → FastAPI → prepared data and model artifacts. Operational telemetry flows to the relevant monitoring service.")

section_heading("Local service directory", "Open the development services exposed by Docker Compose.")
service_columns = st.columns(3)
for index, (name, purpose, url, port) in enumerate(SERVICES):
    with service_columns[index % 3]:
        st.markdown(
            f'<div class="module-card"><div class="eyebrow">Port {port}</div><h3>{name}</h3><p>{purpose}</p></div>',
            unsafe_allow_html=True,
        )
        st.link_button(f"Open {name}", url, width="stretch")

health_column, command_column = st.columns([1, 2])
with health_column:
    if st.button("Check FastAPI health", type="primary", width="stretch"):
        with st.spinner("Checking the backend..."):
            health, error = api_request("GET", "/health")
        if error:
            st.error(error)
        elif health:
            st.success("FastAPI is reachable and healthy.")
            st.json(health, expanded=False)
with command_column:
    st.caption("Start or refresh the complete local stack")
    st.code("docker compose up -d --build", language="powershell")
    st.caption("Check service health with `docker compose ps`.")

section_heading("Technology stack", "A portable Python and container-based implementation.")
stack_columns = st.columns(4)
stack_items = [
    ("Data and ML", "Python, pandas, scikit-learn, XGBoost, joblib, PyArrow"),
    ("Application", "FastAPI, Uvicorn, Streamlit, Plotly, requests"),
    ("ML and RAG operations", "MLflow, LangSmith tracing, privacy-safe RAG metrics"),
    ("Infrastructure", "Docker Compose, Prometheus, Grafana, cAdvisor"),
]
for column, (title, description) in zip(stack_columns, stack_items):
    with column:
        st.markdown(f'<div class="module-card"><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)

section_heading("Responsible use", "Understand what the application demonstrates and where human review remains important.")
guidance_columns = st.columns(2)
with guidance_columns[0]:
    st.success(
        "Use this platform for analytics exploration, model demonstrations, operational learning, and decision support."
    )
    st.markdown(
        """
        - Validate input data and model artifacts before demonstrations.
        - Review predictions alongside the feature context shown on each page.
        - Use monitoring tools to inspect runtime health and experiment history.
        """
    )
with guidance_columns[1]:
    st.warning("Predictions and RAG answers should not be treated as autonomous business decisions.")
    st.markdown(
        """
        - Model quality depends on the training data and evaluation design.
        - Some API responses do not include confidence or historical series.
        - LangSmith tracing is optional and separate from Prometheus infrastructure monitoring.
        """
    )

with st.expander("Developer quick reference"):
    st.code(
        """# Start the platform
docker compose up -d --build

# Verify containers
docker compose ps

# Stop the platform
docker compose down""",
        language="powershell",
    )
    st.markdown(
        "See the repository `README.md` for training workflows, MLflow experiments, Prometheus queries, Grafana dashboards, and LangSmith configuration."
    )
