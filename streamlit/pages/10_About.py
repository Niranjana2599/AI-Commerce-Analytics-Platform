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


# ============================================================
# Header
# ============================================================

page_hero(
    "Platform guide",
    "About AI Commerce Analytics",
    (
        "A modular analytics workspace combining ecommerce "
        "data, predictive machine learning, retrieval-augmented "
        "generation, and operational observability."
    ),
)


# ============================================================
# Overview
# ============================================================

overview_columns = st.columns(4)

overview_columns[0].metric(
    "Application modules",
    len(MODULES),
)

overview_columns[1].metric(
    "Local web services",
    len(SERVICES),
)

overview_columns[2].metric(
    "API architecture",
    "FastAPI",
)

overview_columns[3].metric(
    "Deployment",
    "Docker Compose",
)


# ============================================================
# Platform capabilities
# ============================================================

section_heading(
    "Platform capabilities",
    "Explore how the application modules fit into the analytics lifecycle.",
)

module_frame = pd.DataFrame(
    MODULES,
    columns=[
        "Domain",
        "Module",
        "Purpose",
        "Capability",
    ],
)

treemap = px.treemap(
    module_frame,
    path=[
        px.Constant("AI Commerce"),
        "Domain",
        "Module",
    ],
    values=[1] * len(module_frame),
    color="Domain",
    hover_data={
        "Purpose": True,
        "Capability": True,
    },
    color_discrete_map={
        "Analytics": "#0ea5e9",
        "Machine Learning": "#6366f1",
        "Generative AI": "#a855f7",
        "Operations": "#10b981",
        "(?)": "#64748b",
    },
    title="Module capability map",
)

treemap.update_traces(
    root_color="rgba(15,23,42,.45)"
)

treemap.update_layout(
    height=430,
    margin=dict(
        l=20,
        r=20,
        t=55,
        b=20,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8fafc"),
    title_font=dict(size=17),
)

st.plotly_chart(
    treemap,
    width="stretch",
    config=PLOT_CONFIG,
)

with st.expander(
    "View all modules",
    expanded=False,
):
    st.dataframe(
        module_frame,
        hide_index=True,
        width="stretch",
    )


# ============================================================
# System architecture
# ============================================================

section_heading(
    "System architecture",
    (
        "Each service has a focused responsibility and "
        "communicates through stable interfaces."
    ),
)

st.markdown(
    """
    <div style="
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
        gap:.8rem;
        align-items:stretch
    ">

      <div class="module-card">
        <div class="eyebrow">Experience</div>
        <h3>Streamlit</h3>
        <p>
          Dashboards, forms, Plotly charts, analytics views,
          and conversational RAG interface.
        </p>
      </div>

      <div class="module-card">
        <div class="eyebrow">Application API</div>
        <h3>FastAPI</h3>
        <p>
          Validated requests, analytics services,
          prediction endpoints, and RAG orchestration.
        </p>
      </div>

      <div class="module-card">
        <div class="eyebrow">Retrieval</div>
        <h3>LangChain + FAISS</h3>
        <p>
          LangChain orchestrates retrieval while FAISS
          performs semantic similarity search over
          embedded knowledge-base chunks.
        </p>
      </div>

      <div class="module-card">
        <div class="eyebrow">Generation</div>
        <h3>Ollama + Llama 3.2</h3>
        <p>
          Local LLM generation using retrieved commerce
          context to produce grounded answers.
        </p>
      </div>

      <div class="module-card">
        <div class="eyebrow">Operations</div>
        <h3>Observability</h3>
        <p>
          MLflow, Prometheus, Grafana, and optional
          LangSmith tracing for experiments and runtime
          monitoring.
        </p>
      </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "RAG request flow: Browser → Streamlit → FastAPI → "
    "LangChain → FAISS → retrieved chunks → prompt → "
    "Ollama/Llama 3.2 → grounded answer."
)

st.caption(
    "Operational telemetry flows to Prometheus/Grafana, "
    "while optional LangSmith tracing captures RAG execution."
)


# ============================================================
# RAG architecture
# ============================================================

section_heading(
    "RAG pipeline",
    (
        "The chatbot uses semantic retrieval before local "
        "LLM generation."
    ),
)

rag_columns = st.columns(5)

rag_steps = [
    (
        "01",
        "Knowledge Base",
        "Products, customer reviews, policies/FAQs, and dashboard metrics.",
    ),
    (
        "02",
        "Embeddings",
        "Local Hugging Face sentence-transformer embeddings.",
    ),
    (
        "03",
        "FAISS",
        "Semantic similarity search over the persisted vector index.",
    ),
    (
        "04",
        "LangChain",
        "Retrieval and prompt orchestration.",
    ),
    (
        "05",
        "Ollama",
        "Local Llama 3.2 generation using retrieved context.",
    ),
]

for column, (step, title, description) in zip(
    rag_columns,
    rag_steps,
):
    with column:
        st.markdown(
            f"""
            <div class="module-card">
                <div class="eyebrow">{step}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Local service directory
# ============================================================

section_heading(
    "Local service directory",
    "Open the development services exposed by Docker Compose.",
)

service_columns = st.columns(3)

for index, (
    name,
    purpose,
    url,
    port,
) in enumerate(SERVICES):

    with service_columns[index % 3]:

        st.markdown(
            f"""
            <div class="module-card">
                <div class="eyebrow">Port {port}</div>
                <h3>{name}</h3>
                <p>{purpose}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.link_button(
            f"Open {name}",
            url,
            width="stretch",
        )


# ============================================================
# Health check
# ============================================================

health_column, command_column = st.columns(
    [1, 2]
)

with health_column:

    if st.button(
        "Check FastAPI health",
        type="primary",
        width="stretch",
    ):

        with st.spinner(
            "Checking the backend..."
        ):

            health, error = api_request(
                "GET",
                "/health",
            )

        if error:

            st.error(error)

        elif health:

            st.success(
                "FastAPI is reachable and healthy."
            )

            st.json(
                health,
                expanded=False,
            )


with command_column:

    st.caption(
        "Start or refresh the complete local stack"
    )

    st.code(
        "docker compose up -d --build",
        language="powershell",
    )

    st.caption(
        "Check service health with "
        "`docker compose ps`."
    )


# ============================================================
# Technology stack
# ============================================================

section_heading(
    "Technology stack",
    "A portable Python and container-based implementation.",
)

stack_columns = st.columns(4)

stack_items = [
    (
        "Data and ML",
        "Python, pandas, scikit-learn, XGBoost, joblib, PyArrow",
    ),
    (
        "Application",
        "FastAPI, Uvicorn, Streamlit, Plotly, requests",
    ),
    (
        "RAG and LLM",
        "LangChain, FAISS, Hugging Face embeddings, Ollama, Llama 3.2",
    ),
    (
        "Operations",
        "MLflow, LangSmith, Prometheus, Grafana, cAdvisor",
    ),
]

for column, (
    title,
    description,
) in zip(
    stack_columns,
    stack_items,
):

    with column:

        st.markdown(
            f"""
            <div class="module-card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Responsible use
# ============================================================

section_heading(
    "Responsible use",
    (
        "Understand what the application demonstrates and "
        "where human review remains important."
    ),
)

guidance_columns = st.columns(2)

with guidance_columns[0]:

    st.success(
        "Use this platform for analytics exploration, "
        "model demonstrations, operational learning, "
        "and decision support."
    )

    st.markdown(
        """
        - Validate input data and model artifacts before demonstrations.
        - Review predictions alongside the feature context shown on each page.
        - Use monitoring tools to inspect runtime health and experiment history.
        - Review retrieved sources when using the RAG chatbot.
        """
    )


with guidance_columns[1]:

    st.warning(
        "Predictions and RAG answers should not be treated "
        "as autonomous business decisions."
    )

    st.markdown(
        """
        - Model quality depends on the training data and evaluation design.
        - RAG answer quality depends on retrieval quality and the available knowledge base.
        - The local LLM generates answers from the retrieved context.
        - LangSmith tracing is optional and separate from Prometheus infrastructure monitoring.
        """
    )


# ============================================================
# Developer quick reference
# ============================================================

with st.expander(
    "Developer quick reference"
):

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
        "See the repository `README.md` for training workflows, "
        "MLflow experiments, Prometheus queries, Grafana dashboards, "
        "RAG configuration, and LangSmith configuration."
    )