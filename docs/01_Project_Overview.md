# AI-Commerce-Analytics-Platform: Project Overview

## Business problem

E-commerce teams often work across disconnected reports, operational datasets, and model notebooks. This project consolidates customer intelligence, predictive analytics, recommendations, review analysis, demand signals, and retrieval-assisted data exploration into one deployable workspace.

## Objectives

- Turn prepared commerce data into useful customer and operational insights.
- Expose predictive workflows through a documented FastAPI service and Streamlit dashboard.
- Make model training reproducible with MLflow.
- Make the RAG chatbot observable with LangSmith and local privacy-safe RAG operations logs.
- Operate the stack with Docker Compose, Prometheus, and Grafana.

## Delivered features

| Area | Capability |
| --- | --- |
| Customer analytics | Revenue, order, customer, and average-order-value KPIs; dashboard-ready analysis views. |
| Predictions | Churn and CLV prediction endpoints backed by saved artifacts; a transparent delivery-delay baseline. |
| Recommendations | Popular-unseen product recommendations for a customer. |
| Sentiment | Review sentiment classification with a safe keyword fallback if an artifact is unavailable. |
| Forecasting | Daily demand forecasting endpoint. |
| Agentic Analyst | One LangGraph supervisor orchestrates allowlisted tools around existing commerce services; optional local LLM planning, result validation, and execution evidence. |
| RAG chatbot | Direct retrieval over persisted FAISS commerce documents, versioned prompts, optional local Ollama generation, and source labels. |
| MLOps/LLMOps | MLflow experiments, LangSmith RAG tracing, offline RAG evaluation, and local aggregate RAG reports. |
| Monitoring | Prometheus metrics and provisioned Grafana dashboards/alerts for API, ML, RAG, and infrastructure signals. |

## Technology stack

| Layer | Technologies |
| --- | --- |
| Language & data | Python 3.13, pandas, NumPy, PyArrow |
| ML | scikit-learn, XGBoost, LightGBM, joblib |
| APIs & UI | FastAPI, Uvicorn, Streamlit, Plotly |
| Experiment tracking | MLflow with SQLite metadata and local artifacts |
| RAG | Persisted FAISS index, optional Ollama, LangSmith tracing; also callable by the Agentic Analyst tool registry |
| Delivery | Docker, Docker Compose, Render Blueprint |
| Observability | Prometheus, Node Exporter, cAdvisor, Grafana |
| Quality | pytest, Black, Flake8, GitHub Actions |

> **Accuracy note:** the active RAG service loads the persisted FAISS index under `models/faiss_ecommerce/`; Ollama generation is optional. The Agentic Analyst reuses this service through its registered RAG search tool.

## Key highlights

- A single API and dashboard surface across multiple e-commerce use cases.
- A single LangGraph supervisor routes natural-language business questions to allowlisted tools around existing services; this is not a multi-agent system.
- Artifact-aware startup: data and models are mounted locally and can be bootstrapped to a Render persistent disk.
- Production-minded operational visibility: health checks, metrics, dashboards, alerts, tracing, and CI checks.
- Clear separation between pipelines, API services, models, user interface, and observability tooling.

## Screenshot placeholders

| Asset | Suggested content |
| --- | --- |
| `docs/images/home-dashboard.png` | Streamlit home dashboard and KPI cards |
| `docs/images/mlflow-runs.png` | MLflow experiment comparison |
| `docs/images/grafana-overview.png` | Grafana API or RAG dashboard |
| `docs/images/rag-trace.png` | LangSmith RAG trace showing retrieval and LLM stages |

## Related documents

- [System Architecture](02_System_Architecture.md)
- [Agentic AI Orchestration](11_Agentic_AI_Orchestration.md)
- [Machine Learning](04_Machine_Learning.md)
- [RAG Architecture](05_RAG_Architecture.md)
- [Deployment Guide](07_Deployment_Guide.md)
