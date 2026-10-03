# Project Structure

```text
AI-Commerce-Analytics-Platform/
├── backend/                 FastAPI API, single-supervisor agent, tools, services, monitoring
├── streamlit/               Multi-page Streamlit dashboard and API client utilities
├── src/                     Shared data loading, features, retrieval, recommendations, MLflow helpers
├── pipelines/               Repeatable preparation, feature, training, service, and evaluation jobs
├── notebooks/               Exploratory/model-development notebooks
├── data/                    Raw and generated processed data artifacts
├── models/                  Generated joblib model/retriever artifacts
├── mlflow/                  Generated MLflow SQLite/artifact storage
├── rag_ops/                 Generated privacy-safe RAG events and aggregate reports
├── grafana/                 Provisioned Grafana datasource, alerts, and dashboards
├── scripts/                 Runtime bootstrap scripts, including Render asset restore
├── .github/workflows/       CI workflow
├── docker-compose.yml       Local multi-service stack
├── prometheus.yml           Prometheus scrape jobs
├── render.yaml              Render Blueprint
├── .env.example             Safe environment variable template
└── README.md                Quick-start and operational overview
```

## Application folders

| Path | Responsibility |
| --- | --- |
| `backend/app/api/` | HTTP route definitions and error conversion |
| `backend/app/api/agent.py` | `POST /api/v1/agent/ask` entry point for the Agentic Analyst |
| `backend/app/core/` | Environment-aware settings |
| `backend/app/schemas/` | Pydantic request/response contracts |
| `backend/app/agents/` | Request state, planner, router, selector, supervisor, synthesis, safe presentation |
| `backend/app/tools/` | Allowlisted, validated adapters around existing domain services |
| `backend/app/guardrails/` | Lightweight request screening and policy checks |
| `backend/app/evaluation/` | Golden requests and runtime execution evaluation |
| `backend/app/services/` | Existing model serving, customer analytics, and RAG services used by routes and tools |
| `backend/app/llmops/` | Prompt registry, KB versioning, LangSmith adapter, local RAG observability |
| `backend/app/monitoring/` | Prometheus metric definitions and helpers |
| `backend/tests/` | API, agent runtime, tool contract, guardrail, fallback, and evaluation tests |
| `streamlit/pages/` | Numbered dashboard pages, including Agentic Analyst |
| `streamlit/utils/` | API request and UI helper functions, including shared agent request/evidence presentation |
| `src/data.py` | Dataset loading, cleaning, and master-table construction |
| `src/features/` | Reusable customer feature engineering |
| `src/services/` | TF-IDF retrieval and recommendation baseline classes |

## Pipeline folders

| Path | Command | Output |
| --- | --- | --- |
| `pipelines/data/prepare_data.py` | `python -m pipelines.data.prepare_data` | Cleaned tables and `master_df.parquet` |
| `pipelines/features/build_customer_features.py` | `python -m pipelines.features.build_customer_features` | `customer_features.parquet` |
| `pipelines/training/train.py` | `python -m pipelines.training.train <module>` | MLflow runs and compatibility artifacts |
| `pipelines/services/build_retriever.py` | `python -m pipelines.services.build_retriever` | `models/faiss_ecommerce/` (persisted FAISS index) |
| `pipelines/services/build_recommender.py` | `python -m pipelines.services.build_recommender` | recommender artifact |
| `pipelines/evaluation/evaluate_rag.py` | `python -m pipelines.evaluation.evaluate_rag --dataset <name>` | LangSmith evaluation experiment |

## Operational files

| File | Purpose |
| --- | --- |
| `backend/Dockerfile` | FastAPI runtime image and Render-compatible startup path |
| `streamlit/Dockerfile` | Streamlit runtime image |
| `docker-compose.yml` | API, UI, MLflow, Prometheus, Grafana, and exporters |
| `prometheus.yml` | Target scrape configuration |
| `grafana/provisioning/` | Datasource, dashboard provider, and alert rules as code |
| `render.yaml` | Render web-service configuration and persistent-disk settings |
| `.dockerignore` | Excludes generated data/artifacts from image context |

The supervisor calls existing services through registered tools; it does not contain duplicate ML or RAG implementations. See [Agentic AI Orchestration](11_Agentic_AI_Orchestration.md) for the runtime flow.

## Generated versus source-controlled content

Source code, docs, Compose configuration, Grafana dashboards, and pipeline definitions should be committed. Data, models, `mlflow/`, `mlruns/`, `rag_ops/`, virtual environments, and local runtime dependency folders are generated or potentially sensitive and are ignored by Git.
