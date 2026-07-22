# System Architecture

## Overall architecture

```mermaid
flowchart LR
  User[Business user] --> UI[Streamlit dashboard :8501]
  UI --> API[FastAPI :8000]
  API --> Data[(Prepared Parquet data)]
  API --> Models[(Joblib model artifacts)]
  API --> MLflow[MLflow :5000]
  API --> RAG[RAG service]
  RAG --> Retriever[Persisted TF-IDF retriever]
  RAG -. optional .-> Ollama[Local Ollama]
  RAG -. traces .-> LangSmith[LangSmith]
```

## Request flow

```mermaid
sequenceDiagram
  participant U as User
  participant S as Streamlit
  participant A as FastAPI
  participant M as Model/RAG service
  U->>S: Submit form or chat question
  S->>A: HTTP request to /api/v1/*
  A->>M: Validate input and execute service
  M-->>A: Prediction, forecast, recommendations, or answer
  A-->>S: JSON response or structured error
  S-->>U: Cards, charts, tables, or chat message
```

## Data flow

```mermaid
flowchart LR
  Raw[Raw commerce CSVs] --> Clean[pipelines.data.prepare_data]
  Clean --> Tables[Cleaned CSV tables]
  Clean --> Master[data/processed/master_df.parquet]
  Master --> Features[pipelines.features.build_customer_features]
  Features --> Customer[data/processed/customer_features.parquet]
  Master --> Services[API analytics, retriever, recommender]
  Customer --> Training[Training pipelines]
  Training --> Models[models/*.joblib]
```

## ML pipeline

```mermaid
flowchart TD
  D[master_df.parquet] --> F[Feature preparation]
  F --> Split[Train/test split]
  Split --> Train[scikit-learn training]
  Train --> Metrics[Metrics and plots]
  Metrics --> MLflow[MLflow run]
  Train --> Artifact[MLflow model + joblib compatibility artifact]
  Artifact --> API[FastAPI inference]
```

## RAG pipeline

```mermaid
flowchart LR
  Q[Question] --> Retrieve[TF-IDF retriever]
  Retrieve --> Context[Scored commerce context]
  Context --> Prompt[Versioned prompt template]
  Prompt --> Generate{Ollama enabled?}
  Generate -- Yes --> LLM[Ollama response]
  Generate -- No/failure --> Evidence[Retrieved context fallback]
  LLM --> Parse[Output parser]
  Evidence --> Parse
  Parse --> Answer[Answer + source labels]
  Answer --> Ops[Local RAG metrics]
  Answer -. optional tracing .-> LS[LangSmith]
```

## Docker architecture

```mermaid
flowchart TB
  subgraph Compose[Docker Compose network]
    Streamlit[Streamlit :8501] --> Backend[FastAPI :8000]
    Backend --> MLflow[MLflow :5000]
    Prometheus[Prometheus :9090] --> Backend
    Prometheus --> Node[Node Exporter]
    Prometheus --> Cadvisor[cAdvisor]
    Grafana[Grafana :3000] --> Prometheus
  end
  HostData[data/ and models/] -. mounted .-> Backend
```

## Monitoring architecture

```mermaid
flowchart LR
  API[FastAPI /metrics] --> P[Prometheus]
  Node[Node Exporter] --> P
  CAdvisor[cAdvisor] --> P
  P --> G[Grafana dashboards]
  P --> Alerts[Grafana alert rules]
  RAG[RAG service] -. traces .-> LS[LangSmith]
  Training[Training pipelines] -. experiments .-> MF[MLflow]
```

## Deployment architecture

```mermaid
flowchart LR
  Browser --> Streamlit[Render Streamlit web service]
  Streamlit -->|private hostname| Backend[Render FastAPI web service]
  Backend -->|private hostname| MLflow[Render MLflow web service]
  Backend --> Disk[(Render persistent disk)]
  Disk --> Assets[Prepared data + models]
  Store[Private object storage] -->|signed ZIP URL at startup| Disk
```

## Architectural principles

- Services load data and model artifacts lazily to keep API startup lightweight.
- The frontend uses backend service DNS in Docker and private hostnames on Render, not `localhost`.
- User content is excluded from local RAG logs; LangSmith tracing is opt-in.
- Prometheus labels use bounded values such as route, status, model, and outcome.
