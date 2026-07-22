# Deployment Guide

## Local Docker Compose

Prerequisites: Docker Desktop and the project data/model artifacts under `data/` and `models/`.

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose ps
```

| Service | URL |
| --- | --- |
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Stop without deleting named volumes:

```powershell
docker compose down
```

Rebuild after dependency or Dockerfile changes:

```powershell
docker compose up --build -d
```

## Environment configuration

Use `.env`; never commit it. Important values include:

| Variable | Purpose |
| --- | --- |
| `CORS_ORIGINS` | Allowed browser origin for FastAPI |
| `RAG_LLM_ENABLED` | Enables optional Ollama generation |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Local LLM connection |
| `LANGCHAIN_*` | Opt-in LangSmith tracing configuration |
| `GRAFANA_ADMIN_PASSWORD` | Grafana administrator password |
| `*_PORT` | Local port mappings |

## MLflow

MLflow runs as a Compose service with SQLite metadata and a mounted artifact folder.

```powershell
docker compose exec backend python -m pipelines.training.train all
```

Open MLflow to compare parameters, metrics, artifacts, and registered run context.

## Monitoring

- FastAPI exposes `/metrics`.
- Prometheus scrapes FastAPI, itself, Node Exporter, and cAdvisor every 15 seconds.
- Grafana provisions the Prometheus datasource, dashboards, and alert rules from `grafana/`.
- Local Grafana defaults to `admin`/`admin`; change the password before sharing the environment.

## GitHub Actions

The CI workflow under `.github/workflows/ci.yml` runs on pull requests and pushes to `main`. It installs dependencies, checks imports, tests, formatting/linting, Compose validity, and Docker builds. Keep CI green before deployment.

## Render deployment

The root [render.yaml](../render.yaml) declares FastAPI, Streamlit, and MLflow web services. Render does not run the local Compose stack as one deployment, so Prometheus/Grafana are intentionally local-stack components in the current blueprint.

1. Build a private ZIP containing `processed/master_df.parquet` and required `models/*.joblib` files.
2. Upload it to private object storage and generate a time-limited signed URL.
3. Create a Render Blueprint from the Git repository.
4. Set `ASSET_BUNDLE_URL` as a Render secret.
5. Set backend `CORS_ORIGINS` to the public Streamlit URL after first deployment.
6. Verify `/api/v1/health`, `/docs`, Streamlit, and MLflow URLs.

## Startup-failure troubleshooting

| Symptom | Check |
| --- | --- |
| Backend exits | Model/data mounts, `MASTER_DATA_PATH`, `MODELS_DIR`, artifact file names |
| Streamlit cannot reach API | `FASTAPI_BASE_URL`, backend health, Compose network |
| MLflow has no runs | `MLFLOW_TRACKING_URI`, MLflow health, training command |
| Prometheus target down | `http://localhost:8000/metrics`, `http://localhost:9090/targets` |
| Grafana empty | Prometheus datasource, scrape target state, dashboard provisioning mounts |

## Production checklist

- Replace development credentials and restrict service ports.
- Terminate TLS through a trusted reverse proxy or platform ingress.
- Use secrets management for API keys and asset URLs.
- Back up MLflow, Grafana, and Prometheus persistent volumes according to retention policy.
- Pin/update images and dependencies through a reviewed change process.
