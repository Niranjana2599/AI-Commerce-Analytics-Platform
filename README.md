# AI Commerce Analytics Platform

An ecommerce analytics project that combines data preparation, machine-learning models, a FastAPI backend, and a Streamlit dashboard.

## Project tech stack

- **Data and ML:** Python, pandas, scikit-learn, XGBoost, joblib, PyArrow
- **Backend:** FastAPI and Uvicorn
- **Frontend:** Streamlit, Plotly, and requests
- **Experiment tracking:** MLflow with SQLite metadata and local artifact storage
- **Deployment:** Docker and Docker Compose
- **Data artifacts:** Parquet/CSV prepared datasets and saved `.joblib` models

## Project structure

```text
AI-Commerce-Analytics-Platform/
├── backend/                 # FastAPI application and API Dockerfile
├── streamlit/               # Streamlit dashboard and UI Dockerfile
├── src/                     # Shared data, feature, and service code
├── pipelines/               # Repeatable data/feature/model artifact jobs
├── data/processed/          # Prepared dataset mounted into FastAPI
├── models/                  # Saved model artifacts mounted into FastAPI
├── mlflow/                  # Generated MLflow SQLite metadata and artifacts (ignored by Git)
├── notebooks/               # Exploratory and model-development notebooks
├── docker-compose.yml       # Starts the complete application stack
├── .dockerignore            # Keeps Docker build contexts small
└── .env.example             # Docker configuration template
```

## Docker installation

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and confirm it is running:

```powershell
docker --version
docker compose version
```

Optionally create local configuration from the template:

```powershell
Copy-Item .env.example .env
```

The default configuration works without creating `.env`.

## Start with Docker Compose

Build images and start FastAPI, Streamlit, and the MLflow Tracking Server in the background:

```powershell
docker compose up --build -d
```

Follow service logs during startup:

```powershell
docker compose logs -f
```

## Stop and rebuild

Stop containers while keeping data/models on the host:

```powershell
docker compose down
```

Rebuild after Dockerfile or dependency changes:

```powershell
docker compose build --no-cache
docker compose up -d
```

## Service access and health checks

| Service | URL | Container health check |
| --- | --- | --- |
| Streamlit dashboard | [http://localhost:8501](http://localhost:8501) | `http://localhost:8501/_stcore/health` |
| FastAPI API | [http://localhost:8000](http://localhost:8000) | `http://localhost:8000/api/v1/health` |
| Swagger UI | [http://localhost:8000/docs](http://localhost:8000/docs) | Uses the FastAPI health check |
| MLflow UI | [http://localhost:5000](http://localhost:5000) | `http://localhost:5000/health` |

Check container status:

```powershell
docker compose ps
```

Both services should show `healthy`. You can also query FastAPI directly:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

## Docker networking and volumes

Compose creates the `ai-commerce-network` bridge network. Streamlit uses Docker DNS to call `http://backend:8000/api/v1`; it never uses `localhost` inside its container.

The backend receives these read-only mounts, so large artifacts are not copied into the Docker image:

- `./data:/app/data:ro`
- `./models:/app/models:ro`

MLflow persists experiment metadata in `./mlflow/mlflow.db` and artifacts in `./mlflow/artifacts` using a Docker volume mapping.

## MLflow experiment tracking

The MLflow-enabled training command supports these experiments:

- `churn` — classification metrics, confusion matrix, ROC, PR, and feature importance
- `clv` and `delivery_delay` — regression metrics, prediction plot, and feature importance
- `recommendations` — Precision@K, Recall@K, MAP, and NDCG
- `sentiment` — classification metrics, confusion matrix, ROC, and PR curves
- `demand_forecasting` — RMSE, MAE, MAPE, and prediction plot

Each run logs parameters, dataset version, random seed, feature names, Git commit hash, training duration, MLflow model artifact, and a backward-compatible joblib artifact.

### Start MLflow locally

Install project dependencies, then point MLflow at a local tracking directory:

```powershell
python -m pip install -r backend/requirements.txt
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
```

For the Docker tracking server, start the full stack instead:

```powershell
docker compose up --build -d
```

Open the MLflow UI at [http://localhost:5000](http://localhost:5000).

### Log experiments

With the Docker stack running, execute a training job in the backend container:

```powershell
docker compose exec backend python -m pipelines.training.train churn
docker compose exec backend python -m pipelines.training.train clv
docker compose exec backend python -m pipelines.training.train delivery_delay
docker compose exec backend python -m pipelines.training.train recommendations
docker compose exec backend python -m pipelines.training.train sentiment
docker compose exec backend python -m pipelines.training.train demand_forecasting
```

Run every experiment sequentially with:

```powershell
docker compose exec backend python -m pipelines.training.train all
```

### Compare experiments

1. Open the MLflow UI.
2. Select an `AI-Commerce-*` experiment.
3. Select two or more runs and choose **Compare**.
4. Compare metrics, parameters, feature-importance plots, curves, and model artifacts.

### Verify MLflow

```powershell
docker compose ps
docker compose logs mlflow
```

`ai-commerce-mlflow` should be healthy. After running a training command, refresh the MLflow UI and confirm that the experiment, run metrics, parameters, and artifacts appear.

### MLflow screenshots

Add screenshots here after the first successful run:

- MLflow experiment list
- Run comparison table
- Feature importance, ROC, PR, or regression prediction artifacts

## Troubleshooting

- **Port already in use:** change `BACKEND_PORT` or `STREAMLIT_PORT` in `.env`, then run `docker compose up -d` again.
- **A service is unhealthy:** inspect logs with `docker compose logs backend` or `docker compose logs streamlit`.
- **Model/data not found:** ensure `data/processed/master_df.parquet` and required `.joblib` files exist on the host before starting Compose.
- **Streamlit cannot call the API:** confirm `docker compose ps` shows `backend` as healthy; Compose injects the correct `http://backend:8000/api/v1` URL.
- **Changes are not visible:** run `docker compose up --build -d`, or use the no-cache rebuild command above after dependency changes.
- **MLflow UI has no runs:** execute one of the `pipelines.training.train` commands and confirm `MLFLOW_TRACKING_URI` is `http://mlflow:5000` inside the backend container.
