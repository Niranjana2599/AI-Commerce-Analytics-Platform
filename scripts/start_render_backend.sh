#!/bin/sh
# Stop immediately if asset bootstrap or Uvicorn startup fails.
set -eu

# Render mounts the deployment disk here; allow an override for other platforms.
export DEPLOY_ASSETS_DIR="${DEPLOY_ASSETS_DIR:-/var/data}"
export MASTER_DATA_PATH="${MASTER_DATA_PATH:-${DEPLOY_ASSETS_DIR}/processed/master_df.parquet}"
export MODELS_DIR="${MODELS_DIR:-${DEPLOY_ASSETS_DIR}/models}"

# Fail deployment early when the data/model bundle is unavailable or malformed.
python /app/scripts/bootstrap_render_assets.py

# Render provides PORT; retain 8000 as a safe default for non-Render execution.
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
