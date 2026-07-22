"""Fetch model and data assets to a Render persistent disk at runtime.

The optional ASSET_BUNDLE_URL must point to a ZIP with this layout:
    processed/master_df.parquet
    models/<saved .joblib artifacts>
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ASSETS_DIR = Path(os.getenv("DEPLOY_ASSETS_DIR", "/var/data"))
DATASET = ASSETS_DIR / "processed" / "master_df.parquet"
MODELS_DIR = ASSETS_DIR / "models"
REQUIRED_MODEL_FILES = (
    "customer_churn_model.joblib",
    "customer_clv_model.joblib",
    "customer_review_sentiment_model.joblib",
    "demand_forecasting_model.joblib",
    "product_recommender_system.joblib",
    "rag_retriever.joblib",
)


def assets_ready() -> bool:
    """Check the minimum asset set required by the FastAPI deployment."""
    return DATASET.exists() and all((MODELS_DIR / filename).exists() for filename in REQUIRED_MODEL_FILES)


def download_and_extract(url: str) -> None:
    """Download a ZIP to temporary storage and extract only into the disk mount."""
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "assets.zip"
        print("Downloading deployment assets to the persistent disk...", flush=True)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            if "processed/master_df.parquet" not in names or not any(name.startswith("models/") for name in names):
                raise RuntimeError("Asset bundle must contain processed/master_df.parquet and models/.")
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            destination = ASSETS_DIR.resolve()
            for member in bundle.infolist():
                # Prevent a malicious ZIP entry such as ../../outside-the-disk from escaping.
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise RuntimeError("Asset bundle contains an unsafe file path.")
            bundle.extractall(ASSETS_DIR)


def main() -> int:
    """Ensure assets are available before the web server starts."""
    if assets_ready():
        print("Deployment assets are already available on the persistent disk.", flush=True)
        return 0
    url = os.getenv("ASSET_BUNDLE_URL")
    if not url:
        print("Required data/models are missing. Set ASSET_BUNDLE_URL to a private ZIP bundle URL.", file=sys.stderr)
        return 1
    try:
        download_and_extract(url)
    except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"Asset bootstrap failed: {error}", file=sys.stderr)
        return 1
    if not assets_ready():
        missing_models = [filename for filename in REQUIRED_MODEL_FILES if not (MODELS_DIR / filename).exists()]
        message = "Asset bootstrap completed but required files are still missing."
        if missing_models:
            message += f" Missing models: {', '.join(missing_models)}"
        print(message, file=sys.stderr)
        return 1
    print("Deployment assets are ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
