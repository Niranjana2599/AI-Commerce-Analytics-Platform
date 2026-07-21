"""Small reusable helpers for MLflow experiment tracking."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable

import mlflow
import mlflow.sklearn
import pandas as pd


def configure_tracking() -> None:
    """Use Docker MLflow when configured, otherwise store runs locally."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)


def git_commit_hash() -> str:
    """Return the current commit hash without failing outside a Git checkout."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def dataset_version(path: Path) -> str:
    """Create a lightweight, repeatable dataset identifier from file metadata."""
    stat = path.stat()
    value = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def log_run_context(data_path: Path, random_seed: int, feature_names: Iterable[str]) -> None:
    """Log metadata shared by every training run."""
    mlflow.log_params({
        "dataset_version": dataset_version(data_path),
        "random_seed": random_seed,
        "git_commit": git_commit_hash(),
        "feature_names": ",".join(feature_names),
    })


def log_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Save a small report as an MLflow artifact."""
    frame.to_csv(path, index=False)
    mlflow.log_artifact(str(path))
