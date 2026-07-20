"""Consistent data access for notebooks, scripts, and applications."""

from pathlib import Path

import pandas as pd

from src.config import MASTER_DATA_PATH, RAW_DATA_DIR


def load_raw_datasets(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load every raw CSV file, keyed by its filename stem."""
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")
    return {path.stem: pd.read_csv(path) for path in paths}


def load_master_data(path: Path = MASTER_DATA_PATH) -> pd.DataFrame:
    """Load the prepared master dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Prepared data was not found at {path}.")
    return pd.read_parquet(path)
