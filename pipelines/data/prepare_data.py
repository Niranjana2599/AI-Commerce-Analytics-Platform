"""Build cleaned source tables and the master analytics dataset.

Run with: ``python -m pipelines.data.prepare_data``.
"""

import logging

from src.config import PROCESSED_DATA_DIR
from src.data import build_master_dataset, clean_datasets, load_raw_datasets


def run() -> None:
    """Clean raw CSVs and persist prepared tables for downstream workflows."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = clean_datasets(load_raw_datasets())
    for name, dataset in datasets.items():
        output_path = PROCESSED_DATA_DIR / f"{name}.csv"
        dataset.to_csv(output_path, index=False)
        logging.info("Saved %s (%s rows)", output_path.name, f"{len(dataset):,}")

    master_df = build_master_dataset(datasets)
    master_path = PROCESSED_DATA_DIR / "master_df.parquet"
    master_df.to_parquet(master_path, index=False)
    logging.info("Saved %s (%s rows)", master_path.name, f"{len(master_df):,}")


if __name__ == "__main__":
    run()
