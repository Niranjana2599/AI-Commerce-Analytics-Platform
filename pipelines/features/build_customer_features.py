"""Generate customer features for churn and CLV workflows.

Run with: ``python -m pipelines.features.build_customer_features``.
"""

import logging

from src.config import PROCESSED_DATA_DIR
from src.data import load_master_data
from src.features import build_customer_features


def run() -> None:
    """Create and persist the customer-level feature table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    features = build_customer_features(load_master_data())
    output_path = PROCESSED_DATA_DIR / "customer_features.parquet"
    features.to_parquet(output_path, index=False)
    logging.info("Saved %s customer feature rows to %s", f"{len(features):,}", output_path)


if __name__ == "__main__":
    run()
