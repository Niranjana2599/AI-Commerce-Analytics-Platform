"""Train and persist the baseline product recommendation service.

Run with: ``python -m pipelines.services.build_recommender``.
"""

import logging

from src.config import MODELS_DIR
from src.data import load_master_data
from src.services.recommendations import ProductRecommender


def run() -> None:
    """Fit the popularity-based recommender on completed customer interactions."""
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    interactions = load_master_data()[["customer_unique_id", "product_id"]]
    recommender = ProductRecommender().fit(interactions)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "product_recommender.joblib"
    joblib.dump(recommender, output_path)
    logging.info("Saved recommender with %s products to %s", f"{len(recommender.popular_products_):,}", output_path)


if __name__ == "__main__":
    run()
