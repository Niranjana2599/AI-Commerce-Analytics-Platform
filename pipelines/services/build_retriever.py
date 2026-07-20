"""Build the persisted TF-IDF retriever used by the RAG analyst.

Run with: ``python -m pipelines.services.build_retriever``.
"""

import logging

import pandas as pd

from src.config import MODELS_DIR
from src.data import load_master_data
from src.services.retrieval import EcommerceRetriever


def build_documents(master_df: pd.DataFrame) -> pd.DataFrame:
    """Create product, review, and metric documents from prepared commerce data."""
    products = master_df.groupby("product_id", as_index=False).agg(
        category=("product_category_name", "first"), seller=("seller_id", "first"), price=("payment_value", "mean")
    )
    products["text"] = products.apply(
        lambda row: f"Product {row.product_id} belongs to {row.category}, is sold by {row.seller}, and has average price {row.price:.2f}.",
        axis=1,
    )
    products["source"] = "Product"

    reviews = master_df.dropna(subset=["review_comment_message"])[["product_id", "seller_id", "review_score", "review_comment_message"]]
    reviews = reviews.drop_duplicates("review_comment_message").head(5_000).copy()
    reviews["text"] = reviews.apply(
        lambda row: f"Review for product {row.product_id} from seller {row.seller_id}. Rating: {row.review_score}. Review: {row.review_comment_message}",
        axis=1,
    )
    reviews["source"] = "Review"
    metrics = pd.DataFrame({"source": ["Dashboard", "Dashboard"], "text": [
        f"Dashboard metric: total revenue is {master_df.payment_value.sum():.2f}.",
        f"Dashboard metric: total orders are {master_df.order_id.nunique()}.",
    ]})
    return pd.concat([products[["source", "text"]], reviews[["source", "text"]], metrics], ignore_index=True)


def run() -> None:
    """Index ecommerce documents and save a retrieval artifact."""
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    retriever = EcommerceRetriever().fit(build_documents(load_master_data()))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "rag_retriever.joblib"
    joblib.dump(retriever, output_path)
    logging.info("Saved retriever with %s documents to %s", f"{len(retriever.documents):,}", output_path)


if __name__ == "__main__":
    run()
