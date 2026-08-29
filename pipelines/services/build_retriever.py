"""
Build and persist the FAISS vector store used by the RAG analyst.

Knowledge base:
    - 100,000 customer review documents
    - 2,000 product documents
    - Policy / FAQ documents
    - Dashboard metric documents

Run with:
    python -m pipelines.services.build_retriever
"""

import logging
from pathlib import Path

import pandas as pd

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import MODELS_DIR
from src.data import load_master_data


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_REVIEW_DOCUMENTS = 100_000

BATCH_SIZE = 256

VECTORSTORE_DIR = MODELS_DIR / "faiss_ecommerce"


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Product documents
# ------------------------------------------------------------

def build_product_documents(master_df: pd.DataFrame) -> list[Document]:
    """
    Create one document per product.
    """

    required_columns = {
        "product_id",
        "product_category_name",
        "seller_id",
        "payment_value",
    }

    missing = required_columns - set(master_df.columns)

    if missing:
        raise KeyError(
            f"Missing product columns: {sorted(missing)}"
        )

    products = (
        master_df
        .groupby("product_id", as_index=False)
        .agg(
            category=("product_category_name", "first"),
            seller=("seller_id", "first"),
            average_price=("payment_value", "mean"),
        )
    )

    documents = []

    for row in products.itertuples(index=False):

        documents.append(
            Document(
                page_content=(
                    f"Product {row.product_id}.\n"
                    f"Category: {row.category}.\n"
                    f"Seller: {row.seller}.\n"
                    f"Average price: {row.average_price:.2f}."
                ),
                metadata={
                    "source": "products",
                    "product_id": str(row.product_id),
                    "seller_id": str(row.seller),
                },
            )
        )

    logger.info(
        "Product documents: %s",
        f"{len(documents):,}",
    )

    return documents


# ------------------------------------------------------------
# Customer review documents
# ------------------------------------------------------------

def build_review_documents(
    master_df: pd.DataFrame,
) -> list[Document]:
    """
    Create up to 100,000 representative review-level documents.

    One customer review = one LangChain Document.

    Reviews are sampled proportionally across review scores
    so that negative, neutral and positive feedback are retained.
    """

    required_columns = {
        "product_id",
        "seller_id",
        "review_score",
        "review_comment_message",
    }

    missing = required_columns - set(master_df.columns)

    if missing:
        raise KeyError(
            f"Missing review columns: {sorted(missing)}"
        )

    reviews = (
        master_df[
            [
                "product_id",
                "seller_id",
                "review_score",
                "review_comment_message",
            ]
        ]
        .dropna(
            subset=[
                "product_id",
                "review_comment_message",
            ]
        )
        .copy()
    )

    # Clean review text.
    reviews["review_comment_message"] = (
        reviews["review_comment_message"]
        .astype(str)
        .str.strip()
    )

    reviews = reviews[
        reviews["review_comment_message"].ne("")
    ].copy()

    # Remove exact duplicate review text for the same product.
    reviews = reviews.drop_duplicates(
        subset=[
            "product_id",
            "review_comment_message",
        ]
    )

    logger.info(
        "Usable review rows: %s",
        f"{len(reviews):,}",
    )

    # --------------------------------------------------------
    # Select representative reviews
    # --------------------------------------------------------

    if len(reviews) <= TARGET_REVIEW_DOCUMENTS:

        selected_reviews = reviews.reset_index(drop=True)

    else:

        sampled_parts = []

        for score, group in reviews.groupby(
            "review_score",
            observed=True
        ):

            target_count = round(
                TARGET_REVIEW_DOCUMENTS
                * len(group)
                / len(reviews)
            )

            target_count = max(
                1,
                target_count,
            )

            target_count = min(
                target_count,
                len(group),
            )

            sampled_parts.append(
                group.sample(
                    n=target_count,
                    random_state=42,
                )
            )

        selected_reviews = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        # Guarantee exact target size.
        if len(selected_reviews) > TARGET_REVIEW_DOCUMENTS:

            selected_reviews = (
                selected_reviews
                .sample(
                    n=TARGET_REVIEW_DOCUMENTS,
                    random_state=42,
                )
                .reset_index(drop=True)
            )

        elif len(selected_reviews) < TARGET_REVIEW_DOCUMENTS:

            # Sample additional records from the unused pool.
            selected_ids = set(
                selected_reviews.index
            )

            remaining = reviews[
                ~reviews.index.isin(selected_ids)
            ]

            additional_count = min(
                TARGET_REVIEW_DOCUMENTS
                - len(selected_reviews),
                len(remaining),
            )

            if additional_count > 0:

                additional = remaining.sample(
                    n=additional_count,
                    random_state=42,
                )

                selected_reviews = pd.concat(
                    [
                        selected_reviews,
                        additional,
                    ],
                    ignore_index=True,
                )

    logger.info(
        "Selected review records: %s",
        f"{len(selected_reviews):,}",
    )

    # --------------------------------------------------------
    # One review = one LangChain Document
    # --------------------------------------------------------

    documents = []

    for row in selected_reviews.itertuples(index=False):

        documents.append(
            Document(
                page_content=(
                    f"Customer review for product "
                    f"{row.product_id}.\n"
                    f"Seller: {row.seller_id}.\n"
                    f"Review score: {row.review_score}.\n"
                    f"Customer comment: "
                    f"{row.review_comment_message}"
                ),
                metadata={
                    "source": "customer_reviews",
                    "product_id": str(row.product_id),
                    "seller_id": str(row.seller_id),
                    "review_score": float(
                        row.review_score
                    ),
                },
            )
        )

    logger.info(
        "Review documents: %s",
        f"{len(documents):,}",
    )

    return documents


# ------------------------------------------------------------
# Policy / FAQ documents
# ------------------------------------------------------------

def build_policy_documents() -> list[Document]:
    """
    Build business policy and FAQ knowledge.
    """

    policy_texts = [

        "FAQ: Delivery dates depend on seller processing time and customer location.",

        "FAQ: Delivery performance can vary by seller and destination.",

        "Policy: Freight value represents the delivery-related charge associated with an order item.",

        "FAQ: Customers should provide order information when reporting delivery issues.",

        "FAQ: Customers can review products after an order is completed.",

        "FAQ: Order-specific questions should be answered using the relevant order and product information.",

        "Policy: Product availability can vary by seller.",

        "FAQ: Product information should be interpreted together with its category and seller information.",

        "Policy: Product characteristics and availability may vary between sellers.",

        "FAQ: Review scores represent the customer's submitted rating for the product or order experience.",

        "FAQ: Customer comments provide qualitative feedback about the purchasing experience.",

        "FAQ: Payment method identifies the payment option used for an order.",

        "FAQ: Payment-related analysis can be performed using the payment method associated with orders.",

        "FAQ: Seller performance can be evaluated using delivery performance, customer reviews and order activity.",

        "Policy: Delivery performance can vary across sellers.",

        "FAQ: Customers should provide relevant order or product information when asking order-specific questions.",
    ]

    documents = [
        Document(
            page_content=text,
            metadata={
                "source": "policies_faq",
                "type": "policy_faq",
            },
        )
        for text in policy_texts
    ]

    logger.info(
        "Policy/FAQ documents: %s",
        len(documents),
    )

    return documents


# ------------------------------------------------------------
# Dashboard metric documents
# ------------------------------------------------------------

def build_metric_documents(
    master_df: pd.DataFrame,
) -> list[Document]:
    """
    Create dashboard-level metric documents.
    """

    required_columns = {
        "payment_value",
        "order_id",
    }

    missing = required_columns - set(master_df.columns)

    if missing:
        raise KeyError(
            f"Missing metric columns: {sorted(missing)}"
        )

    total_revenue = master_df[
        "payment_value"
    ].sum()

    total_orders = master_df[
        "order_id"
    ].nunique()

    metrics = [
        (
            f"Dashboard metric: total revenue is "
            f"{total_revenue:.2f}."
        ),
        (
            f"Dashboard metric: total orders are "
            f"{total_orders:,}."
        ),
        (
            f"Dashboard metric: average order value is "
            f"{master_df['payment_value'].mean():.2f}."
        ),
        (
            f"Dashboard metric: total transaction rows are "
            f"{len(master_df):,}."
        ),
    ]

    documents = [
        Document(
            page_content=text,
            metadata={
                "source": "dashboard_metrics",
                "type": "metric",
            },
        )
        for text in metrics
    ]

    logger.info(
        "Metric documents: %s",
        len(documents),
    )

    return documents


# ------------------------------------------------------------
# Build complete knowledge base
# ------------------------------------------------------------

def build_documents(
    master_df: pd.DataFrame,
) -> list[Document]:
    """
    Build the complete LangChain knowledge base.
    """

    product_documents = build_product_documents(
        master_df
    )

    review_documents = build_review_documents(
        master_df
    )

    policy_documents = build_policy_documents()

    metric_documents = build_metric_documents(
        master_df
    )

    all_documents = (
        product_documents
        + review_documents
        + policy_documents
        + metric_documents
    )

    logger.info(
        "Total curated documents: %s",
        f"{len(all_documents):,}",
    )

    return all_documents


# ------------------------------------------------------------
# Build FAISS incrementally
# ------------------------------------------------------------

def build_faiss(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Embed documents in batches and construct the FAISS index.
    """

    if not documents:
        raise ValueError(
            "No documents were created."
        )

    total_documents = len(documents)

    vectorstore = None

    progress_interval = max(
        BATCH_SIZE,
        total_documents // 20,
    )

    next_progress = progress_interval

    logger.info(
        "Documents waiting for embedding: %s",
        f"{total_documents:,}",
    )

    for start in range(
        0,
        total_documents,
        BATCH_SIZE,
    ):

        batch = documents[
            start:start + BATCH_SIZE
        ]

        if vectorstore is None:

            vectorstore = FAISS.from_documents(
                documents=batch,
                embedding=embeddings,
            )

        else:

            vectorstore.add_documents(
                batch
            )

        processed = min(
            start + BATCH_SIZE,
            total_documents,
        )

        if (
            processed >= next_progress
            or processed == total_documents
        ):

            logger.info(
                "Embedded %s/%s chunks (%0.0f%%)",
                f"{processed:,}",
                f"{total_documents:,}",
                processed / total_documents * 100,
            )

            next_progress += progress_interval

    return vectorstore


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def run() -> None:

    logger.info(
        "Loading master commerce data..."
    )

    master_df = load_master_data()

    logger.info(
        "Master dataframe rows: %s",
        f"{len(master_df):,}",
    )

    # --------------------------------------------------------
    # Build documents
    # --------------------------------------------------------

    documents = build_documents(
        master_df
    )

    # --------------------------------------------------------
    # Initialize embedding model
    # --------------------------------------------------------

    logger.info(
        "Embedding model: %s",
        EMBEDDING_MODEL,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    vectorstore = build_faiss(
        documents,
        embeddings,
    )

    # --------------------------------------------------------
    # Persist FAISS index
    # --------------------------------------------------------

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    logger.info(
        "FAISS vector store created successfully."
    )

    logger.info(
        "Indexed documents: %s",
        f"{len(documents):,}",
    )

    logger.info(
        "Saved FAISS index to: %s",
        VECTORSTORE_DIR,
    )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    run()