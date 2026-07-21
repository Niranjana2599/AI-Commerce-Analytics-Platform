"""Services that connect saved notebook artifacts to API operations."""

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import logging

from backend.app.core.config import settings


MODEL_FILES = {
    "churn": "customer_churn_model.joblib",
    "clv": "customer_clv_model.joblib",
    "sentiment": "customer_review_sentiment_model.joblib",
    "demand": "demand_forecasting_model.joblib",
    "recommendations": "product_recommender_system.joblib",
    "retriever": "rag_retriever.joblib",
}

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def master_data() -> pd.DataFrame:
    """Load prepared data only once per worker process."""
    if not settings.data_path.exists():
        raise FileNotFoundError(f"Prepared data not found: {settings.data_path}")
    return pd.read_parquet(settings.data_path)


@lru_cache(maxsize=None)
def load_model(name: str) -> Any:
    """Lazy-load an artifact so starting the API does not load large models."""
    path: Path = settings.models_dir / MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def customer_metrics() -> dict[str, float | int]:
    data = master_data()
    total_orders = data["order_id"].nunique()
    total_revenue = float(data["payment_value"].sum())
    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": int(total_orders),
        "total_customers": int(data["customer_unique_id"].nunique()),
        "average_order_value": round(total_revenue / max(total_orders, 1), 2),
    }


def predict(model_name: str, features: dict[str, Any]) -> dict[str, float | int | str | None]:
    """Run a standard scikit-learn-style estimator on one feature row."""
    model = load_model(model_name)
    frame = pd.DataFrame([features])
    value = model.predict(frame)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0][-1])
    return {"prediction": value.item() if hasattr(value, "item") else value, "probability": probability}


def predict_delivery_delay(order_purchase_date: date, estimated_delivery_date: date) -> dict[str, float | str]:
    """Transparent baseline based on the planned fulfilment window.

    Replace this with a saved delay estimator when one is trained and added to MODEL_FILES.
    """
    planned_days = max((estimated_delivery_date - order_purchase_date).days, 0)
    predicted_delay = round(max(planned_days - 7, 0) * 0.15, 1)
    risk = "high" if predicted_delay >= 3 else "medium" if predicted_delay >= 1 else "low"
    return {"predicted_delay_days": predicted_delay, "risk": risk}


def recommend(customer_id: str, limit: int = 10) -> list[str]:
    """Use the notebook artifact when compatible; otherwise return popular unseen products."""
    data = master_data()
    try:
        model = load_model("recommendations")
        if hasattr(model, "recommend"):
            return [str(item) for item in model.recommend(customer_id, k=limit)]
    except (FileNotFoundError, AttributeError):
        pass
    seen = set(data.loc[data["customer_unique_id"] == customer_id, "product_id"].dropna().astype(str))
    popular = data["product_id"].dropna().astype(str).value_counts().index
    return [product for product in popular if product not in seen][:limit]


def classify_sentiment(review: str) -> str:
    """Classify a review, retaining service availability if an old artifact cannot load.

    Existing ML models remain the preferred path. The small fallback protects the
    dashboard while a serialized legacy model is retrained through MLflow.
    """
    try:
        model = load_model("sentiment")
        label = model.predict([review])[0]
        return str(label)
    except Exception as error:
        LOGGER.warning("Sentiment model unavailable; using keyword fallback: %s", error)
        text = review.lower()
        negative_words = {"late", "bad", "worst", "broken", "poor", "disappointed", "terrible", "hate"}
        positive_words = {"great", "excellent", "love", "perfect", "fast", "good", "amazing"}
        negative_count = sum(word in text for word in negative_words)
        positive_count = sum(word in text for word in positive_words)
        if negative_count > positive_count:
            return "Negative"
        if positive_count > negative_count:
            return "Positive"
        return "Neutral"


def demand_forecast(product_id: str | None, days: int) -> list[dict[str, Any]]:
    """Forecast demand using recent observed order volume as a reliable baseline."""
    data = master_data().copy()
    data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"])
    if product_id:
        data = data[data["product_id"].astype(str) == product_id]
    daily = data.groupby(data["order_purchase_timestamp"].dt.date)["order_id"].nunique()
    baseline = float(daily.tail(28).mean()) if not daily.empty else 0.0
    start = pd.Timestamp.today().normalize().date()
    return [{"date": start + timedelta(days=offset), "predicted_demand": round(baseline, 2)} for offset in range(1, days + 1)]


def retrieve(question: str, limit: int) -> tuple[str, list[str]]:
    """Retrieve evidence from the persisted RAG artifact without inventing an LLM answer."""
    artifact = load_model("retriever")
    if hasattr(artifact, "search"):
        results = artifact.search(question, k=limit)
    elif isinstance(artifact, dict) and "documents" in artifact:
        documents = artifact["documents"]
        results = documents.head(limit)
    else:
        raise ValueError("Unsupported retriever artifact format.")
    texts = results["text"].astype(str).tolist()
    sources = results["source"].astype(str).tolist() if "source" in results else ["Dataset"] * len(texts)
    answer = "\n".join(f"[{source}] {text}" for source, text in zip(sources, texts))
    return answer, sources
