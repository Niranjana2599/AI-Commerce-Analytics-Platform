"""Prometheus metric definitions and small recording helpers.

Only stable, low-cardinality labels are used. Never add user IDs, questions, or
full URLs as labels because each distinct value creates a new time series.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests handled by FastAPI.",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "FastAPI request duration in seconds.",
    ["method", "endpoint"],
)
HTTP_ERRORS = Counter(
    "http_errors_total",
    "FastAPI responses with a 5xx status code.",
    ["method", "endpoint"],
)
ACTIVE_REQUESTS = Gauge("active_requests", "Requests currently being handled.")

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Prediction attempts grouped by model and outcome.",
    ["model_name", "outcome"],
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Prediction inference failures grouped by model.",
    ["model_name"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction inference duration in seconds.",
    ["model_name"],
)
MODEL_LOADING_TIME = Histogram(
    "model_loading_seconds",
    "Time required to deserialize a model artifact.",
    ["model_name"],
)
MODEL_LOADING_ERRORS = Counter(
    "model_loading_errors_total",
    "Model artifact loading failures grouped by model.",
    ["model_name"],
)
LOADED_MODELS = Gauge("loaded_models", "Whether a model artifact is loaded in this API process.", ["model_name"])

CHATBOT_REQUESTS = Counter(
    "chatbot_requests_total",
    "RAG chatbot requests grouped by outcome.",
    ["outcome"],
)
CHATBOT_ERRORS = Counter("chatbot_errors_total", "Failed RAG chatbot requests.")
RAG_LATENCY = Histogram("rag_latency_seconds", "Total RAG chatbot response duration in seconds.")
RAG_RETRIEVER_LATENCY = Histogram("rag_retriever_latency_seconds", "Retriever duration in seconds.")
RAG_LLM_LATENCY = Histogram("rag_llm_latency_seconds", "LLM generation or fallback duration in seconds.")
RAG_RETRIEVED_DOCUMENTS = Histogram(
    "rag_retrieved_documents",
    "Number of documents retrieved for each RAG request.",
    buckets=(0, 1, 2, 3, 5, 10),
)


def observe_model_load(model_name: str, duration_seconds: float, success: bool) -> None:
    """Record model deserialization timing and current loaded state."""
    MODEL_LOADING_TIME.labels(model_name).observe(duration_seconds)
    if success:
        LOADED_MODELS.labels(model_name).set(1)
    else:
        MODEL_LOADING_ERRORS.labels(model_name).inc()


def observe_prediction(model_name: str, duration_seconds: float, success: bool) -> None:
    """Record one model inference outcome without exposing user feature values."""
    outcome = "success" if success else "error"
    PREDICTION_REQUESTS.labels(model_name, outcome).inc()
    PREDICTION_LATENCY.labels(model_name).observe(duration_seconds)
    if not success:
        PREDICTION_ERRORS.labels(model_name).inc()


def observe_rag(duration_seconds: float, retrieval_seconds: float, llm_seconds: float, document_count: int, success: bool) -> None:
    """Record one RAG outcome and its bounded operational measurements."""
    outcome = "success" if success else "error"
    CHATBOT_REQUESTS.labels(outcome).inc()
    RAG_LATENCY.observe(duration_seconds)
    if success:
        RAG_RETRIEVER_LATENCY.observe(retrieval_seconds)
        RAG_LLM_LATENCY.observe(llm_seconds)
        RAG_RETRIEVED_DOCUMENTS.observe(document_count)
    else:
        CHATBOT_ERRORS.inc()
