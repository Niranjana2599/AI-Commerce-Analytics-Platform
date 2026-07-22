"""Pydantic schemas exposed by the API."""

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    environment: str


class CustomerMetricsResponse(BaseModel):
    total_revenue: float
    total_orders: int
    total_customers: int
    average_order_value: float


class ModelPredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature names and values expected by the trained model.")


class PredictionResponse(BaseModel):
    prediction: float | int | str
    probability: float | None = None


class DeliveryDelayRequest(BaseModel):
    estimated_delivery_date: date
    order_purchase_date: date
    customer_state: str | None = None


class DeliveryDelayResponse(BaseModel):
    predicted_delay_days: float
    risk: str


class RecommendationResponse(BaseModel):
    customer_id: str
    product_ids: list[str]


class SentimentRequest(BaseModel):
    review: str = Field(..., min_length=1, max_length=5_000)


class SentimentResponse(BaseModel):
    sentiment: str


class ForecastRequest(BaseModel):
    product_id: str | None = None
    days: int = Field(default=7, ge=1, le=90)


class ForecastPoint(BaseModel):
    date: date
    predicted_demand: float


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    prompt_version: str
    knowledge_base_version: str
    evaluation: dict[str, float]


class RAGMetricsResponse(BaseModel):
    query_count: int
    averages: dict[str, float]
    latency_ms: dict[str, float]
    prompt_versions: dict[str, int]
    knowledge_base_versions: dict[str, int]
