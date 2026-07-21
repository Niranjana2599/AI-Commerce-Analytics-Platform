"""Versioned HTTP endpoints for all completed analytics modules."""

from fastapi import APIRouter, HTTPException, Query

from backend.app.core.config import settings
from backend.app.schemas.contracts import (
    ChatRequest, ChatResponse, CustomerMetricsResponse, DeliveryDelayRequest,
    DeliveryDelayResponse, ForecastPoint, ForecastRequest, HealthResponse,
    ModelPredictionRequest, PredictionResponse, RecommendationResponse,
    SentimentRequest, SentimentResponse,
)
from backend.app.services.commerce import (
    classify_sentiment, customer_metrics, demand_forecast, predict,
    predict_delivery_delay, recommend, retrieve,
)


router = APIRouter()


def service_error(error: Exception) -> HTTPException:
    """Convert expected data/artifact failures into a clear operational response."""
    if isinstance(error, (FileNotFoundError, ValueError, KeyError)):
        return HTTPException(status_code=503, detail=str(error))
    if settings.environment == "development":
        return HTTPException(status_code=500, detail=f"Model execution failed: {error}")
    return HTTPException(status_code=500, detail="The analytics service could not process this request.")


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", environment=settings.environment)


@router.get("/analytics/customer-metrics", response_model=CustomerMetricsResponse, tags=["Customer Analytics"])
def get_customer_metrics() -> CustomerMetricsResponse:
    try:
        return CustomerMetricsResponse(**customer_metrics())
    except Exception as error:
        raise service_error(error) from error


@router.post("/predictions/churn", response_model=PredictionResponse, tags=["Churn Prediction"])
def churn_prediction(request: ModelPredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**predict("churn", request.features))
    except Exception as error:
        raise service_error(error) from error


@router.post("/predictions/clv", response_model=PredictionResponse, tags=["CLV Prediction"])
def clv_prediction(request: ModelPredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**predict("clv", request.features))
    except Exception as error:
        raise service_error(error) from error


@router.post("/predictions/delivery-delay", response_model=DeliveryDelayResponse, tags=["Delivery Delay Prediction"])
def delivery_delay_prediction(request: DeliveryDelayRequest) -> DeliveryDelayResponse:
    return DeliveryDelayResponse(**predict_delivery_delay(request.order_purchase_date, request.estimated_delivery_date))


@router.get("/recommendations/{customer_id}", response_model=RecommendationResponse, tags=["Product Recommendations"])
def product_recommendations(customer_id: str, limit: int = Query(default=10, ge=1, le=50)) -> RecommendationResponse:
    try:
        return RecommendationResponse(customer_id=customer_id, product_ids=recommend(customer_id, limit))
    except Exception as error:
        raise service_error(error) from error


@router.post("/sentiment", response_model=SentimentResponse, tags=["Sentiment Analysis"])
def sentiment_analysis(request: SentimentRequest) -> SentimentResponse:
    try:
        return SentimentResponse(sentiment=classify_sentiment(request.review))
    except Exception as error:
        raise service_error(error) from error


@router.post("/forecast/demand", response_model=list[ForecastPoint], tags=["Demand Forecasting"])
def forecast_demand(request: ForecastRequest) -> list[ForecastPoint]:
    try:
        return [ForecastPoint(**item) for item in demand_forecast(request.product_id, request.days)]
    except Exception as error:
        raise service_error(error) from error


@router.post("/chat", response_model=ChatResponse, tags=["RAG Chatbot"])
def rag_chat(request: ChatRequest) -> ChatResponse:
    try:
        answer, sources = retrieve(request.question, request.limit)
        return ChatResponse(answer=answer, sources=sources)
    except Exception as error:
        raise service_error(error) from error
