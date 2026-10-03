# API Documentation

Base URL: `http://localhost:8000/api/v1`  
Interactive Swagger UI: `http://localhost:8000/docs`

## Authentication

The current local API has **no authentication layer**. Do not expose it publicly without adding authentication, rate limiting, TLS, and restricted CORS settings.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | API status and environment |
| GET | `/analytics/customer-metrics` | Platform KPIs |
| POST | `/predictions/churn` | Churn prediction |
| POST | `/predictions/clv` | CLV prediction |
| POST | `/predictions/delivery-delay` | Delivery risk baseline |
| GET | `/recommendations/{customer_id}` | Product recommendations |
| POST | `/sentiment` | Review sentiment |
| POST | `/forecast/demand` | Demand forecast points |
| POST | `/chat` | RAG chatbot |
| POST | `/agent/ask` | Bounded LangGraph orchestration across allowlisted commerce tools |
| GET | `/rag/metrics` | Privacy-safe RAG aggregate metrics |
| GET | `/metrics` | Prometheus exposition endpoint (outside `/api/v1`) |

### Agentic analyst

```http
POST /api/v1/agent/ask
Content-Type: application/json

{"question":"Did anything unusual happen to revenue or orders?"}
```

The response includes request ID, selected intent, tools used, safe evidence records, sources, errors, retries/fallback metadata in the trace, iteration count, latency, evaluation, anomalies, grounding status, and planner/selector modes. Evidence includes compact feature/prediction summaries and redacted input summaries; raw tool arguments are not returned. An optional `customer_id` scopes supported customer tools, and `review_text` enables sentiment analysis. The API is currently unauthenticated; do not expose customer-scoped calls publicly without authentication and authorization.

Execution uses a single supervisor that selects one allowlisted tool, validates and observes its result, then chooses the next action. It is not a multi-agent system. State and memory are request-scoped. Contact PII and a few common prompt-injection patterns are checked, but no toxicity/misuse classifier is installed. Ollama token usage and cost are unavailable and returned as `null`. Churn and CLV fallback results are explicitly marked heuristic/historical proxies.

The direct RAG Chatbot remains at `POST /api/v1/chat`; the agent may use registered RAG search as one capability in a broader request. See [Agentic AI Orchestration](11_Agentic_AI_Orchestration.md) for runtime flow, tool registry, and safety boundaries.

## Examples

### Health

```http
GET /api/v1/health
```

```json
{"status": "ok", "environment": "production"}
```

### Churn and CLV

```http
POST /api/v1/predictions/churn
Content-Type: application/json

{"features": {"Recency": 45, "Frequency": 2, "Monetary": 350.0}}
```

```json
{"prediction": 1, "probability": 0.72}
```

Use the same envelope for CLV:

```json
{"features": {"Average_Order_Value": 175.0, "Recency": 45, "Frequency": 2}}
```

### Delivery delay

```http
POST /api/v1/predictions/delivery-delay
Content-Type: application/json

{"order_purchase_date": "2026-07-01", "estimated_delivery_date": "2026-07-15", "customer_state": "SP"}
```

```json
{"predicted_delay_days": 1.1, "risk": "medium"}
```

### Recommendations

```http
GET /api/v1/recommendations/customer-123?limit=5
```

```json
{"customer_id": "customer-123", "product_ids": ["product-a", "product-b"]}
```

### Sentiment

```http
POST /api/v1/sentiment
Content-Type: application/json

{"review": "The delivery was fast and the product was excellent."}
```

```json
{"sentiment": "Positive"}
```

### Demand forecast

```http
POST /api/v1/forecast/demand
Content-Type: application/json

{"product_id": "product-a", "days": 7}
```

```json
[{"date": "2026-07-23", "predicted_demand": 18.4}]
```

### RAG chatbot

```http
POST /api/v1/chat
Content-Type: application/json

{"question": "What do reviews say about delivery?", "limit": 5}
```

```json
{
  "answer": "[Review] ...",
  "sources": ["Review", "Product"],
  "prompt_version": "2026-07-21.1",
  "knowledge_base_version": "a1b2c3d4e5f6",
  "evaluation": {"faithfulness": 0.91, "answer_relevance": 0.67, "context_relevance": 0.75}
}
```

## Error responses

| Status | Meaning | Example |
| --- | --- | --- |
| 422 | Request validation failed | Missing `question`, invalid date, or invalid range |
| 503 | Required dataset/model/retriever artifact missing or invalid | `{"detail": "Model artifact not found: ..."}` |
| 500 | Unexpected service failure | Production: generic error; development: additional diagnostics |

## Contract guidance

- Send JSON and `Content-Type: application/json` for POST endpoints.
- Keep prediction feature keys aligned with the artifact’s trained schema.
- Chat questions are limited to 2,000 characters and review text to 5,000 characters.
- Use `/metrics` only for Prometheus or trusted operators; it is not a customer-facing API.
