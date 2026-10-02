# Agentic AI Orchestration

The Agentic Commerce Analyst is one bounded LangGraph supervisor over existing FastAPI commerce services. It selects one tool, validates and observes its actual result, then decides whether another tool is needed. Models, analytics, recommendation, forecasting, sentiment, anomaly detection, and RAG remain services/tools. The optional LLM proposes request goals and can choose only among host-approved next actions; deterministic selection remains the fallback.

```text
Streamlit → POST /api/v1/agent/ask → input guardrail → LangGraph state
                                                   ↓
                                    request understanding (Ollama or rules fallback)
                                                   ↓
                           allowlisted tool registry → service results
                                                   ↓
                               validation / retry / explicit fallback
                                                   ↓
                                   grounded answer + trace + metrics
```

## Code layout

- `backend/app/agents/`: request-scoped typed state, deterministic router, optional Ollama goal planner/next-tool selector, LangGraph supervisor.
- `backend/app/tools/`: bounded adapters around the existing commerce service layer, plus a central registry.
- `backend/app/guardrails/`: basic prompt-injection screening and contact-detail redaction.
- `backend/app/evaluation/`: shared golden requests and runtime outcome/selection metrics for V1 deterministic planner, V2 schema-checked LLM planner with rules selector, and V3 result-aware loop with LLM planner/selector.
- `backend/app/api/agent.py`: `POST /api/v1/agent/ask`.
- `streamlit/pages/13_Agentic_Analyst.py`: UI for questions, optional customer ID/review, safe evidence trace, evaluation, and fallback visibility.

## Runtime and safety behavior

The planner and selector use `AGENT_PLANNER_ENABLED` and the existing Ollama model setting. Invalid responses or Ollama failures fall back to deterministic routing. The graph has an eight-attempt ceiling (retries count toward it); retries are bounded and invalid-input failures are not retried. The registry rejects unknown tools and validates input/output schemas plus selected domain constraints. Input screening covers a few injection patterns and contact PII; these checks are demonstrations, not a complete security boundary. Toxicity and misuse classification are not implemented.

The LLM planner is optional and should be benchmarked only where Ollama is reachable. V1/V2/V3 share the same golden requests and execution metrics; if Ollama is unavailable, the response records deterministic fallback modes instead of claiming an LLM comparison.

Customer cohort churn/CLV predictions require the saved estimator to expose `feature_names_in_` that are present in the observed engineered customer rows. Missing model fields are never guessed. Churn failure can execute a real registered recency-risk proxy; CLV failure can execute a historical-spend ranking. Both are labeled as proxies, not model predictions. No model is retrained by the agent.

Memory exists only in the current request state. There is no persistent or cross-session memory and no autonomous specialist-agent team. RAG documents are treated as untrusted data; selectors receive only compact summaries, not retrieved document bodies. RAG source metadata is returned with the answer. Ollama token usage and cost are unavailable from the current client path and are returned as `null`.

Agent metrics use low-cardinality labels and are exposed by Prometheus at `/metrics`: request outcomes, tool calls/failures/latency, retry/fallback counts, iteration counts, LLM latency, and anomaly counts. Logs include request ID, intent, tool names, and latency; they do not include full question text or model feature values. The API currently has no authentication.

## Local operation

Install `backend/requirements.txt`, ensure Ollama is available if LLM planning/selection is enabled, and start the normal stack with `docker compose up --build`. Disable both LLM calls for deterministic-only operation with `AGENT_PLANNER_ENABLED=false`. The Streamlit page is **Agentic Analyst** and the OpenAPI endpoint is `POST /api/v1/agent/ask`.
