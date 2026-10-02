"""Low-cardinality counters and latency for the agent endpoint."""

from prometheus_client import Counter, Histogram

AGENT_REQUESTS = Counter("agent_requests_total", "Agent requests by outcome", ["outcome"])
AGENT_TOOL_CALLS = Counter("agent_tool_calls_total", "Agent tool calls by allowlisted tool", ["tool"])
AGENT_TOOL_FAILURES = Counter("agent_tool_failures_total", "Agent tool failures by allowlisted tool", ["tool"])
AGENT_TOOL_LATENCY = Histogram("agent_tool_latency_seconds", "Agent tool execution duration", ["tool"])
AGENT_RETRIES = Counter("agent_tool_retries_total", "Agent tool retry attempts")
AGENT_FALLBACKS = Counter("agent_tool_fallbacks_total", "Agent tool fallbacks")
AGENT_LATENCY = Histogram("agent_request_latency_seconds", "End-to-end agent request latency")
AGENT_LLM_LATENCY = Histogram("agent_llm_call_latency_seconds", "Local LLM planner or selector latency")
AGENT_ITERATIONS = Histogram("agent_iterations", "Tool execution attempts per agent request")
AGENT_ANOMALIES = Counter("agent_anomalies_total", "Agent execution anomalies", ["anomaly_type"])


def observe_agent_run(state: dict, latency_ms: float) -> None:
    history = state.get("tool_calls", [])
    success=bool(state.get("evaluation",{}).get("task_success"))
    outcome="blocked" if state.get("guardrail_result",{}).get("status")=="blocked" else "success" if success else "partial_or_failed"
    AGENT_REQUESTS.labels(outcome).inc()
    AGENT_LATENCY.observe(latency_ms / 1_000)
    AGENT_ITERATIONS.observe(state.get("iteration_count",0))
    if state.get("llm_calls",0): AGENT_LLM_LATENCY.observe(state.get("llm_latency_ms",0)/1_000)
    for item in history:
        AGENT_TOOL_CALLS.labels(item["tool"]).inc()
        AGENT_TOOL_LATENCY.labels(item["tool"]).observe(item.get("elapsed_ms",0)/1_000)
        if item.get("status")!="success": AGENT_TOOL_FAILURES.labels(item["tool"]).inc()
        if item.get("fallback_used") and item.get("attempt")==1: AGENT_FALLBACKS.inc()
    retries=state.get("retry_count",0)
    if retries: AGENT_RETRIES.inc(retries)
    for anomaly in state.get("anomalies",[]): AGENT_ANOMALIES.labels(anomaly["anomaly_type"]).inc()
