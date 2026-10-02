"""Explicit per-request state for the observe → decide → act graph."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    request_id: str
    user_query: str
    customer_id: str | None
    review_text: str | None
    messages: list[dict[str, str]]
    intent: str
    objectives: list[str]
    plan: dict[str, Any]
    current_tool: str
    current_action: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    decision: str
    decision_summary: str
    next_action: dict[str, Any] | None
    iteration_count: int
    retry_count: int
    attempt_count: int
    fallback_count: int
    memory_context: dict[str, Any]
    guardrail_result: dict[str, Any]
    errors: list[str]
    anomalies: list[dict[str, Any]]
    sources: list[str]
    final_response: dict[str, Any]
    evaluation: dict[str, Any]
    trace: list[dict[str, Any]]
    max_iterations: int
    max_retries: int
    tool_valid: bool
    fallback_used: bool
    fallback_for: dict[str, Any] | None
    seen_signatures: dict[str, int]
    planner_mode: str
    selector_mode: str
    selector_strategy: str
    llm_calls: int
    llm_latency_ms: float
    latency_ms: float
