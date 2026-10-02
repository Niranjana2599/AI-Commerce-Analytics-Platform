"""Read-only tool policy enforced before plan execution."""

from typing import Any

READ_ONLY_TOOLS = frozenset({
    "rag_search", "customer_cohort_tool", "customer_rfm_tool", "churn_prediction_tool", "clv_prediction_tool",
    "recommendation_tool", "demand_forecast_tool", "sentiment_tool",
    "business_analytics_tool", "anomaly_detection_tool",
})


def validate_plan(plan: list[dict[str, Any]]) -> None:
    """Reject non-object tasks, write-capable tools, and oversized plans."""
    if len(plan) > 6:
        raise ValueError("Agent plans may contain at most six tasks")
    for task in plan:
        if not isinstance(task, dict) or task.get("tool") not in READ_ONLY_TOOLS:
            raise ValueError("Agent plan contains a disallowed tool")
        if not isinstance(task.get("input", {}), dict):
            raise ValueError("Agent tool inputs must be an object")
