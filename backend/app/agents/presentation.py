"""Build privacy-conscious API evidence for the Streamlit analyst view."""

from typing import Any

def public_output(tool: str, output: dict[str, Any] | None) -> dict[str, Any] | None:
    """Expose only concise evidence fields needed to understand a tool result."""
    if not isinstance(output, dict):
        return None
    allowed = {
        "customer_cohort_tool": ("segment", "count", "cohort_definition"),
        "customer_rfm_tool": ("count", "note"),
        "business_analytics_tool": ("metric", "value", "rows"),
        "anomaly_detection_tool": ("metric", "method", "baseline_median", "observations", "flags", "status"),
        "demand_forecast_tool": ("forecast",),
        "sentiment_tool": ("sentiment",),
        "rag_search": ("documents", "sources"),
        "recommendation_tool": ("recommendations",),
        "churn_prediction_tool": ("customers", "model"),
        "clv_prediction_tool": ("customers", "model"),
        "rfm_risk_tool": ("customers", "method", "is_model_prediction"),
        "historical_value_tool": ("customers", "method", "is_model_prediction"),
    }.get(tool, ())
    safe = {key: output[key] for key in allowed if key in output}
    # Keep row-level customer data out of routine execution traces.
    if "customers" in safe:
        safe["customer_count"] = len(safe.pop("customers"))
    if "recommendations" in safe:
        rows = safe.pop("recommendations")
        safe["recommendation_count"] = len(rows)
        safe["product_ids"] = list(dict.fromkeys(
            product for row in rows if isinstance(row, dict)
            for product in row.get("product_ids", []) if isinstance(product, str)
        ))[:10]
    if "documents" in safe:
        safe["document_character_count"] = len(str(safe.pop("documents")))
    return safe


def public_input_summary(inputs: dict[str, Any]) -> dict[str, Any]:
    """Describe tool inputs without returning identifiers, queries, or review text."""
    summary = {key: value for key, value in inputs.items() if key in {"metric", "segment", "limit", "days", "k"}}
    for key in ("customers", "customer_ids"):
        if key in inputs:
            summary[f"{key}_count"] = len(inputs[key])
    if "features" in inputs:
        summary["feature_names"] = sorted(inputs["features"].keys())
    for private in ("customer_id", "review", "query"):
        if private in inputs:
            summary[f"{private}_redacted"] = True
    return summary
