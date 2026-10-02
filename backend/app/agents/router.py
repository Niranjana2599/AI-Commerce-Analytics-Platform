"""Rule-based intent and next-action fallback used when local LLM planning fails."""

from typing import Any

GOAL_NAMES = {"cohort", "rfm", "clv", "churn", "recommendation", "analytics", "forecast", "sentiment", "rag", "anomaly"}


def infer_request(question: str, customer_id: str | None = None) -> tuple[str, list[str]]:
    q = question.lower()
    goals: list[str] = []
    if any(word in q for word in ("anomaly", "unusual", "drop", "spike", "suddenly")):
        goals += ["analytics", "anomaly"]
        if "why" in q or "explain" in q:
            goals.append("rag")
    if any(word in q for word in ("churn", "at risk", "risk of leaving", "leave us")):
        goals += ["cohort", "rfm", "churn"]
    if any(word in q for word in ("high-value", "high value", "clv", "lifetime value", "valuable")):
        goals += ["cohort", "clv"]
    if any(word in q for word in ("recommend", "recommendation", "suggest a product", "product suggestions")):
        goals += ["recommendation"]
        if not customer_id: goals.append("cohort")
    if "forecast" in q or "demand" in q:
        goals += ["forecast"]
    if "sentiment" in q or "review" in q:
        goals += ["sentiment"]
    if any(word in q for word in ("revenue", "orders", "aov", "total customers", "business metric")):
        goals += ["analytics"]
    if "rfm" in q or "recency" in q:
        goals += ["cohort","rfm"]
    if not goals:
        goals = ["rag"]
    goals = list(dict.fromkeys(goals))
    intent = "multi_step" if len(goals) > 1 else goals[0]
    return intent, goals


def rule_select_next(state: dict[str, Any]) -> dict[str, Any] | None:
    """Select one next tool from the current goals and actual validated results."""
    goals = state.get("objectives", [])
    results = state.get("tool_results", [])
    successful = {row["tool"] for row in results if row.get("valid")}
    resolved = set(successful)
    for row in results:
        if row.get("fallback_used") and row.get("valid") and row.get("fallback_for"):
            resolved.add(row["fallback_for"])

    if "cohort" in goals and "customer_cohort_tool" not in successful:
        segment = "high_value" if "clv" in goals else "at_risk" if "churn" in goals else "all"
        inputs={"segment":segment,"limit":100}
        if state.get("customer_id"): inputs["customer_id"]=state["customer_id"]
        return {"tool":"customer_cohort_tool","input":inputs}

    cohort = next((row["output"].get("customers", []) for row in reversed(results)
                   if row.get("tool") == "customer_cohort_tool" and row.get("valid")), [])
    if "cohort" in goals and any(row.get("tool")=="customer_cohort_tool" and row.get("valid") for row in results) and not cohort:
        return None
    rfm_result = next((row["output"].get("customers", []) for row in reversed(results)
                       if row.get("tool")=="customer_rfm_tool" and row.get("valid")),None)
    clv_result = next((row["output"].get("customers", []) for row in reversed(results)
                       if row.get("tool") in {"clv_prediction_tool", "historical_value_tool"} and row.get("valid")), None)
    churn_result = next((row["output"].get("customers", []) for row in reversed(results)
                         if row.get("tool") in {"churn_prediction_tool", "rfm_risk_tool"} and row.get("valid")), None)

    if "rfm" in goals and "customer_rfm_tool" not in successful and cohort:
        return {"tool":"customer_rfm_tool","input":{"customer_ids":[str(row["customer_unique_id"]) for row in cohort if row.get("customer_unique_id")]}}
    if "clv" in goals and "clv_prediction_tool" not in resolved and (rfm_result or cohort):
        return {"tool":"clv_prediction_tool","input":{"customers":rfm_result or cohort}}
    if "churn" in goals and "churn_prediction_tool" not in resolved and (rfm_result or clv_result or cohort):
        candidates = clv_result or rfm_result or cohort
        return {"tool": "churn_prediction_tool", "input": {"customers": candidates}}
    if "churn" in goals and "churn_prediction_tool" not in resolved:
        # An upstream cohort/RFM/CLV action failed; do not manufacture inputs
        # or silently substitute recommendations from unrelated customer data.
        return None
    if "churn" in goals and "churn_prediction_tool" in successful:
        churn_result = next((row["output"].get("customers", []) for row in reversed(results)
                             if row.get("tool") == "churn_prediction_tool" and row.get("valid")), churn_result)
    if "churn" in goals and "churn_prediction_tool" in resolved and churn_result is None:
        churn_result=next((row["output"].get("customers",[]) for row in reversed(results)
                           if row.get("fallback_used") and row.get("valid") and row.get("fallback_for")=="churn_prediction_tool"),None)
    if "recommendation" in goals and "recommendation_tool" not in successful:
        # For a churn-targeted request, only recommend after actual churn scores
        # or an explicitly executed recency-proxy fallback exist.
        if "churn" in goals and churn_result is None:
            return None
        candidates = churn_result or clv_result or cohort
        ids = [str(row["customer_unique_id"]) for row in candidates[:5] if row.get("customer_unique_id")]
        if ids:
            return {"tool": "recommendation_tool", "input": {"customer_ids": ids, "limit": 5}}
        if state.get("customer_id"):
            return {"tool": "recommendation_tool", "input": {"customer_ids": [state["customer_id"]], "limit": 5}}

    if "analytics" in goals and "business_analytics_tool" not in successful:
        q = state.get("user_query", "").lower()
        metric = "orders" if "order" in q else "customers" if "customer" in q and "revenue" not in q else "aov" if "aov" in q else "revenue"
        return {"tool": "business_analytics_tool", "input": {"metric": metric}}
    if "anomaly" in goals and "anomaly_detection_tool" not in successful:
        q = state.get("user_query", "").lower()
        metric = "orders" if "order" in q and "revenue" not in q else "revenue"
        return {"tool": "anomaly_detection_tool", "input": {"metric": metric}}
    if "anomaly" in goals and "anomaly_detection_tool" in successful:
        latest = next(row["output"] for row in reversed(results) if row["tool"] == "anomaly_detection_tool" and row.get("valid"))
        if latest.get("flags") and "rag" in goals and "rag_search" not in successful:
            return {"tool":"rag_search", "input":{"query":f"Business context for detected {latest['metric']} anomaly", "k":5}}
    if "forecast" in goals and "demand_forecast_tool" not in successful:
        return {"tool":"demand_forecast_tool", "input":{"days":7}}
    if "sentiment" in goals and "sentiment_tool" not in successful:
        review = state.get("review_text")
        if review:
            return {"tool":"sentiment_tool", "input":{"review":review}}
    conditional_rag=("anomaly" in goals and "why" in state.get("user_query", "").lower())
    anomaly_result=next((row["output"] for row in reversed(results) if row["tool"]=="anomaly_detection_tool" and row.get("valid")),None)
    if "rag" in goals and not (conditional_rag and anomaly_result is not None and not anomaly_result.get("flags")) and "rag_search" not in successful:
        return {"tool":"rag_search", "input":{"query":state.get("user_query", ""),"k":5}}
    return None
