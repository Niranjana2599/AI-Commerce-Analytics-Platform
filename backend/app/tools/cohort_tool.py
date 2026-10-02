"""Build a customer cohort from existing engineered features without inventing fields."""

from typing import Any

from backend.app.services.commerce import master_data
from src.features.customer import build_customer_features


def run(args: dict[str, Any]) -> dict[str, Any]:
    segment = args.get("segment", "all")
    limit = args.get("limit", 100)
    customer_id = args.get("customer_id")
    if segment not in {"all", "high_value", "at_risk"}:
        raise ValueError("segment must be all, high_value, or at_risk")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 through 100")
    customers = build_customer_features(master_data())
    if customer_id:
        customers = customers[customers["customer_unique_id"].astype(str) == str(customer_id)]
        if customers.empty:
            raise ValueError("Unknown customer_id")
        segment = "customer_id"
    elif segment == "high_value":
        threshold = float(customers["monetary"].quantile(0.75))
        customers = customers[customers["monetary"] >= threshold].sort_values("monetary", ascending=False)
    elif segment == "at_risk":
        # Recency is a transparent descriptive rule; it is not a churn-model score.
        customers = customers[customers["recency_days"] > 180].sort_values("recency_days", ascending=False)
    else:
        customers = customers.sort_values("monetary", ascending=False)
    rows = customers.head(limit).where(customers.head(limit).notna(), None).to_dict("records")
    if not rows:
        return {"segment":segment,"customers":[],"count":0,"feature_columns":list(customers.columns),
                "cohort_definition":"No customers matched the requested cohort."}
    return {"segment":segment,"customers":rows,"count":int(len(rows)),"feature_columns":list(customers.columns),
            "cohort_definition":"historical monetary >= 75th percentile" if segment=="high_value" else "recency_days > 180" if segment=="at_risk" else "all available customers"}
