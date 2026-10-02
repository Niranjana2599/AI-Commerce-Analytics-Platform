"""Transparent recency heuristic used only as a churn-model fallback."""

from typing import Any


def run(args: dict[str, Any]) -> dict[str, Any]:
    rows = args.get("customers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("customers must be a non-empty cohort list")
    ranked = [dict(row) for row in sorted(rows, key=lambda row: float(row.get("recency_days", 0) or 0), reverse=True)]
    for row in ranked:
        row["risk_proxy"] = "high" if float(row.get("recency_days", 0) or 0) > 180 else "lower"
    return {"customers":ranked[:20],"method":"recency_days > 180 is a high-risk proxy","is_model_prediction":False}
