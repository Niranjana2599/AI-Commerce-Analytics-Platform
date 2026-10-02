"""Simple explainable anomaly detection over recent daily business metrics."""

from typing import Any
import pandas as pd

from backend.app.services.commerce import master_data


def run(args: dict[str, Any]) -> dict[str, Any]:
    metric = args.get("metric", "revenue")
    if metric not in {"revenue", "orders", "aov"}:
        raise ValueError("metric must be revenue, orders, or aov")
    data = master_data().copy()
    data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"], errors="coerce")
    daily = data.groupby(data.order_purchase_timestamp.dt.floor("D")).agg(revenue=("payment_value","sum"), orders=("order_id","nunique"))
    daily["aov"] = daily.revenue / daily.orders.replace(0, float("nan"))
    values = daily[metric].dropna().tail(90)
    if len(values) < 8:
        return {"metric": metric, "status": "insufficient_history", "observations": int(len(values))}
    median = float(values.median())
    mad = float((values - median).abs().median())
    scores = (values - median) / (1.4826 * mad) if mad else values * 0
    flags = [{"date": str(day.date()), "value": float(values.loc[day]), "modified_z": float(scores.loc[day])} for day in values.index[scores.abs() > 3.5]]
    return {"metric": metric, "method": "median absolute deviation, modified z-score > 3.5", "baseline_median": median, "observations": int(len(values)), "flags": flags}
