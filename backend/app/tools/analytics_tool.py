"""Allowlisted KPI queries over prepared commerce data."""

from typing import Any

from backend.app.services import commerce

ALLOWED_METRICS = {"revenue", "orders", "customers", "aov", "product_performance", "category_performance", "seller_performance"}


def run(args: dict[str, Any]) -> dict[str, Any]:
    metric = args.get("metric")
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(ALLOWED_METRICS)}")
    if metric in {"revenue", "orders", "customers", "aov"}:
        metrics = commerce.customer_metrics()
        key = {"revenue":"total_revenue", "orders":"total_orders", "customers":"total_customers", "aov":"average_order_value"}[metric]
        return {"metric": metric, "value": metrics[key]}
    data = commerce.master_data()
    dimension = {"product_performance":"product_id", "category_performance":"product_category_name", "seller_performance":"seller_id"}[metric]
    if dimension not in data:
        raise ValueError(f"Prepared dataset does not contain {dimension}")
    grouped = data.groupby(dimension, dropna=False).agg(revenue=("payment_value", "sum"), orders=("order_id", "nunique"))
    return {"metric": metric, "rows": grouped.sort_values("revenue", ascending=False).head(10).reset_index().fillna("unknown").to_dict("records")}
