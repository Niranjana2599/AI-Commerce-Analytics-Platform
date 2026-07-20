"""Customer-level features used by churn and CLV workflows."""

import pandas as pd


def build_customer_features(master_df: pd.DataFrame, churn_after_days: int = 180) -> pd.DataFrame:
    """Aggregate transactions into a customer-level RFM-style feature table."""
    required = {"customer_unique_id", "order_id", "order_purchase_timestamp", "payment_value"}
    missing = required.difference(master_df.columns)
    if missing:
        raise ValueError(f"Missing customer columns: {', '.join(sorted(missing))}")
    data = master_df.copy()
    for column in ("order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"):
        if column in data:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(data):
        data["delivery_delay"] = (data["order_delivered_customer_date"] - data["order_estimated_delivery_date"]).dt.days.clip(lower=0)
    aggregations = {
        "total_orders": ("order_id", "nunique"), "total_spend": ("payment_value", "sum"),
        "average_order_value": ("payment_value", "mean"), "last_purchase": ("order_purchase_timestamp", "max"),
    }
    optional = {
        "average_review_score": ("review_score", "mean"), "average_discount": ("discount_rate", "mean"),
        "average_freight": ("freight_value", "mean"), "product_diversity": ("product_category_name", "nunique"),
        "average_delivery_delay": ("delivery_delay", "mean"),
    }
    aggregations.update({name: spec for name, spec in optional.items() if spec[0] in data})
    customers = data.groupby("customer_unique_id", as_index=False).agg(**aggregations)
    customers["recency_days"] = (data["order_purchase_timestamp"].max() - customers["last_purchase"]).dt.days
    customers["frequency"] = customers["total_orders"]
    customers["monetary"] = customers["total_spend"]
    customers["churn"] = (customers["recency_days"] > churn_after_days).astype("int8")
    return customers
