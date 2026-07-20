"""Cleaning functions shared by the data preparation workflow."""

from collections.abc import Mapping

import pandas as pd


DATETIME_COLUMNS = {
    "orders": ("order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"),
    "order_items": ("shipping_limit_date",),
    "order_reviews": ("review_creation_date", "review_answer_timestamp"),
}


def clean_dataframe(frame: pd.DataFrame, datetime_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Trim text, parse dates, remove duplicates, and compact numeric dtypes."""
    cleaned = frame.copy().drop_duplicates()
    for column in cleaned.select_dtypes(include="object"):
        cleaned[column] = cleaned[column].str.strip()
    for column in datetime_columns:
        if column in cleaned:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
    for column in cleaned.select_dtypes(include="float64"):
        cleaned[column] = pd.to_numeric(cleaned[column], downcast="float")
    for column in cleaned.select_dtypes(include="int64"):
        cleaned[column] = pd.to_numeric(cleaned[column], downcast="integer")
    return cleaned


def clean_datasets(datasets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: clean_dataframe(frame, DATETIME_COLUMNS.get(name, ())) for name, frame in datasets.items()}


def build_master_dataset(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create the order-item-level master dataset used by analytics workflows."""
    required = {"orders", "customers", "order_items", "products", "sellers", "order_payments", "order_reviews"}
    missing = required.difference(datasets)
    if missing:
        raise ValueError(f"Missing datasets: {', '.join(sorted(missing))}")
    master = datasets["orders"].merge(datasets["customers"], on="customer_id", how="left")
    for name, key in (("order_items", "order_id"), ("products", "product_id"), ("sellers", "seller_id"), ("order_payments", "order_id"), ("order_reviews", "order_id")):
        master = master.merge(datasets[name], on=key, how="left")
    return master
