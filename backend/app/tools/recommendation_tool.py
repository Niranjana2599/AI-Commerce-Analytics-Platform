"""Adapter for the existing recommendation service."""

from typing import Any
from backend.app.services.commerce import recommend


def run(args: dict[str, Any]) -> dict[str, Any]:
    customer_ids = args.get("customer_ids")
    if not isinstance(customer_ids, list) or not customer_ids or any(not isinstance(value,str) or not value.strip() for value in customer_ids):
        raise ValueError("customer_ids must contain customer identifiers")
    limit = int(args.get("limit", 5))
    if not 1 <= limit <= 20:
        raise ValueError("limit must be from 1 through 20")
    return {"recommendations":[{"customer_id":customer_id,"product_ids":recommend(customer_id,limit)} for customer_id in customer_ids]}
