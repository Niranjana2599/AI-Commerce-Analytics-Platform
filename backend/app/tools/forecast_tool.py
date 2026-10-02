"""Adapter for the existing demand forecasting service."""

from typing import Any
from backend.app.services.commerce import demand_forecast


def run(args: dict[str, Any]) -> dict[str, Any]:
    days = int(args.get("days", 7))
    if not 1 <= days <= 90:
        raise ValueError("days must be from 1 through 90")
    points=demand_forecast(args.get("product_id"), days)
    return {"forecast":[{**point,"date":point["date"].isoformat() if hasattr(point.get("date"),"isoformat") else point.get("date")} for point in points]}
