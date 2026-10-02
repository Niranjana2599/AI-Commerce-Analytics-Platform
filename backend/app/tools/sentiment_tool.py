"""Adapter for the existing review sentiment service."""

from typing import Any
from backend.app.services.commerce import classify_sentiment


def run(args: dict[str, Any]) -> dict[str, Any]:
    review = args.get("review")
    if not isinstance(review, str) or not review.strip() or len(review) > 5_000:
        raise ValueError("review must be non-empty and at most 5,000 characters")
    return {"sentiment": classify_sentiment(review)}
