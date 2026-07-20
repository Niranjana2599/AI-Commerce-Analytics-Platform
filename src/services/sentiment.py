"""Review-label helpers that match the notebook's rating-based labels."""

import pandas as pd


def label_sentiment(review_scores: pd.Series) -> pd.Series:
    """Map 1–5 review scores to Negative, Neutral, and Positive labels."""
    return pd.cut(review_scores, bins=[0, 2, 3, 5], labels=["Negative", "Neutral", "Positive"], include_lowest=True).astype("string")
