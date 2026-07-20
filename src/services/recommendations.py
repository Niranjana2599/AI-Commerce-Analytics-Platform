"""Lightweight, explainable product recommendations."""

import pandas as pd


class ProductRecommender:
    """Recommend popular unseen products from historical customer interactions."""

    def fit(self, interactions: pd.DataFrame) -> "ProductRecommender":
        required = {"customer_unique_id", "product_id"}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(f"Missing interaction columns: {', '.join(sorted(missing))}")
        data = interactions.dropna(subset=list(required)).copy()
        self.seen_products_ = data.groupby("customer_unique_id")["product_id"].agg(set).to_dict()
        self.popular_products_ = data["product_id"].value_counts().index.tolist()
        return self

    def recommend(self, customer_id: str, k: int = 10) -> list[str]:
        if not hasattr(self, "popular_products_"):
            raise RuntimeError("Call fit before requesting recommendations.")
        seen = self.seen_products_.get(customer_id, set())
        return [product for product in self.popular_products_ if product not in seen][:k]
