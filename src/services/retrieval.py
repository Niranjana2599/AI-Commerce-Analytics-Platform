"""Dependency-light TF-IDF retrieval for the ecommerce analyst RAG workflow."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EcommerceRetriever:
    """Index text documents and return relevant, scored context for an LLM prompt."""

    def __init__(self, max_features: int = 10_000) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    def fit(self, documents: pd.DataFrame) -> "EcommerceRetriever":
        if "text" not in documents:
            raise ValueError("Documents must contain a 'text' column.")
        self.documents = documents.dropna(subset=["text"]).copy().reset_index(drop=True)
        self.documents["text"] = self.documents["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        self.matrix = self.vectorizer.fit_transform(self.documents["text"])
        return self

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        if not hasattr(self, "matrix"):
            raise RuntimeError("Call fit before searching.")
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        results = self.documents.copy()
        results["score"] = scores
        return results.nlargest(k, "score")

    def build_prompt(self, question: str, k: int = 5) -> str:
        context = "\n".join(f"[{getattr(row, 'source', 'Dataset')}] {row.text}" for row in self.search(question, k).itertuples())
        return f"Answer using only this ecommerce context. Cite source types.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
