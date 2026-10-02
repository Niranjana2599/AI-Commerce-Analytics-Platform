"""Adapter for existing persisted RAG retrieval."""

from typing import Any
from backend.app.services.rag_service import retrieve_evidence


def run(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 2_000:
        raise ValueError("query must be non-empty and at most 2,000 characters")
    k = int(args.get("k", 5))
    if not 1 <= k <= 10:
        raise ValueError("k must be from 1 through 10")
    documents, sources = retrieve_evidence(query, k)
    return {"documents": documents, "sources": sources}
