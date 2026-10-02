"""Reuse the platform's existing RAG evaluators without substituting agent metrics."""

from backend.app.llmops.observability import evaluate_rag


def evaluate_rag_answer(question: str, answer: str, context: str) -> dict[str, float]:
    """Return the existing faithfulness, answer-relevance and context-relevance scores."""
    return evaluate_rag(question, answer, context)
