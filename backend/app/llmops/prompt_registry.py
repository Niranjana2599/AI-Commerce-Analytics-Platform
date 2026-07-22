"""Versioned prompts for the RAG chatbot."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptDefinition:
    """A named prompt template with a stable version identifier."""

    name: str
    version: str
    template: str


RAG_ANALYST_V1 = PromptDefinition(
    name="rag_analyst",
    version="2026-07-21.1",
    template=(
        "You are an ecommerce analytics assistant. Answer only from the supplied context. "
        "If the context is insufficient, say so clearly. Cite source types in square brackets.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)


def build_rag_prompt(question: str, context: str) -> tuple[str, PromptDefinition]:
    """Render the current production prompt and return its version metadata."""
    return RAG_ANALYST_V1.template.format(question=question, context=context), RAG_ANALYST_V1
