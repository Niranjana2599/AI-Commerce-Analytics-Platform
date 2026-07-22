"""Optional LangSmith tracing helpers for the RAG chatbot only.

The fallback decorator makes local development possible before ``langsmith`` is
installed. Production tracing is enabled entirely through environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar


def configure_langsmith_environment() -> None:
    """Map requested LangChain-style variables to LangSmith SDK variables."""
    if "LANGSMITH_TRACING" not in os.environ:
        os.environ["LANGSMITH_TRACING"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
    aliases = {
        "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
        "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
    }
    for current_name, legacy_name in aliases.items():
        if current_name not in os.environ and os.getenv(legacy_name):
            os.environ[current_name] = os.environ[legacy_name]


configure_langsmith_environment()

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - protects source-only local usage
    def traceable(*_args: Any, **_kwargs: Any) -> Callable[[FunctionType], FunctionType]:
        """Return the original function when the optional SDK is not installed."""
        return lambda function: function


RAG_TAGS = ["ai-commerce", "rag", "ollama"]


def trace_metadata(*, prompt_version: str, knowledge_base_version: str, model_name: str) -> dict[str, str]:
    """Return consistent metadata visible on LangSmith RAG runs."""
    return {
        "application": "AI-Commerce-Analytics-Platform",
        "component": "rag_chatbot",
        "prompt_version": prompt_version,
        "knowledge_base_version": knowledge_base_version,
        "model_name": model_name,
    }
