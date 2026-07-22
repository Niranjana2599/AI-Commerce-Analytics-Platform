"""Traceable RAG orchestration with optional local Ollama generation."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from backend.app.core.config import settings
from backend.app.llmops.knowledge_base import knowledge_base_version
from backend.app.llmops.langsmith_tracing import RAG_TAGS, trace_metadata, traceable
from backend.app.llmops.observability import build_event, log_rag_event
from backend.app.llmops.prompt_registry import build_rag_prompt
from backend.app.monitoring.metrics import observe_rag
from backend.app.services.commerce import load_model


@traceable(name="rag_retriever", run_type="retriever", tags=RAG_TAGS + ["retrieval"])
def _retrieve(question: str, limit: int) -> tuple[str, list[str], list[float]]:
    """Retrieve documents, sources, and TF-IDF similarity scores."""
    artifact = load_model("retriever")
    if hasattr(artifact, "search"):
        results = artifact.search(question, k=limit)
    elif isinstance(artifact, dict) and "documents" in artifact:
        results = artifact["documents"].head(limit)
    else:
        raise ValueError("Unsupported retriever artifact format.")
    if results.empty:
        raise ValueError("The retriever returned no documents.")
    texts = results["text"].astype(str).tolist()
    sources = results["source"].astype(str).tolist() if "source" in results else ["Dataset"] * len(texts)
    scores = results["score"].astype(float).round(6).tolist() if "score" in results else []
    context = "\n".join(f"[{source}] {text}" for source, text in zip(sources, texts))
    return context, sources, scores


@traceable(name="rag_prompt_template", run_type="prompt", tags=RAG_TAGS + ["prompt"])
def _render_prompt(question: str, context: str) -> tuple[str, str]:
    """Render the reusable, versioned prompt outside the retrieval business logic."""
    prompt, definition = build_rag_prompt(question, context)
    return prompt, definition.version


@traceable(name="rag_ollama_request", run_type="llm", tags=RAG_TAGS + ["llm"])
def _invoke_ollama(prompt: str) -> dict[str, Any]:
    """Invoke Ollama and preserve its usage fields for the LangSmith trace."""
    body = json.dumps({"model": settings.ollama_model, "prompt": prompt, "stream": False}).encode()
    request = Request(f"{settings.ollama_base_url.rstrip('/')}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    return {
        "answer": str(payload.get("response", "")).strip(),
        "model_name": str(payload.get("model", settings.ollama_model)),
        "prompt_tokens": int(payload.get("prompt_eval_count", 0)),
        "completion_tokens": int(payload.get("eval_count", 0)),
    }


@traceable(name="rag_llm", run_type="llm", tags=RAG_TAGS + ["llm"])
def _ollama_answer(prompt: str) -> dict[str, Any]:
    """Call Ollama with a safe context-only fallback while retaining failed child traces."""
    if not settings.rag_llm_enabled:
        return {"answer": None, "model_name": "retrieval-only", "prompt_tokens": 0, "completion_tokens": 0, "error": None}
    try:
        result = _invoke_ollama(prompt)
        result["error"] = None
        return result
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        # The nested request trace is marked failed; the parent can still return evidence.
        return {"answer": None, "model_name": settings.ollama_model, "prompt_tokens": 0, "completion_tokens": 0, "error": type(error).__name__}


@traceable(name="rag_output_parser", run_type="parser", tags=RAG_TAGS + ["parser"])
def _parse_answer(answer: str) -> str:
    """Normalize the final response as an explicit, observable output-parser stage."""
    return answer.strip()


@traceable(name="ai_commerce_rag_pipeline", run_type="chain", tags=RAG_TAGS)
def run_rag_pipeline(question: str, limit: int) -> dict[str, object]:
    """Execute the complete RAG pipeline and return internal evaluation inputs."""
    total_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    context, sources, similarity_scores = _retrieve(question, limit)
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1_000

    prompt, prompt_version = _render_prompt(question, context)
    generation_started = time.perf_counter()
    llm_result = _ollama_answer(prompt)
    response_latency_ms = (time.perf_counter() - generation_started) * 1_000
    answer = _parse_answer(str(llm_result["answer"] or context))

    kb_version = knowledge_base_version()
    return {
        "answer": answer,
        "context": context,
        "final_prompt": prompt,
        "sources": sources,
        "similarity_scores": similarity_scores,
        "retrieved_chunk_count": len(sources),
        "prompt_version": prompt_version,
        "knowledge_base_version": kb_version,
        "retrieval_latency_ms": retrieval_latency_ms,
        "response_latency_ms": response_latency_ms,
        "total_execution_ms": (time.perf_counter() - total_started) * 1_000,
        "model_name": str(llm_result["model_name"]),
        "prompt_tokens": int(llm_result["prompt_tokens"]),
        "completion_tokens": int(llm_result["completion_tokens"]),
        "llm_error": llm_result["error"],
        "trace_metadata": trace_metadata(prompt_version=prompt_version, knowledge_base_version=kb_version, model_name=str(llm_result["model_name"])),
    }


def answer_question(question: str, limit: int) -> dict[str, object]:
    """Return the public answer while recording privacy-safe local RAG operations."""
    started = time.perf_counter()
    try:
        result = run_rag_pipeline(question, limit)
        event = build_event(
            question,
            answer=str(result["answer"]),
            context=str(result["context"]),
            sources=list(result["sources"]),
            prompt_version=str(result["prompt_version"]),
            knowledge_base_version=str(result["knowledge_base_version"]),
            retrieval_latency_ms=float(result["retrieval_latency_ms"]),
            prompt_latency_ms=float(result["response_latency_ms"]),
            total_execution_ms=float(result["total_execution_ms"]),
            similarity_scores=list(result["similarity_scores"]),
            retrieved_chunk_count=int(result["retrieved_chunk_count"]),
            model_name=str(result["model_name"]),
            prompt_tokens=int(result["prompt_tokens"]),
            completion_tokens=int(result["completion_tokens"]),
            llm_error=result["llm_error"],
        )
        log_rag_event(event)
    except Exception:
        observe_rag(time.perf_counter() - started, 0, 0, 0, success=False)
        raise
    observe_rag(
        time.perf_counter() - started,
        float(result["retrieval_latency_ms"]) / 1_000,
        float(result["response_latency_ms"]) / 1_000,
        int(result["retrieved_chunk_count"]),
        success=True,
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "prompt_version": result["prompt_version"],
        "knowledge_base_version": result["knowledge_base_version"],
        "evaluation": {key: event[key] for key in ("faithfulness", "answer_relevance", "context_relevance")},
    }
