"""Privacy-aware RAG query logging, evaluation, and aggregate reports."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from backend.app.core.config import settings


def _tokens(text: str) -> set[str]:
    return {word.strip(".,!?;:()[]{}\"'").lower() for word in text.split() if len(word) > 2}


def _safe_query_id(question: str) -> str:
    """Hash the question so raw user text is never persisted in operational logs."""
    salt = "ai-commerce-rag-v1"
    return hashlib.sha256(f"{salt}:{question.strip().lower()}".encode()).hexdigest()[:16]


def evaluate_rag(question: str, answer: str, context: str) -> dict[str, float]:
    """Return transparent overlap-based proxy scores for offline RAG monitoring.

    These are operational signals, not a substitute for human review or LLM-as-judge evaluation.
    """
    question_terms, answer_terms, context_terms = _tokens(question), _tokens(answer), _tokens(context)
    faithfulness = len(answer_terms & context_terms) / max(len(answer_terms), 1)
    answer_relevance = len(question_terms & answer_terms) / max(len(question_terms), 1)
    context_relevance = len(question_terms & context_terms) / max(len(question_terms), 1)
    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_relevance": round(context_relevance, 3),
    }


def _events_path() -> Path:
    settings.rag_ops_dir.mkdir(parents=True, exist_ok=True)
    return settings.rag_ops_dir / "rag_events.jsonl"


def log_rag_event(event: dict[str, object]) -> None:
    """Append one privacy-safe RAG event and refresh the aggregate report."""
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with _events_path().open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
    write_evaluation_report()


def read_events(limit: int | None = None) -> list[dict[str, object]]:
    """Read valid events; corrupted lines are skipped rather than breaking the dashboard."""
    path = _events_path()
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-limit:] if limit else events


def metrics_summary() -> dict[str, object]:
    """Compute dashboard-ready RAG monitoring metrics from logged events."""
    events = read_events()
    if not events:
        return {"query_count": 0, "averages": {}, "latency_ms": {}, "prompt_versions": {}, "knowledge_base_versions": {}}

    def average(name: str) -> float:
        return round(statistics.mean(float(event.get(name, 0)) for event in events), 3)

    retrieval = [float(event.get("retrieval_latency_ms", 0)) for event in events]
    prompt = [float(event.get("prompt_latency_ms", 0)) for event in events]
    total = [float(event.get("total_execution_ms", 0)) for event in events]
    return {
        "query_count": len(events),
        "averages": {key: average(key) for key in ("faithfulness", "answer_relevance", "context_relevance")},
        "latency_ms": {
            "retrieval_mean": round(statistics.mean(retrieval), 1),
            "retrieval_p95": round(sorted(retrieval)[max(int(len(retrieval) * 0.95) - 1, 0)], 1),
            "prompt_mean": round(statistics.mean(prompt), 1),
            "prompt_p95": round(sorted(prompt)[max(int(len(prompt) * 0.95) - 1, 0)], 1),
            "total_mean": round(statistics.mean(total), 1),
            "total_p95": round(sorted(total)[max(int(len(total) * 0.95) - 1, 0)], 1),
        },
        "prompt_versions": _counts(events, "prompt_version"),
        "knowledge_base_versions": _counts(events, "knowledge_base_version"),
    }


def _counts(events: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        value = str(event.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_evaluation_report() -> None:
    """Persist a compact aggregate report for sharing or auditing."""
    path = settings.rag_ops_dir / "evaluation_report.json"
    path.write_text(json.dumps(metrics_summary(), indent=2), encoding="utf-8")


def build_event(
    question: str,
    *,
    answer: str,
    context: str,
    sources: list[str],
    prompt_version: str,
    knowledge_base_version: str,
    retrieval_latency_ms: float,
    prompt_latency_ms: float,
    total_execution_ms: float,
    similarity_scores: list[float],
    retrieved_chunk_count: int,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    llm_error: object,
) -> dict[str, object]:
    """Build the event schema without storing raw user questions or answers."""
    return {
        "query_id": _safe_query_id(question),
        "query_length": len(question),
        "answer_length": len(answer),
        "source_types": sorted(set(sources)),
        "prompt_version": prompt_version,
        "knowledge_base_version": knowledge_base_version,
        "retrieval_latency_ms": round(retrieval_latency_ms, 1),
        "prompt_latency_ms": round(prompt_latency_ms, 1),
        "total_execution_ms": round(total_execution_ms, 1),
        "retrieved_chunk_count": retrieved_chunk_count,
        "similarity_scores": similarity_scores,
        "model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_error": str(llm_error) if llm_error else None,
        **evaluate_rag(question, answer, context),
    }
