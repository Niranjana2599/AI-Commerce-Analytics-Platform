"""Run offline RAG evaluations against a named LangSmith dataset.

Dataset inputs require ``question`` and may include ``expected_sources`` as a
list of source labels (for example, ``["Product", "Review"]``).
"""

from __future__ import annotations

import argparse
from typing import Any

from backend.app.llmops.observability import evaluate_rag
from backend.app.services.rag_service import run_rag_pipeline


def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute the production RAG pipeline for a LangSmith evaluation example."""
    result = run_rag_pipeline(str(inputs["question"]), int(inputs.get("limit", 5)))
    return {
        "answer": result["answer"],
        "context": result["context"],
        "sources": result["sources"],
        "final_prompt": result["final_prompt"],
    }


def _scores(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, float]:
    return evaluate_rag(str(inputs["question"]), str(outputs["answer"]), str(outputs["context"]))


def answer_relevance(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    return {"key": "answer_relevance", "score": _scores(inputs, outputs)["answer_relevance"]}


def context_relevance(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    return {"key": "context_relevance", "score": _scores(inputs, outputs)["context_relevance"]}


def faithfulness(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    return {"key": "faithfulness", "score": _scores(inputs, outputs)["faithfulness"]}


def retrieval_accuracy(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """Score expected-source overlap, or use context relevance as a labeled-data proxy."""
    expected = {str(item) for item in inputs.get("expected_sources", [])}
    retrieved = {str(item) for item in outputs.get("sources", [])}
    score = len(expected & retrieved) / len(expected) if expected else _scores(inputs, outputs)["context_relevance"]
    return {"key": "retrieval_accuracy", "score": round(score, 3)}


def hallucination_risk(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """Expose ungrounded-answer risk as the inverse of the faithfulness proxy."""
    score = 1 - _scores(inputs, outputs)["faithfulness"]
    return {"key": "hallucination_risk", "score": round(score, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AI Commerce RAG chatbot in LangSmith.")
    parser.add_argument("--dataset", required=True, help="Existing LangSmith dataset name.")
    parser.add_argument("--experiment-prefix", default="ai-commerce-rag-evaluation", help="Prefix for the generated LangSmith experiment.")
    arguments = parser.parse_args()

    from langsmith import Client

    client = Client()
    client.evaluate(
        rag_target,
        data=arguments.dataset,
        evaluators=[answer_relevance, context_relevance, faithfulness, retrieval_accuracy, hallucination_risk],
        experiment_prefix=arguments.experiment_prefix,
        metadata={"application": "AI-Commerce-Analytics-Platform", "component": "rag_chatbot"},
    )
    print(f"Submitted RAG evaluation for dataset: {arguments.dataset}")


if __name__ == "__main__":
    main()
