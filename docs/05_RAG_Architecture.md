# RAG Architecture

## Purpose

The RAG chatbot answers commerce questions using persisted product, review, and dashboard documents. It returns source labels and uses retrieved evidence as a safe fallback when local LLM generation is disabled or unavailable.

## Components

| Component | Current role |
| --- | --- |
| Retriever | Persisted `EcommerceRetriever` using TF-IDF cosine similarity; returns text, source, and score. |
| Prompt registry | Holds the versioned `rag_analyst` prompt outside service business logic. |
| Ollama | Optional local `/api/generate` call when `RAG_LLM_ENABLED=true`. |
| Output parser | Normalizes the final string response. |
| LangSmith | Optional tracing of retriever, prompt, Ollama, parser, and parent chain. |
| Local RAG operations | Privacy-safe hashes, timing, source types, versions, and proxy evaluation aggregates. |

> **Current state:** LangChain and FAISS are not required by the active execution path. The implementation uses LangSmith’s `traceable` decorator and a persisted TF-IDF retriever. A future FAISS/LangChain migration should preserve the same service contract and observability names.

## Retrieval and prompt flow

```mermaid
sequenceDiagram
  participant U as User
  participant API as FastAPI /chat
  participant R as TF-IDF retriever
  participant P as Prompt registry
  participant O as Ollama (optional)
  participant L as LangSmith (optional)
  U->>API: question + chunk limit
  API->>R: search(question, k)
  R-->>API: documents, sources, similarity scores
  API->>P: render(question, context)
  P-->>API: final prompt + prompt version
  alt Ollama enabled and reachable
    API->>O: generate(prompt)
    O-->>API: response + token counters
  else Disabled or unavailable
    API-->>API: use retrieved context as fallback
  end
  API->>API: parse answer and evaluate proxy signals
  API-. trace stages .->L
  API-->>U: answer, sources, prompt version, KB version, scores
```

## Context generation

The retriever builds documents from:

- Product/category/seller/price summaries.
- Deduplicated review documents with score and comment text.
- Small dashboard metric summaries.

Results are ranked by TF-IDF cosine similarity. The current service joins source label and text into a context block, then passes it to the versioned prompt. Similarity scores and retrieved count are observable through LangSmith and Prometheus; the public API response exposes source labels rather than raw context.

## Prompt management

The prompt registry contains a named, versioned template. It instructs the assistant to use only supplied context, identify insufficient evidence, and cite source types. Updating a prompt should mean:

1. Create a new immutable version identifier.
2. Evaluate it against a curated LangSmith dataset.
3. Compare quality, latency, and failure signals.
4. Promote only after review.

## Evaluation

`pipelines.evaluation.evaluate_rag` runs a production-equivalent RAG target against a LangSmith dataset. It produces proxy scores for answer relevance, context relevance, faithfulness/groundedness, retrieval accuracy, and hallucination risk.

```powershell
docker compose exec backend python -m pipelines.evaluation.evaluate_rag --dataset "AI Commerce RAG Evaluation"
```

These are token-overlap operational signals. Use reference answers, human review, or an LLM-as-judge before high-stakes release decisions.

## Failure handling

| Failure | Behavior |
| --- | --- |
| Missing retriever artifact | API returns a clear 503 service error. |
| Unsupported artifact/no documents | RAG request fails and increments chatbot failure metrics. |
| Ollama timeout/connection/invalid JSON | Nested LangSmith request trace records failure; service falls back to retrieved context. |
| LangSmith unavailable | Tracing is optional and must not prevent local RAG service execution. |

## Privacy

LangSmith captures questions, retrieved context, prompts, and answers for debugging when enabled. The local `rag_ops` log intentionally stores only a salted question hash, lengths, source types, versions, timings, and evaluation values. Do not enable external tracing for sensitive data without approved retention and access controls.
