# RAG Architecture

## Purpose

The RAG chatbot answers e-commerce questions using a persisted **FAISS vector index** built from product, review, and dashboard knowledge. It retrieves relevant evidence, passes that evidence to a versioned prompt, and optionally uses a local **Ollama (Llama 3.2)** LLM to generate a grounded answer.

The API returns the answer together with source labels, prompt version, knowledge-base version, and evaluation signals.

---

## Current RAG Architecture

```text
User Question
      |
      v
FastAPI /api/v1/chat
      |
      v
RAG Service
      |
      +----------------------+
      |                      |
      v                      v
FAISS Vector Store      Prompt Registry
      |                      |
      | retrieved docs       |
      +-----------> Context -+
                             |
                             v
                    Ollama / Llama 3.2
                       (optional)
                             |
                             v
                       Answer Parser
                             |
                             v
              Answer + Sources + Metadata
```

### Current implementation

| Component | Current role |
|---|---|
| FastAPI | Exposes the `/api/v1/chat` endpoint. |
| RAG service | Coordinates retrieval, prompt construction, LLM generation, fallback handling, and response metadata. |
| FAISS | Persists and searches the vector index for relevant commerce documents. |
| Embeddings | Convert knowledge-base documents and user questions into vectors for semantic retrieval. |
| Prompt registry | Provides the versioned `rag_analyst` prompt. |
| Ollama | Optional local LLM generation using `Llama 3.2`. |
| Output parser | Normalizes the generated/fallback answer. |
| LangSmith | Optional tracing and RAG observability. |
| Prometheus/Grafana | Tracks operational metrics such as request timing, retrieval information, and failures. |
| `rag_ops` | Stores privacy-conscious operational and evaluation information. |

> **Current state:** The active chatbot execution path uses the persisted **FAISS-based retriever**. The older `rag_retriever.joblib` retriever artifact is no longer the active retrieval implementation.

---

## Knowledge Base

The RAG knowledge base is built from three main commerce sources:

### 1. Products

Product information such as:

- Product/category information
- Seller information
- Price-related information
- Other product attributes used by the application

### 2. Customer Reviews

Review documents containing:

- Review text
- Review score
- Relevant review metadata

Duplicate review content is reduced during knowledge-base preparation.

### 3. Dashboard Metrics

Compact summaries of important business metrics used by the analytics dashboard.

Examples include:

- Revenue
- Customer metrics
- Other aggregated commerce KPIs

These sources allow the chatbot to answer both **descriptive commerce questions** and questions requiring supporting business evidence.

---

## Vector Retrieval

The retrieval layer uses **FAISS** to search the persisted vector index.

The high-level process is:

```text
Knowledge Documents
        |
        v
Text Preparation
        |
        v
Embedding Generation
        |
        v
FAISS Vector Index
        |
        | persisted
        v
models/faiss_rag_index
```

At query time:

```text
User Question
      |
      v
Question Embedding
      |
      v
FAISS Similarity Search
      |
      v
Top-K Relevant Documents
      |
      v
Context for LLM
```

The `limit` supplied to the chatbot controls how many retrieved documents are used for the request.

---

## Retrieval and Prompt Flow

```mermaid
sequenceDiagram
  participant U as User
  participant API as FastAPI
  participant R as FAISS Retriever
  participant P as Prompt Registry
  participant O as Ollama
  participant L as LangSmith

  U->>API: question + limit
  API->>R: retrieve relevant documents
  R-->>API: documents + sources + retrieval information
  API->>P: render question + retrieved context
  P-->>API: versioned prompt

  alt Ollama enabled and reachable
    API->>O: generate(prompt)
    O-->>API: generated answer
  else Ollama disabled/unavailable
    API-->>API: use retrieved evidence as fallback
  end

  API->>API: parse answer + evaluate proxy signals
  API-. trace stages .->L
  API-->>U: answer + sources + metadata
```

---

## RAG API

The chatbot endpoint is:

```text
POST /api/v1/chat
```

Example request:

```json
{
  "question": "What is the total revenue?",
  "limit": 5
}
```

Example response structure:

```json
{
  "answer": "The total revenue is 3294778880.00, as stated in the dashboard metrics.",
  "sources": [
    "dashboard_metrics",
    "dashboard_metrics",
    "dashboard_metrics",
    "products",
    "products"
  ],
  "prompt_version": "2026-07-21.1",
  "knowledge_base_version": "faiss-1788012552746534000",
  "evaluation": {
    "faithfulness": 0.571,
    "answer_relevance": 0.75,
    "context_relevance": 0.5
  }
}
```

The public response exposes **source labels and metadata**, rather than returning the complete retrieved context.

---

## LLM Generation

The project uses a local **Ollama** service for optional answer generation.

Current configuration:

```text
RAG_LLM_ENABLED=true
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
```

The flow is:

```text
FAISS Retrieval
      |
      v
Retrieved Context
      |
      v
Versioned RAG Prompt
      |
      v
Ollama / Llama 3.2
      |
      v
Generated Answer
```

Using a local LLM keeps generation within the local development/deployment environment rather than requiring a hosted LLM API.

---

## Grounded Answering and Fallback

The prompt instructs the model to answer using the supplied retrieved evidence.

If LLM generation is disabled or unavailable, the service can fall back to the retrieved evidence instead of requiring successful LLM generation.

This gives the application two useful behaviors:

```text
LLM available
    -> Retrieved context
    -> Prompt
    -> Llama 3.2
    -> Generated grounded answer

LLM unavailable
    -> Retrieved context
    -> Evidence-based fallback answer
```

---

## Prompt Management

The RAG prompt is maintained as a named, versioned prompt rather than embedding the complete prompt logic directly inside the service.

The prompt is responsible for:

- Using supplied context.
- Avoiding unsupported claims.
- Identifying insufficient evidence.
- Referencing the available source types.
- Producing a concise business-oriented response.

When changing the prompt:

1. Create a new immutable prompt version.
2. Run the RAG evaluation dataset.
3. Compare answer quality and operational metrics.
4. Review failures.
5. Promote the new version only after validation.

---

## Evaluation

The RAG evaluation pipeline is:

```text
Evaluation Dataset
       |
       v
Production-equivalent RAG flow
       |
       +--> Answer relevance
       +--> Context relevance
       +--> Faithfulness / groundedness
       +--> Retrieval accuracy
       +--> Hallucination-risk signals
       +--> Latency / operational signals
```

Run the evaluation with:

```powershell
docker compose exec backend python -m pipelines.evaluation.evaluate_rag --dataset "AI Commerce RAG Evaluation"
```

The current evaluation values are **proxy/operational signals**. They should not be treated as a complete measure of production quality. Human review, reference answers, or an LLM-as-judge approach can be added for stronger evaluation.

---

## Observability

The RAG system has multiple observability layers.

### LangSmith

When enabled, LangSmith can trace stages such as:

```text
RAG Request
   |
   +-- Retrieval
   +-- Prompt
   +-- Ollama generation
   +-- Output parsing
   +-- Parent request
```

This helps debug retrieval quality, prompt behavior, LLM failures, and latency.

### Prometheus / Grafana

Operational metrics are exposed for monitoring application behavior, including request and chatbot-related metrics.

### Local RAG Operations

The `rag_ops` layer records privacy-conscious operational information such as:

- Salted question hash
- Input/output lengths
- Source types
- Knowledge-base version
- Prompt version
- Timing information
- Evaluation values

---

## Failure Handling

| Failure | Expected behavior |
|---|---|
| FAISS index/artifact is missing | RAG request returns a clear service error rather than silently generating an answer. |
| No usable documents are retrieved | RAG request handles the retrieval failure according to the service error path. |
| Ollama timeout/connection failure | The service records the failure and can fall back to retrieved evidence. |
| Invalid Ollama response | The service handles the generation failure and can use the retrieved evidence fallback. |
| LangSmith unavailable | Tracing is optional and should not prevent local RAG execution. |

---

## Deployment

The backend runs inside Docker Compose.

The main runtime relationship is:

```text
Docker Compose
     |
     +--------------------+
     |                    |
     v                    v
 FastAPI Backend       MLflow
     |
     +---- FAISS index
     |
     +---- RAG service
     |
     +---- Ollama connection
```

The backend container uses the persisted RAG assets available under the application's configured model/RAG directories.

The current project should keep the **FAISS index and the code that loads it consistent**. The old `rag_retriever.joblib` path should not be reintroduced unless the retriever implementation is intentionally changed back to that artifact format.

---

## Important Architecture Decision

The project previously contained a `rag_retriever.joblib`-based retrieval path.

The current implementation has moved to a **FAISS-based retrieval architecture**.

Therefore:

```text
Old
rag_retriever.joblib
        |
        v
TF-IDF EcommerceRetriever
```

has been replaced by:

```text
FAISS persisted index
        |
        v
FAISS vector retrieval
        |
        v
Retrieved commerce context
```

This means the **artifact format and loader must match**.

A FAISS directory should be loaded as a FAISS vector store; it should not be renamed to `rag_retriever.joblib` because a directory containing a FAISS index is not the same artifact type as a Joblib file.

---

## Verification

After starting the backend:

```powershell
docker compose up -d --force-recreate backend
```

Verify the API is available:

```powershell
docker compose exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').read().decode())"
```

Verify the RAG endpoint:

```powershell
docker compose exec backend python -c "import urllib.request,json; data=json.dumps({'question':'What is the total revenue?','limit':5}).encode(); req=urllib.request.Request('http://127.0.0.1:8000/api/v1/chat',data=data,headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req,timeout=120).read().decode())"
```

A successful response should contain:

```text
answer
sources
prompt_version
knowledge_base_version
evaluation
```

This confirms that the complete RAG path is working:

```text
API
 -> RAG service
 -> FAISS retrieval
 -> context construction
 -> prompt
 -> Ollama/fallback
 -> answer
 -> source + evaluation metadata
```

---

## Privacy

When LangSmith tracing is enabled, questions, retrieved context, prompts, and answers may be sent to the configured tracing service for debugging and evaluation.

The local `rag_ops` layer is designed to store privacy-conscious operational information rather than raw user questions.

External tracing should therefore be enabled only when the data, retention, and access requirements are appropriate.
