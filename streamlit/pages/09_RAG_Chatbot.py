"""Production-style Streamlit interface for the commerce RAG assistant."""

from collections import Counter
from datetime import datetime
import json
import time

import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("RAG Chatbot")

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

SUGGESTED_QUESTIONS = [
    "What factors are associated with customer churn?",
    "Which product categories appear in the knowledge base?",
    "Summarize the available customer review information.",
    "What commerce data can you help me explore?",
]


def safe_score(scores: dict, key: str) -> float | None:
    value = scores.get(key)

    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def score_label(value: float | None) -> str:
    return "Not available" if value is None else f"{value:.0%}"


def render_assistant_message(message: dict) -> None:
    """Render answer with sources and RAG diagnostics."""

    st.markdown(
        message.get(
            "content",
            "No answer returned.",
        )
    )

    sources = message.get("sources", [])

    if sources:
        source_counts = Counter(
            str(source)
            for source in sources
        )

        with st.expander(
            f"Sources ({len(sources)})"
        ):
            for index, source in enumerate(
                sources,
                start=1,
            ):
                count = source_counts[
                    str(source)
                ]

                duplicate_note = (
                    f" · used {count} times"
                    if count > 1
                    else ""
                )

                st.markdown(
                    f"**[{index}] {source}**"
                    f"{duplicate_note}"
                )

            st.caption(
                "Sources are returned by the RAG API."
            )

    scores = message.get(
        "evaluation",
        {},
    )

    score_columns = st.columns(3)

    score_columns[0].metric(
        "Faithfulness",
        score_label(
            safe_score(
                scores,
                "faithfulness",
            )
        ),
    )

    score_columns[1].metric(
        "Answer relevance",
        score_label(
            safe_score(
                scores,
                "answer_relevance",
            )
        ),
    )

    score_columns[2].metric(
        "Context relevance",
        score_label(
            safe_score(
                scores,
                "context_relevance",
            )
        ),
    )

    metadata = []

    if message.get("latency_seconds") is not None:
        metadata.append(
            f"Response: "
            f"{message['latency_seconds']:.2f}s"
        )

    if message.get("prompt_version"):
        metadata.append(
            f"Prompt: "
            f"{message['prompt_version']}"
        )

    if message.get("knowledge_base_version"):
        metadata.append(
            f"Knowledge base: "
            f"{message['knowledge_base_version']}"
        )

    if message.get("timestamp"):
        metadata.append(
            message["timestamp"]
        )

    if metadata:
        st.caption(
            " · ".join(metadata)
        )


def export_conversation(
    history: list[dict],
) -> bytes:
    """Create a portable JSON conversation."""

    return json.dumps(
        history,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")


# ============================================================
# Page header
# ============================================================

page_hero(
    "Retrieval-augmented intelligence",
    "AI Commerce Analyst",
    (
        "Ask grounded questions about the commerce "
        "knowledge base using LangChain, FAISS retrieval, "
        "and local Ollama generation."
    ),
)


# ============================================================
# Session state
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

history = st.session_state.chat_history

user_messages = sum(
    message.get("role") == "user"
    for message in history
)

assistant_messages = [
    message
    for message in history
    if message.get("role") == "assistant"
]

average_latency = (
    sum(
        message.get(
            "latency_seconds",
            0,
        )
        for message in assistant_messages
    )
    / len(assistant_messages)
    if assistant_messages
    else None
)


# ============================================================
# Session summary
# ============================================================

summary_columns = st.columns(4)

summary_columns[0].metric(
    "Questions",
    user_messages,
)

summary_columns[1].metric(
    "Answers",
    len(assistant_messages),
)

summary_columns[2].metric(
    "Average response",
    (
        "—"
        if average_latency is None
        else f"{average_latency:.2f}s"
    ),
)

summary_columns[3].metric(
    "Session sources",
    sum(
        len(
            message.get(
                "sources",
                [],
            )
        )
        for message in assistant_messages
    ),
)


# ============================================================
# Controls
# ============================================================

control_column, download_column, clear_column = (
    st.columns([3, 1, 1])
)

with control_column:

    retrieval_limit = st.select_slider(
        "Retrieved chunks per question",
        options=list(range(1, 11)),
        value=5,
        help=(
            "Controls how many relevant FAISS "
            "results are retrieved by the API."
        ),
    )

with download_column:

    st.write("")
    st.write("")

    st.download_button(
        "Export chat",
        data=export_conversation(history),
        file_name=(
            f"rag_conversation_"
            f"{datetime.now():%Y%m%d_%H%M}.json"
        ),
        mime="application/json",
        disabled=not history,
        width="stretch",
    )

with clear_column:

    st.write("")
    st.write("")

    if st.button(
        "Clear chat",
        disabled=not history,
        width="stretch",
    ):
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        st.rerun()


# ============================================================
# Suggested questions
# ============================================================

if not history:

    section_heading(
        "Start a conversation",
        "Choose a prompt or write your own question below.",
    )

    prompt_columns = st.columns(2)

    for index, prompt in enumerate(
        SUGGESTED_QUESTIONS
    ):

        if prompt_columns[
            index % 2
        ].button(
            prompt,
            key=f"suggestion_{index}",
            width="stretch",
        ):
            st.session_state.pending_question = prompt
            st.rerun()


# ============================================================
# Conversation
# ============================================================

section_heading(
    "Conversation",
    (
        "Questions are answered using relevant "
        "documents retrieved from the FAISS knowledge base "
        "and generated with the local Ollama LLM."
    ),
)

conversation = st.container(
    border=True
)

with conversation:

    if not history:
        st.info(
            "No messages yet. Ask a question to begin."
        )

    for message in history:

        avatar = (
            "🛍️"
            if message.get("role") == "assistant"
            else "👤"
        )

        with st.chat_message(
            message.get(
                "role",
                "assistant",
            ),
            avatar=avatar,
        ):

            if message.get("role") == "assistant":
                render_assistant_message(
                    message
                )
            else:
                st.markdown(
                    message.get(
                        "content",
                        "",
                    )
                )


# ============================================================
# New question
# ============================================================

typed_question = st.chat_input(
    "Ask a question about your commerce data"
)

question = (
    st.session_state.pending_question
    or typed_question
)


if question:

    st.session_state.pending_question = None

    question = question.strip()

    if not question:

        st.warning(
            "Enter a question before sending."
        )

    else:

        user_message = {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime(
                "%H:%M:%S"
            ),
        }

        st.session_state.chat_history.append(
            user_message
        )

        with conversation:

            with st.chat_message(
                "user",
                avatar="👤",
            ):
                st.markdown(question)

            with st.chat_message(
                "assistant",
                avatar="🛍️",
            ):

                started = time.perf_counter()

                with st.status(
                    "Running RAG pipeline...",
                    expanded=True,
                ) as status:

                    st.write(
                        "Searching FAISS knowledge base"
                    )

                    result, error = api_request(
                        "POST",
                        "/chat",
                        json={
                            "question": question,
                            "limit": retrieval_limit,
                        },
                        timeout=120,
                    )

                    elapsed = (
                        time.perf_counter()
                        - started
                    )

                    if error:

                        status.update(
                            label=(
                                "RAG request failed"
                            ),
                            state="error",
                        )

                    else:

                        st.write(
                            "Generating grounded answer "
                            "with Ollama"
                        )

                        status.update(
                            label="Answer ready",
                            state="complete",
                            expanded=False,
                        )

                show_api_error(error)

                if result and not error:

                    assistant_message = {
                        "role": "assistant",
                        "content": str(
                            result.get(
                                "answer",
                                "No answer returned.",
                            )
                        ),
                        "sources": (
                            result.get(
                                "sources"
                            )
                            or []
                        ),
                        "prompt_version": (
                            result.get(
                                "prompt_version"
                            )
                        ),
                        "knowledge_base_version": (
                            result.get(
                                "knowledge_base_version"
                            )
                        ),
                        "evaluation": (
                            result.get(
                                "evaluation"
                            )
                            or {}
                        ),
                        "latency_seconds": elapsed,
                        "timestamp": datetime.now().strftime(
                            "%H:%M:%S"
                        ),
                    }

                    render_assistant_message(
                        assistant_message
                    )

                    st.session_state.chat_history.append(
                        assistant_message
                    )

                elif error:

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": (
                                "The RAG request failed. "
                                "Please check that the "
                                "FastAPI backend and Ollama "
                                "service are running."
                            ),
                            "sources": [],
                            "evaluation": {},
                            "latency_seconds": elapsed,
                            "timestamp": datetime.now().strftime(
                                "%H:%M:%S"
                            ),
                        }
                    )


# ============================================================
# Conversation quality
# ============================================================

if assistant_messages:

    section_heading(
        "Conversation quality",
        (
            "Evaluation scores reported by the "
            "RAG API for this session."
        ),
    )

    score_names = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer relevance",
        "context_relevance": "Context relevance",
    }

    averages = {}

    for key, label in score_names.items():

        values = [
            safe_score(
                message.get(
                    "evaluation",
                    {},
                ),
                key,
            )
            for message in assistant_messages
        ]

        valid_values = [
            value
            for value in values
            if value is not None
        ]

        if valid_values:
            averages[label] = (
                sum(valid_values)
                / len(valid_values)
            )

    if averages:

        first_label = next(
            iter(averages)
        )

        radar_labels = (
            list(averages)
            + [first_label]
        )

        radar_values = (
            list(averages.values())
            + [averages[first_label]]
        )

        radar = go.Figure(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_labels,
                fill="toself",
                line=dict(
                    color="#6366f1",
                    width=3,
                ),
                fillcolor=(
                    "rgba(99,102,241,.25)"
                ),
                hovertemplate=(
                    "%{theta}: "
                    "%{r:.0%}"
                    "<extra></extra>"
                ),
            )
        )

        radar.update_layout(
            height=360,
            margin=dict(
                l=45,
                r=45,
                t=50,
                b=35,
            ),
            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            font=dict(
                color="#cbd5e1"
            ),
            title=(
                "Average session evaluation"
            ),
            polar=dict(
                bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                radialaxis=dict(
                    range=[0, 1],
                    tickformat=".0%",
                    gridcolor=(
                        "rgba(148,163,184,.18)"
                    ),
                ),
            ),
        )

        st.plotly_chart(
            radar,
            width="stretch",
            config=PLOT_CONFIG,
        )


# ============================================================
# Explanation
# ============================================================

with st.expander(
    "How to interpret this assistant"
):

    st.markdown(
        """
        - Questions are converted into embeddings using a
          local Hugging Face sentence-transformer model.
        - FAISS retrieves the most relevant knowledge-base
          chunks for the question.
        - The retrieved context is passed to the local
          Ollama Llama 3.2 model.
        - The answer is therefore grounded in the retrieved
          commerce knowledge base rather than generated only
          from the LLM's general knowledge.
        - Sources identify the knowledge-base categories
          returned by the API.
        - Faithfulness, answer relevance, and context
          relevance are automated RAG diagnostics.
        - Response time includes the Streamlit API request,
          retrieval, and LLM generation.
        - Avoid entering secrets or sensitive personal
          information in questions.
        """
    )