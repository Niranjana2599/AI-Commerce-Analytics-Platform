"""Interactive review sentiment analysis with transparent session insights."""

from collections import Counter
from datetime import datetime
from html import escape
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("Sentiment Analysis")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
POSITIVE_WORDS = {"amazing", "excellent", "fast", "fantastic", "good", "great", "happy", "love", "perfect", "recommend"}
NEGATIVE_WORDS = {"awful", "bad", "broken", "delay", "disappointed", "hate", "late", "poor", "terrible", "worst"}
NEGATED_POSITIVE = {f"not {word}" for word in POSITIVE_WORDS} | {f"never {word}" for word in POSITIVE_WORDS}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "i", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "you", "my", "very",
}
EXAMPLES = {
    "Write my own": "",
    "Positive example": "The product is excellent and delivery was fast.",
    "Negative example": "The product arrived late and the quality was terrible.",
    "Negation challenge": "The product was not good and I would not recommend it.",
    "Mixed opinion": "Delivery was fast, but the product stopped working after two days.",
}


def style_figure(figure, *, height=360):
    """Apply the platform's accessible Plotly presentation."""
    figure.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=54, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        title_font=dict(size=17, color="#f8fafc"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
    )
    return figure


def classify_highlight(token: str, previous: str | None) -> str | None:
    """Classify a token for UI highlighting, preserving simple negation."""
    lowered = token.lower()
    if lowered in POSITIVE_WORDS and previous in {"not", "never", "no"}:
        return "negative"
    if lowered in POSITIVE_WORDS:
        return "positive"
    if lowered in NEGATIVE_WORDS:
        return "negative"
    return None


def highlight_review(text: str) -> str:
    """Safely highlight common sentiment terms with simple negation awareness."""
    parts = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    previous_word = None
    highlighted = []
    for part in parts:
        if re.fullmatch(r"[A-Za-z']+", part):
            signal = classify_highlight(part, previous_word)
            safe_part = escape(part)
            if signal == "positive":
                safe_part = f'<mark style="background:rgba(16,185,129,.18);color:#34d399;padding:.08rem .2rem;border-radius:.25rem">{safe_part}</mark>'
            elif signal == "negative":
                safe_part = f'<mark style="background:rgba(239,68,68,.18);color:#f87171;padding:.08rem .2rem;border-radius:.25rem">{safe_part}</mark>'
            highlighted.append(safe_part)
            previous_word = part.lower()
        else:
            highlighted.append(escape(part))
    return "".join(highlighted)


def meaningful_words(texts: list[str]) -> Counter:
    """Count display-friendly words from session reviews."""
    words = []
    for text in texts:
        words.extend(
            word
            for word in re.findall(r"[a-zA-Z']+", text.lower())
            if len(word) > 2 and word not in STOP_WORDS
        )
    return Counter(words)


page_hero(
    "Voice of customer",
    "Customer review sentiment analysis",
    "Classify review sentiment, inspect visible language signals, and explore patterns across this browser session.",
)

if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []
if "sentiment_result" not in st.session_state:
    st.session_state.sentiment_result = None

section_heading("Analyze a review", "Paste customer feedback or select an example to test the deployed model.")
example_name = st.selectbox(
    "Review example",
    list(EXAMPLES),
    help="Examples make it easy to test direct sentiment, negation, and mixed language.",
)

with st.form("sentiment_form", clear_on_submit=False):
    review = st.text_area(
        "Customer review",
        value=EXAMPLES[example_name],
        placeholder="Write or paste a customer review here...",
        height=170,
        max_chars=5_000,
        help="The FastAPI contract accepts between 1 and 5,000 characters.",
    )
    st.caption(f"{len(review):,} / 5,000 characters")
    action_left, action_right = st.columns([3, 1])
    with action_left:
        submitted = st.form_submit_button(
            "Analyze sentiment",
            type="primary",
            width="stretch",
        )
    with action_right:
        clear_result = st.form_submit_button("Clear result", width="stretch")

if clear_result:
    st.session_state.sentiment_result = None
    st.rerun()

if submitted:
    normalized_review = review.strip()
    if not normalized_review:
        st.warning("Enter a review before analyzing sentiment.")
    elif len(normalized_review.split()) < 2:
        st.warning("Add more context than a single word for a more meaningful assessment.")
    else:
        with st.spinner("Analyzing review sentiment..."):
            result, error = api_request(
                "POST",
                "/sentiment",
                json={"review": normalized_review},
            )
        show_api_error(error)
        if result:
            label = str(result["sentiment"])
            item = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "review": normalized_review,
                "sentiment": label,
            }
            st.session_state.sentiment_result = item
            st.session_state.sentiment_history = (st.session_state.sentiment_history + [item])[-50:]

current = st.session_state.sentiment_result
if current:
    label = current["sentiment"]
    normalized_label = label.lower()
    is_positive = normalized_label in {"positive", "1", "true"}
    display_label = "Positive" if is_positive else "Negative" if normalized_label in {"negative", "0", "false"} else label.title()
    word_count = len(re.findall(r"[A-Za-z']+", current["review"]))
    sentence_count = max(len(re.findall(r"[.!?]+", current["review"])), 1)
    signal_count = sum(
        1
        for index, token in enumerate(re.findall(r"[A-Za-z']+", current["review"].lower()))
        if classify_highlight(
            token,
            re.findall(r"[A-Za-z']+", current["review"].lower())[index - 1] if index else None,
        )
    )

    section_heading("Sentiment result", "Review the model label alongside transparent text-level context.")
    result_columns = st.columns(4)
    result_columns[0].metric("Predicted sentiment", display_label)
    result_columns[1].metric("Confidence score", "Not provided", help="The current API returns a label only.")
    result_columns[2].metric("Word count", word_count)
    result_columns[3].metric("Visible signal terms", signal_count)

    if is_positive:
        st.success(f"Predicted sentiment: {display_label}", icon="😊")
    elif normalized_label in {"negative", "0", "false"}:
        st.error(f"Predicted sentiment: {display_label}", icon="☹️")
    else:
        st.info(f"Predicted sentiment: {display_label}")

    st.markdown("#### Highlighted review")
    st.markdown(
        f'<div class="module-card" style="min-height:auto;font-size:1.05rem;line-height:1.8">{highlight_review(current["review"])}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Green marks common positive terms; red marks common negative terms and positive words following simple negators. "
        "Highlights are explanatory UI rules, not model feature importance."
    )

    lowered_review = current["review"].lower()
    if any(phrase in lowered_review for phrase in NEGATED_POSITIVE):
        st.warning(
            "This review contains a negated positive phrase. The deployed model was trained with English stop-word removal and may mishandle words such as 'not'."
        )
    if sentence_count > 1 and any(word in lowered_review for word in POSITIVE_WORDS) and any(word in lowered_review for word in NEGATIVE_WORDS):
        st.info("This review contains mixed positive and negative language; a single binary label may not capture every aspect.")

history = st.session_state.sentiment_history
if history:
    section_heading("Session insights", "Explore up to the latest 50 reviews analyzed in this browser session.")
    toolbar_left, toolbar_right = st.columns([5, 1])
    with toolbar_left:
        st.caption(f"Session contains {len(history)} analyzed review{'s' if len(history) != 1 else ''}.")
    with toolbar_right:
        if st.button("Clear history", width="stretch"):
            st.session_state.sentiment_history = []
            st.session_state.sentiment_result = None
            st.rerun()

    history_frame = pd.DataFrame(history)
    sentiment_counts = history_frame["sentiment"].astype(str).str.title().value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
    word_counts = meaningful_words(history_frame["review"].tolist())
    frequent_words = pd.DataFrame(word_counts.most_common(15), columns=["Word", "Count"])

    sentiment_column, frequency_column = st.columns(2)
    with sentiment_column:
        donut = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Reviews",
            hole=0.58,
            title="Session sentiment distribution",
            color="Sentiment",
            color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#94a3b8"},
        )
        donut.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_figure(donut), width="stretch", config=PLOT_CONFIG)

    with frequency_column:
        if frequent_words.empty:
            st.info("No meaningful words are available for frequency analysis yet.")
        else:
            frequency = px.bar(
                frequent_words.sort_values("Count"),
                x="Count",
                y="Word",
                orientation="h",
                title="Most frequent session words",
                color="Count",
                color_continuous_scale=["#6366f1", "#0ea5e9"],
            )
            frequency.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_figure(frequency), width="stretch", config=PLOT_CONFIG)

    if not frequent_words.empty:
        treemap = px.treemap(
            frequent_words,
            path=["Word"],
            values="Count",
            color="Count",
            title="Frequent-term treemap",
            color_continuous_scale="Purples",
        )
        st.plotly_chart(style_figure(treemap, height=330), width="stretch", config=PLOT_CONFIG)

    st.dataframe(
        history_frame.rename(columns={"timestamp": "Time", "review": "Review", "sentiment": "Sentiment"})[
            ["Time", "Review", "Sentiment"]
        ].iloc[::-1],
        hide_index=True,
        width="stretch",
    )

    with st.expander("Interpretation and model limitations"):
        st.markdown(
            """
            - Session charts summarize only reviews analyzed in this browser session; they are not platform-wide distributions.
            - The API does not return probability, so no confidence gauge is displayed.
            - Keyword highlighting is a presentation aid and is independent of the trained model.
            - Negation and mixed-opinion reviews should be validated after the sentiment model is retrained on more realistic language.
            """
        )
