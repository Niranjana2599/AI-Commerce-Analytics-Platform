"""Customer-review sentiment analysis page."""

import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


POSITIVE_WORDS = {"great", "excellent", "love", "perfect", "fast", "good", "amazing"}
NEGATIVE_WORDS = {"bad", "worst", "late", "broken", "poor", "disappointed", "terrible"}


def highlight_words(text: str) -> str:
    """Use simple HTML spans to make common sentiment words easier to scan."""
    for word in POSITIVE_WORDS:
        text = text.replace(word, f"<span style='color:#1b8a3a;font-weight:bold'>{word}</span>")
    for word in NEGATIVE_WORDS:
        text = text.replace(word, f"<span style='color:#c62828;font-weight:bold'>{word}</span>")
    return text


setup_page("Sentiment Analysis")
st.title("Review Sentiment Analysis")
review = st.text_area("Customer review", placeholder="Write or paste a review here...", height=160)

if st.button("Analyse sentiment", type="primary"):
    if not review.strip():
        st.warning("Enter a review before analysing it.")
    else:
        with st.spinner("Analysing review sentiment..."):
            result, error = api_request("POST", "/sentiment", json={"review": review})
        show_api_error(error)
        if result:
            st.success(f"Predicted sentiment: {result['sentiment']}")
            st.markdown(highlight_words(review.lower()), unsafe_allow_html=True)
            st.caption("Green highlights positive words and red highlights negative words. Probability is not currently returned by the API.")
