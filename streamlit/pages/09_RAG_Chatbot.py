"""Conversation interface for the retrieval-augmented commerce chatbot."""

import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("RAG Chatbot")
st.title("AI Commerce Analyst")
st.caption("Ask questions about products, reviews, customers, and commerce metrics.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your commerce data")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant commerce context..."):
            result, error = api_request("POST", "/chat", json={"question": question, "limit": 5})
        show_api_error(error)
        if result:
            answer = result["answer"]
            st.markdown(answer)
            if result["sources"]:
                st.caption("Sources: " + ", ".join(result["sources"]))
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

if st.button("Clear chat"):
    st.session_state.chat_history = []
    st.rerun()
