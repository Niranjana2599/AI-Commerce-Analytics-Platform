"""Project information page."""

import streamlit as st

from utils.ui import setup_page


setup_page("About")
st.title("About the Project")
st.write("AI Commerce Analytics Platform combines ecommerce analytics, machine-learning models, and retrieval-augmented answers in one dashboard.")

st.subheader("Architecture")
st.code("Streamlit frontend  →  FastAPI backend  →  saved models + prepared ecommerce data")

st.subheader("Included modules")
st.markdown("""
- Customer analytics and platform KPIs
- Churn, CLV, and delivery-delay prediction
- Product recommendations and sentiment analysis
- Demand forecasting and RAG chatbot answers
""")

st.subheader("Run locally")
st.code("streamlit run streamlit/app.py", language="bash")
st.info("Start the FastAPI backend first, then set FASTAPI_BASE_URL if it is not running at the default local address.")
