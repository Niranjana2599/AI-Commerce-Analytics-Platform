"""Main entry point for the AI Commerce Analytics Streamlit frontend."""

import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("Home Dashboard")
st.title("AI Commerce Analytics Platform")
st.caption("One workspace for customer intelligence, machine-learning predictions, and AI-assisted insights.")

with st.spinner("Checking backend status..."):
    health, error = api_request("GET", "/health")
show_api_error(error)
if health:
    st.success(f"Backend status: {health['status'].title()} ({health['environment']})")

st.subheader("Project overview")
st.write("Explore customer behaviour, predict business outcomes, generate recommendations, and ask questions of your commerce data.")

st.subheader("Available modules")
columns = st.columns(3)
for column, label, description in zip(columns * 3, [
    "Customer Analytics", "Churn Prediction", "CLV Prediction", "Delivery Delay", "Recommendations",
    "Sentiment Analysis", "Demand Forecasting", "RAG Chatbot", "About Project",
], [
    "KPIs and customer insights", "Identify retention risk", "Estimate customer value", "Assess delivery risk",
    "Find relevant products", "Understand review tone", "Plan future demand", "Ask data-grounded questions", "Architecture and usage",
]):
    with column:
        st.markdown(f"### {label}\n{description}")
