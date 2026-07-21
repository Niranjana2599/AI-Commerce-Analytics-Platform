"""Navigable home dashboard with live commerce KPIs."""

import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("Home Dashboard")
st.title("Home Dashboard")
st.write("A live overview of the AI Commerce Analytics Platform.")

with st.spinner("Loading commerce KPIs..."):
    metrics, error = api_request("GET", "/analytics/customer-metrics")
show_api_error(error)
if metrics:
    a, b, c, d = st.columns(4)
    a.metric("Total revenue", f"${metrics['total_revenue']:,.0f}")
    b.metric("Orders", f"{metrics['total_orders']:,}")
    c.metric("Customers", f"{metrics['total_customers']:,}")
    d.metric("Average order value", f"${metrics['average_order_value']:,.2f}")
    chart = px.bar(x=["Revenue", "AOV"], y=[metrics["total_revenue"], metrics["average_order_value"]], title="Revenue snapshot")
    st.plotly_chart(chart, use_container_width=True)

st.subheader("Model summary")
st.info("Available models: churn, CLV, sentiment, demand forecasting, product recommendations, and retrieval-augmented chat.")
st.subheader("Navigate")
st.write("Use the sidebar to open any analytics or AI module.")
