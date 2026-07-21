"""Delivery delay prediction interface."""

from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("Delivery Delay")
st.title("Delivery Delay Prediction")
with st.form("delivery_form"):
    purchase_date = st.date_input("Order purchase date", value=date.today())
    estimated_date = st.date_input("Estimated delivery date", value=date.today() + timedelta(days=10))
    state = st.text_input("Customer state (optional)", value="SP")
    submitted = st.form_submit_button("Assess delivery risk")

if submitted:
    payload = {"order_purchase_date": str(purchase_date), "estimated_delivery_date": str(estimated_date), "customer_state": state or None}
    with st.spinner("Assessing delivery risk..."):
        result, error = api_request("POST", "/predictions/delivery-delay", json=payload)
    show_api_error(error)
    if result:
        st.success(f"Delivery risk: {result['risk'].title()}")
        st.metric("Predicted delay", f"{result['predicted_delay_days']:.1f} days")
        st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=result["predicted_delay_days"], title={"text": "Expected delay (days)"})), use_container_width=True)
