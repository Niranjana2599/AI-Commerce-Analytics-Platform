"""Demand forecasting dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("Demand Forecasting")
st.title("Demand Forecasting")
with st.form("forecast_form"):
    product_id = st.text_input("Product ID (optional)")
    days = st.slider("Forecast days", min_value=1, max_value=90, value=14)
    submitted = st.form_submit_button("Generate forecast")

if submitted:
    with st.spinner("Building demand forecast..."):
        forecast, error = api_request("POST", "/forecast/demand", json={"product_id": product_id or None, "days": days})
    show_api_error(error)
    if forecast:
        data = pd.DataFrame(forecast)
        data["date"] = pd.to_datetime(data["date"])
        st.success("Forecast generated.")
        st.metric("Average predicted daily demand", f"{data['predicted_demand'].mean():,.2f}")
        st.plotly_chart(px.line(data, x="date", y="predicted_demand", markers=True, title="Predicted demand"), use_container_width=True)
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.caption("Historical-demand plotting can be enabled when the API returns historical series alongside the forecast.")
