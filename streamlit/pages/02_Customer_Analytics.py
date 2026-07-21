"""Customer analytics dashboard backed by FastAPI commerce metrics."""

import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("Customer Analytics")
st.title("Customer Analytics Dashboard")
with st.spinner("Loading customer analytics..."):
    metrics, error = api_request("GET", "/analytics/customer-metrics")
show_api_error(error)
if metrics:
    left, right = st.columns(2)
    left.metric("Customers", f"{metrics['total_customers']:,}")
    right.metric("Average order value", f"${metrics['average_order_value']:,.2f}")
    funnel = px.funnel(y=["Customers", "Orders"], x=[metrics["total_customers"], metrics["total_orders"]], title="Customer-to-order journey")
    st.plotly_chart(funnel, use_container_width=True)

tab_names = ["RFM Analysis", "Segmentation", "Cohorts", "Personas", "ABC Analysis"]
for tab, tab_name, description in zip(st.tabs(tab_names), tab_names, [
    "Recency, frequency, and monetary scores.", "Customer groups by behaviour and value.",
    "Retention by first-purchase cohort.", "Actionable customer profiles.", "Revenue contribution tiers.",
]):
    with tab:
        st.subheader(tab_name)
        st.write(description)
        st.info("This view is ready for a granular customer-analytics API endpoint. The current backend exposes live platform KPIs only.")
