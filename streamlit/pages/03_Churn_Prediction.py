"""Interactive customer churn prediction page."""

import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("Churn Prediction")
st.title("Customer Churn Prediction")
st.write("Enter known customer features, then ask the trained churn model for a risk prediction.")

with st.form("churn_form"):
    total_orders = st.number_input("Total orders", min_value=0, value=2)
    total_spend = st.number_input("Total spend", min_value=0.0, value=350.0)
    review_score = st.slider("Average review score", 1.0, 5.0, 4.0)
    delivery_delay = st.number_input("Average delivery delay (days)", min_value=0.0, value=1.0)
    discount = st.slider("Average discount rate", 0.0, 1.0, 0.10)
    freight = st.number_input("Average freight", min_value=0.0, value=20.0)
    payment_method = st.selectbox("Most-used payment method", ["credit_card", "boleto", "voucher", "debit_card"])
    product_diversity = st.number_input("Product category diversity", min_value=0, value=2)
    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    average_order_value = total_spend / max(total_orders, 1)
    features = {
        "Total_Orders": total_orders, "Total_Spend": total_spend,
        "Average_Order_Value": average_order_value, "Frequency": total_orders,
        "Monetary": total_spend, "Avg_Review_Score": review_score,
        "Avg_Delivery_Delay": delivery_delay, "Avg_Discount": discount,
        "Avg_Freight": freight, "Payment_Method": payment_method,
        "Product_Diversity": product_diversity,
    }
    with st.spinner("Running churn prediction..."):
        result, error = api_request("POST", "/predictions/churn", json={"features": features})
    show_api_error(error)
    if result:
        st.success(f"Churn prediction: {result['prediction']}")
        if result["probability"] is not None:
            probability = result["probability"]
            st.metric("Churn confidence", f"{probability:.1%}")
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, title={"text": "Churn risk (%)"})), use_container_width=True)
