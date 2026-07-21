"""Interactive customer lifetime value prediction page."""

import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("CLV Prediction")
st.title("Customer Lifetime Value (CLV) Prediction")
with st.form("clv_form"):
    orders = st.number_input("Total orders", min_value=0, value=2)
    spend = st.number_input("Total spend", min_value=0.0, value=350.0)
    frequency = st.number_input("Purchase frequency", min_value=0.0, value=2.0)
    recency = st.number_input("Days since last purchase", min_value=0, value=30)
    review_score = st.slider("Average review score", 1.0, 5.0, 4.0)
    delivery_delay = st.number_input("Average delivery delay (days)", min_value=0.0, value=1.0)
    discount = st.slider("Average discount rate", 0.0, 1.0, 0.10)
    freight = st.number_input("Average freight", min_value=0.0, value=20.0)
    payment_method = st.selectbox("Most-used payment method", ["credit_card", "boleto", "voucher", "debit_card"])
    product_diversity = st.number_input("Product category diversity", min_value=0, value=2)
    submitted = st.form_submit_button("Predict CLV")

if submitted:
    features = {
        "Total_Orders": orders, "Total_Spend": spend,
        "Average_Order_Value": spend / max(orders, 1), "Recency": recency,
        "Frequency": frequency, "Monetary": spend,
        "Avg_Review_Score": review_score, "Avg_Delivery_Delay": delivery_delay,
        "Avg_Discount": discount, "Avg_Freight": freight,
        "Payment_Method": payment_method, "Product_Diversity": product_diversity,
    }
    with st.spinner("Estimating customer lifetime value..."):
        result, error = api_request("POST", "/predictions/clv", json={"features": features})
    show_api_error(error)
    if result:
        value = float(result["prediction"])
        st.success("CLV prediction completed.")
        st.metric("Predicted customer lifetime value", f"${value:,.2f}")
        st.plotly_chart(px.bar(x=["Current spend", "Predicted CLV"], y=[spend, value], title="Current versus predicted value"), use_container_width=True)
