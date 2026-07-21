"""Product recommendation interface."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import setup_page, show_api_error


setup_page("Product Recommendations")
st.title("Product Recommendation System")
customer_id = st.text_input("Customer ID", placeholder="Paste a customer_unique_id")
limit = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get recommendations", type="primary"):
    if not customer_id.strip():
        st.warning("Enter a customer ID first.")
    else:
        with st.spinner("Finding products for this customer..."):
            result, error = api_request("GET", f"/recommendations/{customer_id}?limit={limit}")
        show_api_error(error)
        if result:
            products = result["product_ids"]
            table = pd.DataFrame({"Rank": range(1, len(products) + 1), "Product ID": products})
            table["Rank relevance"] = (1 - (table["Rank"] - 1) / max(len(table), 1)).round(2)
            st.success(f"Found {len(products)} recommendations.")
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(table, x="Product ID", y="Rank relevance", title="Recommendation ranking"), use_container_width=True)
            st.caption("Product images and model confidence scores will appear automatically when the backend supplies product metadata and scores.")
