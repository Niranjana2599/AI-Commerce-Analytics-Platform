"""Customer-specific product recommendation experience using the existing API."""

from urllib.parse import quote

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("Product Recommendations")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(ttl=120, show_spinner=False)
def load_recommendations(customer_id: str, limit: int):
    """Cache identical recommendation requests briefly to reduce API work."""
    encoded_customer_id = quote(customer_id, safe="")
    return api_request(
        "GET",
        f"/recommendations/{encoded_customer_id}?limit={limit}",
        timeout=180,
    )


def style_figure(figure, *, height=420):
    """Apply the platform's accessible Plotly presentation."""
    figure.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=54, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        title_font=dict(size=17, color="#f8fafc"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
    )
    return figure


def short_id(product_id: str, length: int = 18) -> str:
    """Shorten long identifiers for cards without losing the full table value."""
    return product_id if len(product_id) <= length else f"{product_id[:length]}…"


page_hero(
    "Personalized discovery",
    "Product recommendation system",
    "Retrieve an ordered set of unseen products for a customer using the deployed recommendation service.",
)

if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None

section_heading("Recommendation request", "Enter a valid customer identifier and choose the result-set size.")
with st.form("recommendation_form", clear_on_submit=False):
    customer_column, limit_column = st.columns([2.4, 1])
    with customer_column:
        customer_id = st.text_input(
            "Customer ID",
            placeholder="Paste a customer_unique_id",
            help="The identifier is URL-encoded before it is sent to FastAPI.",
        )
    with limit_column:
        limit = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum number of ordered product IDs to request.",
        )

    action_left, action_right = st.columns([3, 1])
    with action_left:
        submitted = st.form_submit_button(
            "Generate recommendations",
            type="primary",
            width="stretch",
        )
    with action_right:
        reset = st.form_submit_button("Reset result", width="stretch")

if reset:
    st.session_state.recommendation_result = None
    st.rerun()

if submitted:
    normalized_customer_id = customer_id.strip()
    if not normalized_customer_id:
        st.warning("Enter a customer ID before requesting recommendations.")
    elif len(normalized_customer_id) > 200:
        st.error("Customer ID is unexpectedly long. Verify the identifier and try again.")
    else:
        with st.spinner("Finding relevant unseen products..."):
            result, error = load_recommendations(normalized_customer_id, limit)
        show_api_error(error)
        if result:
            st.session_state.recommendation_result = {
                "result": result,
                "requested_limit": limit,
            }

assessment = st.session_state.recommendation_result
if assessment:
    result = assessment["result"]
    requested_limit = int(assessment["requested_limit"])
    returned_customer_id = str(result["customer_id"])
    products = [str(product_id) for product_id in result.get("product_ids", [])]

    section_heading("Recommendation set", "Products are ordered by the recommender; rank 1 is the first recommendation.")
    summary_columns = st.columns(3)
    summary_columns[0].metric(
        "Products returned",
        len(products),
        help="Number of product IDs returned by the API.",
    )
    summary_columns[1].metric("Requested limit", requested_limit)
    summary_columns[2].metric(
        "Result coverage",
        f"{len(products) / requested_limit:.0%}" if requested_limit else "N/A",
        help="Returned count divided by the requested maximum. This is not recommendation accuracy.",
    )

    if not products:
        st.warning("The recommendation service returned no products for this customer.")
        st.stop()

    st.success(f"Generated {len(products)} ordered recommendations for customer {returned_customer_id}.", icon="🎯")

    section_heading("Recommended products", "Scan the ranked cards or use the full identifier table below.")
    for row_start in range(0, len(products), 4):
        columns = st.columns(4)
        for offset, product_id in enumerate(products[row_start:row_start + 4]):
            rank = row_start + offset + 1
            with columns[offset]:
                st.markdown(
                    f"""
                    <div class="module-card">
                        <div class="eyebrow">Recommendation #{rank}</div>
                        <h3>📦 {short_id(product_id)}</h3>
                        <p>Product ID</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    table = pd.DataFrame(
        {
            "Rank": range(1, len(products) + 1),
            "Product ID": products,
        }
    )
    table["Rank order"] = table["Rank"].map(lambda value: f"#{value}")

    chart_column, table_column = st.columns([1.2, 1])
    with chart_column:
        chart_frame = table.sort_values("Rank", ascending=False)
        rank_chart = px.bar(
            chart_frame,
            x="Rank",
            y="Product ID",
            orientation="h",
            title="Recommendation order",
            color="Rank",
            color_continuous_scale=["#6366f1", "#0ea5e9"],
            labels={"Rank": "Rank position"},
        )
        rank_chart.update_xaxes(autorange="reversed", dtick=1)
        rank_chart.update_traces(
            hovertemplate="Rank %{x}<br>Product %{y}<extra></extra>",
        )
        rank_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            style_figure(rank_chart, height=max(390, len(products) * 35)),
            width="stretch",
            config=PLOT_CONFIG,
        )
        st.caption("Lower rank is better. Rank position is not a probability or confidence score.")

    with table_column:
        st.dataframe(
            table[["Rank order", "Product ID"]],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download recommendations",
            data=table[["Rank", "Product ID"]].to_csv(index=False).encode("utf-8"),
            file_name=f"recommendations_{returned_customer_id}.csv",
            mime="text/csv",
            width="stretch",
        )

    section_heading("Recommendation metadata", "Capabilities depend on fields supplied by the API contract.")
    unavailable_left, unavailable_middle, unavailable_right = st.columns(3)
    with unavailable_left:
        with st.expander("Product images and details"):
            st.info("The endpoint returns product IDs only; image URLs, names, prices, and descriptions are not available.")
    with unavailable_middle:
        with st.expander("Category distribution"):
            st.info("Product categories are not returned, so a category chart cannot be calculated without inventing metadata.")
    with unavailable_right:
        with st.expander("Confidence and similarity network"):
            st.info("The endpoint does not return recommendation scores or item similarity edges. Rank is shown without being presented as confidence.")

    with st.expander("How to interpret these recommendations"):
        st.markdown(
            """
            - Products are presented in the order returned by the recommendation service.
            - The service prefers unseen products when customer purchase history is available.
            - A full requested list is not guaranteed when the candidate catalog is limited.
            - Evaluate recommendation quality with ranking metrics and online outcomes, not rank position alone.
            """
        )
