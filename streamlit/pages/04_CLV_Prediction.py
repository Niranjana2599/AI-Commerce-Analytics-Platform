"""Customer lifetime value assessment backed by the existing FastAPI model."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("CLV Prediction")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def style_figure(figure, *, height=360):
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


def value_tier(predicted_value: float, current_spend: float) -> tuple[str, str]:
    """Create presentation guidance from predicted value growth."""
    growth = predicted_value - current_spend
    if growth > max(current_spend * 0.50, 100):
        return "High growth potential", "Prioritize loyalty benefits, relevant cross-sell offers, and premium service recovery."
    if growth > 0:
        return "Moderate growth potential", "Use personalized engagement and category-relevant recommendations to support the next purchase."
    return "Value retention needed", "Review satisfaction and delivery signals before investing in acquisition-style incentives."


page_hero(
    "Customer value intelligence",
    "Customer lifetime value prediction",
    "Estimate customer value from observed purchase, experience, and commerce behavior using the deployed CLV model.",
)

if "clv_assessment" not in st.session_state:
    st.session_state.clv_assessment = None

section_heading("Customer value profile", "Enter the latest aggregate information available for one customer.")
with st.form("clv_form", clear_on_submit=False):
    purchase_tab, experience_tab, commerce_tab = st.tabs(
        ["Purchase behavior", "Customer experience", "Commerce profile"]
    )

    with purchase_tab:
        first, second = st.columns(2)
        with first:
            orders = st.number_input(
                "Total orders",
                min_value=0,
                value=2,
                step=1,
                help="Number of distinct historical customer orders.",
            )
            spend = st.number_input(
                "Current cumulative spend (USD)",
                min_value=0.0,
                value=350.0,
                step=25.0,
                help="Observed customer spend to date.",
            )
        with second:
            recency = st.number_input(
                "Days since last purchase",
                min_value=0,
                value=30,
                step=1,
                help="Elapsed days since the customer's most recent purchase.",
            )
            average_order_value = spend / max(orders, 1)
            st.metric(
                "Derived average order value",
                f"${average_order_value:,.2f}",
                help="Current spend divided by total orders, with a protected denominator for zero orders.",
            )

    with experience_tab:
        first, second = st.columns(2)
        with first:
            review_score = st.slider(
                "Average review score",
                min_value=1.0,
                max_value=5.0,
                value=4.0,
                step=0.1,
                help="Mean rating across known customer reviews.",
            )
        with second:
            delivery_delay = st.number_input(
                "Average delivery delay (days)",
                min_value=0.0,
                value=1.0,
                step=0.5,
                help="Average number of days delivered after the estimated date.",
            )

    with commerce_tab:
        first, second = st.columns(2)
        with first:
            discount = st.slider(
                "Average discount rate",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
                format="%.0f%%",
                help="Average discount expressed as a fraction of original price.",
            )
            freight = st.number_input(
                "Average freight cost (USD)",
                min_value=0.0,
                value=20.0,
                step=1.0,
                help="Mean shipping charge across the customer's orders.",
            )
        with second:
            payment_method = st.selectbox(
                "Most-used payment method",
                ["credit_card", "boleto", "voucher", "debit_card"],
                help="Most frequently observed payment type.",
            )
            product_diversity = st.number_input(
                "Product category diversity",
                min_value=0,
                value=2,
                step=1,
                help="Number of distinct product categories purchased.",
            )

    action_left, action_right = st.columns([3, 1])
    with action_left:
        submitted = st.form_submit_button(
            "Predict customer value",
            type="primary",
            width="stretch",
            help="Submit the profile to the existing FastAPI CLV endpoint.",
        )
    with action_right:
        reset = st.form_submit_button("Reset result", width="stretch")

if reset:
    st.session_state.clv_assessment = None
    st.rerun()

if submitted:
    if orders == 0 and spend > 0:
        st.warning("Current spend is positive while total orders is zero. Verify these values before relying on the estimate.")

    features = {
        "Total_Orders": orders,
        "Total_Spend": spend,
        "Average_Order_Value": spend / max(orders, 1),
        "Avg_Review_Score": review_score,
        "Avg_Delivery_Delay": delivery_delay,
        "Avg_Discount": discount,
        "Avg_Freight": freight,
        "Payment_Method": payment_method,
        "Product_Diversity": product_diversity,
        "Recency": recency,
    }
    with st.spinner("Estimating customer lifetime value..."):
        result, error = api_request(
            "POST",
            "/predictions/clv",
            json={"features": features},
        )
    show_api_error(error)
    if result:
        st.session_state.clv_assessment = {"result": result, "features": features}

assessment = st.session_state.clv_assessment
if assessment:
    result = assessment["result"]
    features = assessment["features"]
    predicted_value = float(result["prediction"])
    current_spend = float(features["Total_Spend"])
    expected_growth = predicted_value - current_spend
    growth_ratio = expected_growth / current_spend if current_spend else None
    tier, recommendation = value_tier(predicted_value, current_spend)

    section_heading("Value assessment", "Compare observed spend with the model's predicted customer value.")
    result_columns = st.columns(4)
    result_columns[0].metric(
        "Predicted CLV",
        f"${predicted_value:,.2f}",
        help="Raw value returned by the deployed CLV model.",
    )
    result_columns[1].metric("Current spend", f"${current_spend:,.2f}")
    result_columns[2].metric(
        "Expected growth",
        f"${expected_growth:,.2f}",
        delta=f"{growth_ratio:.1%}" if growth_ratio is not None else None,
        help="Predicted CLV minus current spend.",
    )
    result_columns[3].metric(
        "Value growth ratio",
        f"{growth_ratio:.1%}" if growth_ratio is not None else "N/A",
        help="Expected value growth divided by current spend. This is a value proxy, not accounting ROI.",
    )

    if predicted_value < 0:
        st.warning("The model returned a negative value. Treat this as an out-of-range model result and review the inputs/model before use.")
    elif expected_growth > 0:
        st.success(f"{tier}: {recommendation}", icon="💎")
    else:
        st.warning(f"{tier}: {recommendation}", icon="⚠️")

    comparison_column, gauge_column = st.columns([1.25, 1])
    with comparison_column:
        value_frame = pd.DataFrame(
            {
                "Value": ["Current spend", "Predicted CLV"],
                "USD": [current_spend, predicted_value],
            }
        )
        comparison = px.bar(
            value_frame,
            x="Value",
            y="USD",
            color="Value",
            title="Current versus predicted value",
            text_auto="$.3s",
            color_discrete_map={"Current spend": "#0ea5e9", "Predicted CLV": "#8b5cf6"},
        )
        comparison.update_xaxes(title=None)
        comparison.update_traces(hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>")
        st.plotly_chart(style_figure(comparison), width="stretch", config=PLOT_CONFIG)

    with gauge_column:
        safe_current = max(current_spend, 0)
        safe_predicted = max(predicted_value, 0)
        gauge_maximum = max(safe_current, safe_predicted, 1) * 1.25
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=safe_predicted,
                number={"prefix": "$", "valueformat": ",.0f"},
                delta={"reference": safe_current, "relative": True, "valueformat": ".1%"},
                title={"text": "Customer value gauge"},
                gauge={
                    "axis": {"range": [0, gauge_maximum], "tickprefix": "$"},
                    "bar": {"color": "#8b5cf6"},
                    "bgcolor": "rgba(148,163,184,.12)",
                    "borderwidth": 0,
                    "threshold": {"line": {"color": "#0ea5e9", "width": 4}, "value": safe_current},
                },
            )
        )
        st.plotly_chart(style_figure(gauge), width="stretch", config=PLOT_CONFIG)

    section_heading("Customer value profile", "Review the submitted signals used for this prediction.")
    profile_labels = ["Orders", "Spend", "Review", "Recency", "Delivery", "Discount", "Diversity"]
    profile_values = [
        min(float(features["Total_Orders"]) / 10, 1),
        min(float(features["Total_Spend"]) / 2_000, 1),
        float(features["Avg_Review_Score"]) / 5,
        min(float(features["Recency"]) / 365, 1),
        min(float(features["Avg_Delivery_Delay"]) / 15, 1),
        float(features["Avg_Discount"]),
        min(float(features["Product_Diversity"]) / 10, 1),
    ]
    radar = go.Figure(
        go.Scatterpolar(
            r=profile_values + [profile_values[0]],
            theta=profile_labels + [profile_labels[0]],
            fill="toself",
            name="Customer value profile",
            line={"color": "#8b5cf6"},
            fillcolor="rgba(139,92,246,.22)",
        )
    )
    radar.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "range": [0, 1], "showticklabels": False, "gridcolor": "rgba(148,163,184,.22)"},
            "angularaxis": {"gridcolor": "rgba(148,163,184,.22)"},
        },
        showlegend=False,
        title="Normalized value profile",
    )

    summary_rows = [
        ("Total orders", features["Total_Orders"]),
        ("Current spend", f"${features['Total_Spend']:,.2f}"),
        ("Average order value", f"${features['Average_Order_Value']:,.2f}"),
        ("Days since last purchase", features["Recency"]),
        ("Average review score", f"{features['Avg_Review_Score']:.1f} / 5"),
        ("Average delivery delay", f"{features['Avg_Delivery_Delay']:.1f} days"),
        ("Average discount", f"{features['Avg_Discount']:.0%}"),
        ("Average freight", f"${features['Avg_Freight']:,.2f}"),
        ("Payment method", features["Payment_Method"].replace("_", " ").title()),
        ("Product diversity", features["Product_Diversity"]),
    ]
    radar_column, table_column = st.columns([1.15, 1])
    with radar_column:
        st.plotly_chart(style_figure(radar, height=420), width="stretch", config=PLOT_CONFIG)
        st.caption("Radar values use display scaling and are not model feature importance.")
    with table_column:
        st.dataframe(
            pd.DataFrame(summary_rows, columns=["Feature", "Submitted value"]),
            hide_index=True,
            width="stretch",
        )

    unavailable_left, unavailable_right = st.columns(2)
    with unavailable_left:
        with st.expander("CLV distribution"):
            st.info("A population-level CLV distribution requires batch predictions or customer-level API data, which this endpoint does not return.")
    with unavailable_right:
        with st.expander("Historical spending trend"):
            st.info("A spending trend requires dated transaction history. The current endpoint accepts one aggregate customer profile only.")

    with st.expander("How to use this prediction responsibly"):
        st.markdown(
            """
            - Use predicted CLV to prioritize analysis, not to deny service or benefits.
            - Validate unusual or negative predictions before taking action.
            - Compare model performance across customer groups and over time.
            - The displayed growth ratio is not profit, margin, or accounting ROI.
            """
        )
