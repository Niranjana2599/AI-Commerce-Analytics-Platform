"""Actionable customer churn-risk assessment backed by the existing API."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("Churn Prediction")

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


def risk_details(probability: float) -> tuple[str, str, str]:
    """Translate a probability into consistent presentation guidance."""
    if probability >= 0.70:
        return (
            "High risk",
            "error",
            "Prioritize personal outreach, investigate recent service issues, and consider a targeted retention offer.",
        )
    if probability >= 0.40:
        return (
            "Medium risk",
            "warning",
            "Schedule proactive engagement, review delivery and satisfaction signals, and monitor the next purchase window.",
        )
    return (
        "Low risk",
        "success",
        "Maintain regular engagement and reinforce loyalty without unnecessary discounting.",
    )


def show_status(message: str, status: str) -> None:
    """Render a risk-aware status message."""
    if status == "error":
        st.error(message, icon="🔴")
    elif status == "warning":
        st.warning(message, icon="🟠")
    else:
        st.success(message, icon="🟢")


page_hero(
    "Predictive retention",
    "Customer churn risk assessment",
    "Combine purchase, satisfaction, delivery, and product signals to support timely retention decisions.",
)

if "churn_assessment" not in st.session_state:
    st.session_state.churn_assessment = None

section_heading("Customer profile", "Provide the latest known aggregate behavior for one customer.")
with st.form("churn_form", clear_on_submit=False):
    purchase_tab, experience_tab, commerce_tab = st.tabs(
        ["Purchase behavior", "Customer experience", "Commerce profile"]
    )

    with purchase_tab:
        left, right = st.columns(2)
        with left:
            total_orders = st.number_input(
                "Total orders",
                min_value=0,
                value=2,
                step=1,
                help="Number of distinct historical orders for this customer.",
            )
        with right:
            total_spend = st.number_input(
                "Total spend (USD)",
                min_value=0.0,
                value=350.0,
                step=25.0,
                help="Cumulative observed customer spend.",
            )
        st.caption(
            f"Derived average order value: ${total_spend / max(total_orders, 1):,.2f}. "
            "The denominator is protected when total orders is zero."
        )

    with experience_tab:
        left, right = st.columns(2)
        with left:
            review_score = st.slider(
                "Average review score",
                min_value=1.0,
                max_value=5.0,
                value=4.0,
                step=0.1,
                help="Mean rating across known customer reviews.",
            )
        with right:
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
                help="Average discount as a fraction of the original price.",
            )
            freight = st.number_input(
                "Average freight cost (USD)",
                min_value=0.0,
                value=20.0,
                step=1.0,
                help="Average shipping charge associated with the customer's orders.",
            )
        with second:
            payment_method = st.selectbox(
                "Most-used payment method",
                ["credit_card", "boleto", "voucher", "debit_card"],
                help="The customer's most frequently observed payment type.",
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
            "Assess churn risk",
            type="primary",
            width="stretch",
            help="Submit this profile to the existing FastAPI churn endpoint.",
        )
    with action_right:
        reset = st.form_submit_button("Reset result", width="stretch")

if reset:
    st.session_state.churn_assessment = None
    st.rerun()

if submitted:
    if total_orders == 0 and total_spend > 0:
        st.warning("Total spend is positive while total orders is zero. Verify the purchase values before relying on the result.")

    average_order_value = total_spend / max(total_orders, 1)
    features = {
        "Total_Orders": total_orders,
        "Total_Spend": total_spend,
        "Average_Order_Value": average_order_value,
        "Frequency": total_orders,
        "Monetary": total_spend,
        "Avg_Review_Score": review_score,
        "Avg_Delivery_Delay": delivery_delay,
        "Avg_Discount": discount,
        "Avg_Freight": freight,
        "Payment_Method": payment_method,
        "Product_Diversity": product_diversity,
    }
    with st.spinner("Assessing customer retention risk..."):
        result, error = api_request(
            "POST",
            "/predictions/churn",
            json={"features": features},
        )
    show_api_error(error)
    if result:
        st.session_state.churn_assessment = {"result": result, "features": features}

assessment = st.session_state.churn_assessment
if assessment:
    result = assessment["result"]
    features = assessment["features"]
    raw_prediction = result["prediction"]
    probability = result.get("probability")

    section_heading("Risk assessment", "Interpret the model response alongside the submitted customer profile.")
    if probability is not None:
        probability = min(max(float(probability), 0.0), 1.0)
        retention_probability = 1.0 - probability
        confidence = max(probability, retention_probability)
        risk_label, status, recommendation = risk_details(probability)

        result_columns = st.columns(4)
        result_columns[0].metric("Prediction", "Churn" if int(raw_prediction) == 1 else "Retain")
        result_columns[1].metric("Churn probability", f"{probability:.1%}")
        result_columns[2].metric("Risk band", risk_label)
        result_columns[3].metric(
            "Class confidence",
            f"{confidence:.1%}",
            help="The larger of churn and retention probability. This is not a guarantee or calibration score.",
        )
        show_status(f"{risk_label}: {recommendation}", status)

        gauge_column, comparison_column = st.columns([1, 1.25])
        with gauge_column:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "valueformat": ".1f"},
                    title={"text": "Churn probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444" if probability >= 0.7 else "#f59e0b" if probability >= 0.4 else "#10b981"},
                        "bgcolor": "rgba(148,163,184,.12)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "rgba(16,185,129,.12)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,.12)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,.12)"},
                        ],
                        "threshold": {"line": {"color": "#f8fafc", "width": 3}, "value": probability * 100},
                    },
                )
            )
            st.plotly_chart(style_figure(gauge), width="stretch", config=PLOT_CONFIG)

        with comparison_column:
            comparison = pd.DataFrame(
                {
                    "Outcome": ["Churn", "Retention"],
                    "Probability": [probability, retention_probability],
                }
            )
            risk_chart = px.bar(
                comparison,
                x="Outcome",
                y="Probability",
                color="Outcome",
                title="Outcome probability comparison",
                text_auto=".1%",
                color_discrete_map={"Churn": "#ef4444", "Retention": "#10b981"},
            )
            risk_chart.update_yaxes(tickformat=".0%", range=[0, 1], title=None)
            risk_chart.update_xaxes(title=None)
            risk_chart.update_traces(hovertemplate="%{x}<br>%{y:.1%}<extra></extra>")
            st.plotly_chart(style_figure(risk_chart), width="stretch", config=PLOT_CONFIG)
    else:
        predicted_label = "Churn" if str(raw_prediction) in {"1", "True", "true"} else "Retain"
        st.warning(
            f"The model returned a class prediction ({predicted_label}) without probability. "
            "Risk bands and confidence charts require probability output."
        )

    section_heading("Customer profile summary", "Input values used for this specific model request.")
    profile_labels = ["Order volume", "Spend", "Review score", "Delivery", "Discount", "Freight", "Diversity"]
    profile_values = [
        min(float(features["Total_Orders"]) / 10, 1),
        min(float(features["Total_Spend"]) / 2_000, 1),
        float(features["Avg_Review_Score"]) / 5,
        min(float(features["Avg_Delivery_Delay"]) / 15, 1),
        float(features["Avg_Discount"]),
        min(float(features["Avg_Freight"]) / 100, 1),
        min(float(features["Product_Diversity"]) / 10, 1),
    ]
    radar = go.Figure(
        go.Scatterpolar(
            r=profile_values + [profile_values[0]],
            theta=profile_labels + [profile_labels[0]],
            fill="toself",
            name="Customer profile",
            line={"color": "#6366f1"},
            fillcolor="rgba(99,102,241,.22)",
        )
    )
    radar.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "range": [0, 1], "showticklabels": False, "gridcolor": "rgba(148,163,184,.22)"},
            "angularaxis": {"gridcolor": "rgba(148,163,184,.22)"},
        },
        showlegend=False,
        title="Normalized customer profile",
    )

    summary_rows = [
        ("Total orders", features["Total_Orders"]),
        ("Total spend", f"${features['Total_Spend']:,.2f}"),
        ("Average order value", f"${features['Average_Order_Value']:,.2f}"),
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
        st.caption("Radar values are scaled to display ranges for comparison; they are not model feature importance.")
    with table_column:
        st.dataframe(
            pd.DataFrame(summary_rows, columns=["Feature", "Submitted value"]),
            hide_index=True,
            width="stretch",
        )
        st.info(
            "Model feature importance is not returned by the current API. Displaying estimated importance here would be misleading."
        )

    with st.expander("How to use this assessment responsibly"):
        st.markdown(
            """
            - Treat the score as decision support, not a definitive statement about a customer.
            - Confirm unusual input values before taking retention action.
            - Avoid using protected personal characteristics or sensitive data.
            - Monitor model performance and probability calibration as customer behavior changes.
            """
        )
