"""Operational delivery-delay assessment using the existing FastAPI baseline."""

from datetime import date, timedelta
from math import ceil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("Delivery Delay")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
STATE_OPTIONS = [
    "Not specified", "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT",
    "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
]


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


def risk_guidance(risk: str) -> tuple[str, str]:
    """Map API risk labels to operational presentation guidance."""
    if risk == "high":
        return "error", "Escalate fulfilment monitoring and communicate proactively with the customer."
    if risk == "medium":
        return "warning", "Monitor carrier progress and prepare an early status update if movement stalls."
    return "success", "Delivery risk is currently low; continue standard tracking and notifications."


def show_risk(message: str, status: str) -> None:
    if status == "error":
        st.error(message, icon="🔴")
    elif status == "warning":
        st.warning(message, icon="🟠")
    else:
        st.success(message, icon="🟢")


page_hero(
    "Delivery intelligence",
    "Delivery delay risk assessment",
    "Evaluate the planned fulfilment window and communicate a transparent delay-risk estimate before customer impact grows.",
)

if "delivery_assessment" not in st.session_state:
    st.session_state.delivery_assessment = None

section_heading("Order timeline", "Enter the purchase and promised delivery dates for one order.")
with st.form("delivery_form", clear_on_submit=False):
    date_column, destination_column = st.columns([1.35, 1])
    with date_column:
        purchase_date = st.date_input(
            "Order purchase date",
            value=date.today(),
            help="Date the customer placed the order.",
        )
        estimated_date = st.date_input(
            "Estimated delivery date",
            value=date.today() + timedelta(days=10),
            help="Current promised or estimated arrival date.",
        )
    with destination_column:
        state = st.selectbox(
            "Customer state",
            STATE_OPTIONS,
            index=STATE_OPTIONS.index("SP"),
            help="Retained as request context. The current date-based baseline does not use state in its calculation.",
        )
        planned_window = max((estimated_date - purchase_date).days, 0)
        st.metric(
            "Planned fulfilment window",
            f"{planned_window} days",
            help="Calendar days from purchase to estimated delivery.",
        )
        st.caption("Destination is informational until the API returns a regional delay model.")

    action_left, action_right = st.columns([3, 1])
    with action_left:
        submitted = st.form_submit_button(
            "Assess delivery risk",
            type="primary",
            width="stretch",
            help="Submit these dates to the existing delivery-delay endpoint.",
        )
    with action_right:
        reset = st.form_submit_button("Reset result", width="stretch")

if reset:
    st.session_state.delivery_assessment = None
    st.rerun()

if submitted:
    if estimated_date < purchase_date:
        st.error("Estimated delivery date must be on or after the purchase date.")
    else:
        planned_window = (estimated_date - purchase_date).days
        if planned_window <= 1:
            st.warning("The planned window is one day or less. Confirm that both dates are correct.")
        elif planned_window > 60:
            st.warning("The planned window exceeds 60 days. Review whether this is an exceptional or international order.")

        payload = {
            "order_purchase_date": str(purchase_date),
            "estimated_delivery_date": str(estimated_date),
            "customer_state": None if state == "Not specified" else state,
        }
        with st.spinner("Assessing delivery risk..."):
            result, error = api_request(
                "POST",
                "/predictions/delivery-delay",
                json=payload,
            )
        show_api_error(error)
        if result:
            st.session_state.delivery_assessment = {
                "result": result,
                "payload": payload,
            }

assessment = st.session_state.delivery_assessment
if assessment:
    result = assessment["result"]
    payload = assessment["payload"]
    purchase_date = date.fromisoformat(payload["order_purchase_date"])
    estimated_date = date.fromisoformat(payload["estimated_delivery_date"])
    state = payload["customer_state"] or "Not specified"
    planned_window = (estimated_date - purchase_date).days
    predicted_delay = max(float(result["predicted_delay_days"]), 0.0)
    risk = str(result["risk"]).lower()
    adjusted_arrival = estimated_date + timedelta(days=ceil(predicted_delay))
    status, guidance = risk_guidance(risk)

    section_heading("Delivery assessment", "Review the estimate and operational response for this order.")
    result_columns = st.columns(4)
    result_columns[0].metric("Predicted delay", f"{predicted_delay:.1f} days")
    result_columns[1].metric("Delivery status", f"{risk.title()} risk")
    result_columns[2].metric("Planned window", f"{planned_window} days")
    result_columns[3].metric(
        "Adjusted arrival",
        adjusted_arrival.strftime("%d %b %Y"),
        help="Estimated delivery date plus the predicted delay rounded up to a whole calendar day.",
    )
    show_risk(f"{risk.title()} delivery risk: {guidance}", status)

    gauge_column, comparison_column = st.columns([1, 1.25])
    with gauge_column:
        gauge_maximum = max(7.0, predicted_delay * 1.6)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_delay,
                number={"suffix": " days", "valueformat": ".1f"},
                title={"text": "Expected delivery delay"},
                gauge={
                    "axis": {"range": [0, gauge_maximum]},
                    "bar": {"color": "#ef4444" if risk == "high" else "#f59e0b" if risk == "medium" else "#10b981"},
                    "bgcolor": "rgba(148,163,184,.12)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 1], "color": "rgba(16,185,129,.12)"},
                        {"range": [1, 3], "color": "rgba(245,158,11,.12)"},
                        {"range": [3, gauge_maximum], "color": "rgba(239,68,68,.12)"},
                    ],
                    "threshold": {"line": {"color": "#f8fafc", "width": 3}, "value": predicted_delay},
                },
            )
        )
        st.plotly_chart(style_figure(gauge), width="stretch", config=PLOT_CONFIG)

    with comparison_column:
        duration_frame = pd.DataFrame(
            {
                "Measure": ["Planned fulfilment", "Predicted delay"],
                "Days": [planned_window, predicted_delay],
            }
        )
        comparison = px.bar(
            duration_frame,
            x="Measure",
            y="Days",
            color="Measure",
            title="Planned window versus delay estimate",
            text_auto=".2s",
            color_discrete_map={"Planned fulfilment": "#0ea5e9", "Predicted delay": "#f59e0b"},
        )
        comparison.update_xaxes(title=None)
        comparison.update_traces(hovertemplate="%{x}<br>%{y:.1f} days<extra></extra>")
        st.plotly_chart(style_figure(comparison), width="stretch", config=PLOT_CONFIG)

    timeline_data = pd.DataFrame(
        {
            "Milestone": ["Purchase", "Estimated arrival", "Risk-adjusted arrival"],
            "Date": [purchase_date, estimated_date, adjusted_arrival],
            "Position": [1, 1, 1],
            "Color": ["Purchase", "Estimate", "Adjusted"],
        }
    )
    timeline = px.scatter(
        timeline_data,
        x="Date",
        y="Position",
        color="Color",
        text="Milestone",
        title="Delivery timeline",
        color_discrete_map={"Purchase": "#6366f1", "Estimate": "#0ea5e9", "Adjusted": "#f59e0b"},
    )
    timeline.update_traces(marker={"size": 18}, textposition="top center")
    timeline.update_yaxes(visible=False, range=[0.75, 1.3])
    timeline.update_xaxes(title=None)
    timeline.update_layout(showlegend=False)
    st.plotly_chart(style_figure(timeline, height=300), width="stretch", config=PLOT_CONFIG)

    section_heading("Assessment details", "Inputs and calculation boundaries for this result.")
    details = pd.DataFrame(
        [
            ("Purchase date", purchase_date.strftime("%d %b %Y")),
            ("Estimated delivery", estimated_date.strftime("%d %b %Y")),
            ("Risk-adjusted arrival", adjusted_arrival.strftime("%d %b %Y")),
            ("Customer state", state),
            ("Planned window", f"{planned_window} days"),
            ("Predicted delay", f"{predicted_delay:.1f} days"),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(details, hide_index=True, width="stretch")

    unavailable_left, unavailable_middle, unavailable_right = st.columns(3)
    with unavailable_left:
        with st.expander("Delay distribution"):
            st.info("A distribution requires multiple historical or batch predictions; this endpoint returns one order assessment.")
    with unavailable_middle:
        with st.expander("Actual vs predicted"):
            st.info("Actual delivery dates are not returned by the prediction endpoint, so forecast error cannot be shown here.")
    with unavailable_right:
        with st.expander("Regional delay map"):
            st.info("The current baseline does not use state or return regional delay aggregates. A map would imply unsupported precision.")

    with st.expander("How this baseline works"):
        st.markdown(
            """
            - The API estimates delay from the planned fulfilment window between purchase and estimated delivery.
            - Customer state is retained as context but is not currently used in the calculation.
            - The adjusted arrival is a UI interpretation: estimated delivery plus predicted delay rounded up.
            - Use carrier tracking and operational events for final customer communication.
            """
        )
