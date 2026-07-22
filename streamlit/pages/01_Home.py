"""Production-style commerce overview backed by the existing FastAPI APIs."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("Home Dashboard")

ACTIVE_MODELS = 6
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(ttl=60, show_spinner=False)
def load_dashboard_data():
    """Cache read-only API calls briefly to avoid unnecessary page reruns."""
    health, health_error = api_request("GET", "/health")
    metrics, metrics_error = api_request("GET", "/analytics/customer-metrics")
    return health, health_error, metrics, metrics_error


def style_figure(figure, *, height=360):
    """Apply a consistent accessible Plotly presentation."""
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=54, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        title_font=dict(size=17, color="#f8fafc"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
    )
    return figure


page_hero(
    "Executive overview",
    "Commerce intelligence at a glance",
    "Monitor business performance, model readiness, and API health from one responsive workspace.",
)

toolbar_left, toolbar_right = st.columns([5, 1])
with toolbar_left:
    st.caption("Live FastAPI metrics · cached for 60 seconds")
with toolbar_right:
    if st.button("↻ Refresh", width="stretch", help="Clear the cache and fetch fresh API data."):
        load_dashboard_data.clear()
        st.rerun()

with st.spinner("Loading platform overview..."):
    health, health_error, metrics, metrics_error = load_dashboard_data()

api_online = bool(health and health.get("status") == "ok")
if health_error:
    st.warning("The platform health check is unavailable. Business metrics may still be accessible.")
if metrics_error:
    show_api_error(metrics_error)

section_heading("Performance snapshot", "Core commerce and platform-readiness indicators.")
kpi_columns = st.columns(6)
if metrics:
    total_revenue = float(metrics["total_revenue"])
    total_orders = int(metrics["total_orders"])
    total_customers = int(metrics["total_customers"])
    average_order_value = float(metrics["average_order_value"])
    kpis = [
        ("Customers", f"{total_customers:,}", "Unique customer base"),
        ("Orders", f"{total_orders:,}", "Completed commerce records"),
        ("Revenue", f"${total_revenue:,.0f}", "Gross recorded revenue"),
        ("Average order", f"${average_order_value:,.2f}", "Revenue per order"),
        ("Active models", str(ACTIVE_MODELS), "Prediction capabilities"),
        ("API status", "Online" if api_online else "Offline", health.get("environment", "Unavailable") if health else "Check backend"),
    ]
    for column, (label, value, help_text) in zip(kpi_columns, kpis):
        column.metric(label, value, help=help_text)
else:
    for column, label in zip(kpi_columns, ["Customers", "Orders", "Revenue", "Average order", "Active models", "API status"]):
        column.metric(label, "—", help="Metric unavailable until the FastAPI service responds.")

if metrics:
    section_heading("Business efficiency", "Derived ratios use only values returned by the customer-metrics endpoint.")
    left, right = st.columns([1.55, 1])
    revenue_per_customer = total_revenue / total_customers if total_customers else 0
    orders_per_customer = total_orders / total_customers if total_customers else 0

    with left:
        economics = pd.DataFrame(
            {"Metric": ["Average order value", "Revenue per customer"], "Value": [average_order_value, revenue_per_customer]}
        )
        figure = px.bar(
            economics, x="Metric", y="Value", color="Metric", text_auto="$.2s",
            color_discrete_sequence=["#6366f1", "#22c55e"], title="Unit economics", labels={"Value": "USD"},
        )
        figure.update_traces(marker_line_width=0, hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>")
        figure.update_xaxes(title=None)
        st.plotly_chart(style_figure(figure), width="stretch", config=PLOT_CONFIG)

    with right:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number", value=orders_per_customer,
                number={"suffix": " orders", "valueformat": ".2f"}, title={"text": "Orders per customer"},
                gauge={
                    "axis": {"range": [0, max(3, orders_per_customer * 1.5)]}, "bar": {"color": "#6366f1"},
                    "bgcolor": "rgba(148,163,184,.12)", "borderwidth": 0,
                },
            )
        )
        st.plotly_chart(style_figure(gauge), width="stretch", config=PLOT_CONFIG)

    with st.expander("Why aren't monthly, category, state, and product charts shown?"):
        st.info(
            "The current FastAPI contract returns aggregate customer metrics only. Monthly revenue, category, product, "
            "state, segment, and order-status series are not exposed. This page avoids fabricating them and keeps all endpoints unchanged."
        )

section_heading("Analytics workspace", "Open a focused workflow from the sidebar navigation.")
modules = [
    ("👥", "Customer analytics", "Explore customer performance and segmentation."),
    ("⚠️", "Churn prediction", "Identify customers who may need retention action."),
    ("💎", "Lifetime value", "Estimate future customer value and growth."),
    ("🚚", "Delivery intelligence", "Assess delivery-delay risk and timing."),
    ("🎯", "Recommendations", "Generate relevant product suggestions."),
    ("💬", "Review sentiment", "Understand customer-review tone."),
    ("📈", "Demand forecasting", "Plan demand over a selected horizon."),
    ("🤖", "RAG assistant", "Ask grounded questions about commerce data."),
]
for row_start in range(0, len(modules), 4):
    columns = st.columns(4)
    for column, (icon, name, description) in zip(columns, modules[row_start:row_start + 4]):
        with column:
            st.markdown(
                f'<div class="module-card"><div>{icon}</div><h3>{name}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

if api_online and metrics:
    st.success("Dashboard synchronized with the FastAPI service.", icon="✅")
elif metrics:
    st.warning("Business metrics loaded, but the health check is unavailable.", icon="⚠️")
