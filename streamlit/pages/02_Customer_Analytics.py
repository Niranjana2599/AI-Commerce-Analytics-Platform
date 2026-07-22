"""Interactive customer-economics view backed by existing FastAPI metrics."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("Customer Analytics")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(ttl=60, show_spinner=False)
def load_customer_metrics():
    """Cache the aggregate analytics response to avoid calls on every rerun."""
    return api_request("GET", "/analytics/customer-metrics")


def style_figure(figure, *, height=360):
    """Apply the platform's accessible Plotly presentation."""
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


def unavailable_analysis(title: str, purpose: str, required_fields: str) -> None:
    """Render an honest empty state for unsupported aggregate analyses."""
    st.markdown(f"#### {title}")
    st.write(purpose)
    st.info(
        "This analysis requires customer-level records that are not returned by the current FastAPI contract. "
        "No estimated or synthetic values are displayed."
    )
    with st.expander("Required API data"):
        st.code(required_fields, language=None)


page_hero(
    "Customer intelligence",
    "Understand customer value and purchasing intensity",
    "Explore live customer, order, and revenue economics while keeping unsupported granular analyses transparent.",
)

toolbar_left, toolbar_right = st.columns([5, 1])
with toolbar_left:
    st.caption("Live FastAPI aggregates · cached for 60 seconds")
with toolbar_right:
    if st.button(
        "↻ Refresh",
        width="stretch",
        help="Clear cached values and request fresh customer metrics.",
    ):
        load_customer_metrics.clear()
        st.rerun()

with st.spinner("Loading customer analytics..."):
    metrics, error = load_customer_metrics()

if error:
    show_api_error(error)

if not metrics:
    st.warning("Customer analytics are unavailable until the FastAPI metrics service responds.")
    st.stop()

total_revenue = float(metrics["total_revenue"])
total_orders = int(metrics["total_orders"])
total_customers = int(metrics["total_customers"])
average_order_value = float(metrics["average_order_value"])
orders_per_customer = total_orders / total_customers if total_customers else 0
revenue_per_customer = total_revenue / total_customers if total_customers else 0

section_heading("Customer snapshot", "Live scale, spend, and purchase-intensity indicators.")
kpi_columns = st.columns(4)
kpi_columns[0].metric(
    "Total customers",
    f"{total_customers:,}",
    help="Distinct customers reported by the commerce analytics API.",
)
kpi_columns[1].metric(
    "Orders per customer",
    f"{orders_per_customer:.2f}",
    help="Total orders divided by distinct customers; this is an aggregate intensity measure, not a repeat-customer rate.",
)
kpi_columns[2].metric(
    "Revenue per customer",
    f"${revenue_per_customer:,.2f}",
    help="Total revenue divided by distinct customers.",
)
kpi_columns[3].metric(
    "Average order value",
    f"${average_order_value:,.2f}",
    help="Total revenue divided by total orders.",
)

section_heading("Customer economics", "Compare audience scale, order volume, and customer-level value ratios.")
left, right = st.columns([1.45, 1])

with left:
    funnel_data = pd.DataFrame(
        {
            "Stage": ["Distinct customers", "Total orders"],
            "Count": [total_customers, total_orders],
        }
    )
    funnel = px.funnel(
        funnel_data,
        y="Stage",
        x="Count",
        title="Customer-to-order volume",
        color="Stage",
        color_discrete_sequence=["#6366f1", "#22c55e"],
    )
    funnel.update_traces(
        texttemplate="%{value:,.0f}",
        hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
    )
    st.plotly_chart(style_figure(funnel), width="stretch", config=PLOT_CONFIG)

with right:
    gauge_maximum = max(3.0, orders_per_customer * 1.6)
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=orders_per_customer,
            number={"suffix": " orders", "valueformat": ".2f"},
            title={"text": "Average purchase intensity"},
            gauge={
                "axis": {"range": [0, gauge_maximum]},
                "bar": {"color": "#6366f1", "thickness": 0.75},
                "bgcolor": "rgba(148,163,184,.12)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, gauge_maximum / 3], "color": "rgba(245,158,11,.12)"},
                    {"range": [gauge_maximum / 3, gauge_maximum], "color": "rgba(16,185,129,.10)"},
                ],
            },
        )
    )
    st.plotly_chart(style_figure(gauge), width="stretch", config=PLOT_CONFIG)

economics = pd.DataFrame(
    {
        "Metric": ["Average order value", "Revenue per customer"],
        "USD": [average_order_value, revenue_per_customer],
    }
)
economics_chart = px.bar(
    economics,
    x="Metric",
    y="USD",
    color="Metric",
    title="Order value vs customer value",
    color_discrete_sequence=["#0ea5e9", "#8b5cf6"],
    text_auto="$.3s",
)
economics_chart.update_traces(
    marker_line_width=0,
    hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
)
economics_chart.update_xaxes(title=None)
st.plotly_chart(style_figure(economics_chart, height=330), width="stretch", config=PLOT_CONFIG)

section_heading("Advanced customer analysis", "Granular workflows are retained without presenting fabricated results.")
tabs = st.tabs(["RFM", "Segments", "Cohorts", "Personas", "ABC value"])
with tabs[0]:
    unavailable_analysis(
        "RFM distribution",
        "Score customers by recency, purchase frequency, and monetary value.",
        "customer_id, order_date, order_id, order_value",
    )
with tabs[1]:
    unavailable_analysis(
        "Customer segments",
        "Compare behavioural and value-based customer groups.",
        "customer_id, recency, frequency, monetary_value, segment",
    )
with tabs[2]:
    unavailable_analysis(
        "Cohort retention",
        "Measure repeat activity by customers' first-purchase period.",
        "customer_id, first_purchase_date, order_date",
    )
with tabs[3]:
    unavailable_analysis(
        "Customer personas",
        "Translate segment characteristics into actionable customer profiles.",
        "customer_id, segment, category_preferences, location, order_history",
    )
with tabs[4]:
    unavailable_analysis(
        "ABC customer value",
        "Group customers by cumulative revenue contribution.",
        "customer_id, customer_revenue",
    )

with st.expander("Metric definitions and limitations"):
    st.markdown(
        """
        - **Orders per customer** is an aggregate ratio and does not prove that the same customer placed multiple orders.
        - **Revenue per customer** is historical observed revenue, not predicted customer lifetime value.
        - **Average order value** is revenue divided by orders and may differ from item-level average price.
        - Geographic, RFM, cohort, and distribution charts require granular API data that is not currently exposed.
        """
    )

st.success("Customer aggregates synchronized with FastAPI.", icon="✅")
