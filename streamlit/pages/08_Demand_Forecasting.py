"""Interactive demand-forecasting workspace backed by the existing API."""

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, section_heading, setup_page, show_api_error


setup_page("Demand Forecasting")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def style_figure(figure: go.Figure, *, height: int = 370) -> go.Figure:
    """Apply the platform chart style."""
    figure.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=58, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        title_font=dict(size=17, color="#f8fafc"),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
        legend_title_text="",
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    figure.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    return figure


def normalize_forecast(payload: list[dict]) -> pd.DataFrame:
    """Validate and normalize forecast points returned by FastAPI."""
    frame = pd.DataFrame(payload)
    required = {"date", "predicted_demand"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "predicted_demand"])
    frame = frame[["date", "predicted_demand"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["predicted_demand"] = pd.to_numeric(frame["predicted_demand"], errors="coerce")
    return frame.dropna().sort_values("date").reset_index(drop=True)


page_hero(
    "Planning intelligence",
    "Demand forecasting",
    "Estimate near-term order demand for the full catalog or a specific product and compare planning scenarios.",
)

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []

section_heading("Forecast setup", "Choose a horizon and optionally narrow the baseline to one product.")
with st.form("forecast_form", clear_on_submit=False):
    product_column, horizon_column = st.columns([1.4, 1])
    with product_column:
        product_id = st.text_input(
            "Product ID",
            placeholder="Leave blank for all products",
            help="Enter an exact product ID from the commerce dataset. A missing ID may produce zero demand.",
        ).strip()
    with horizon_column:
        days = st.slider(
            "Forecast horizon",
            min_value=1,
            max_value=90,
            value=14,
            help="Number of future calendar days returned by the API.",
        )
    submitted = st.form_submit_button("Generate forecast", type="primary", width="stretch")

if submitted:
    with st.spinner("Building the demand forecast..."):
        forecast, error = api_request(
            "POST",
            "/forecast/demand",
            json={"product_id": product_id or None, "days": days},
        )
    show_api_error(error)
    if forecast is not None and not error:
        data = normalize_forecast(forecast)
        if data.empty:
            st.warning("The API returned no valid forecast points for this selection.")
        else:
            result = {
                "generated_at": datetime.now().strftime("%H:%M:%S"),
                "product_id": product_id or "All products",
                "days": days,
                "data": data,
            }
            st.session_state.forecast_result = result
            summary = {
                "Generated": result["generated_at"],
                "Product": result["product_id"],
                "Horizon": days,
                "Average demand": float(data["predicted_demand"].mean()),
                "Total demand": float(data["predicted_demand"].sum()),
            }
            st.session_state.forecast_history = (st.session_state.forecast_history + [summary])[-20:]

result = st.session_state.forecast_result
if result is None:
    st.info("Configure the forecast above and select **Generate forecast** to view planning insights.")
else:
    data = result["data"].copy()
    average_demand = float(data["predicted_demand"].mean())
    total_demand = float(data["predicted_demand"].sum())
    peak_row = data.loc[data["predicted_demand"].idxmax()]

    section_heading("Forecast summary", f"Planning outlook for {result['product_id']} across {result['days']} days.")
    metrics = st.columns(5)
    metrics[0].metric("Average daily demand", f"{average_demand:,.2f}")
    metrics[1].metric("Forecast total", f"{total_demand:,.2f}")
    metrics[2].metric("Peak daily demand", f"{peak_row['predicted_demand']:,.2f}")
    metrics[3].metric("Horizon", f"{len(data)} days")
    metrics[4].metric("Confidence", "Not provided")

    if total_demand == 0:
        st.warning("This selection produced zero forecast demand. Verify the product ID and source-data coverage.")
    else:
        st.success(f"Forecast generated successfully for {result['product_id']}.")

    section_heading("Demand outlook", "Explore the daily trajectory and cumulative volume returned by the API.")
    line_chart = go.Figure()
    line_chart.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["predicted_demand"],
            mode="lines+markers",
            name="Predicted demand",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=7, color="#a5b4fc"),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,.12)",
            hovertemplate="%{x|%d %b %Y}<br>Demand: %{y:,.2f}<extra></extra>",
        )
    )
    line_chart.update_layout(title="Daily demand forecast", xaxis_title="Date", yaxis_title="Predicted orders")
    st.plotly_chart(style_figure(line_chart, height=410), width="stretch", config=PLOT_CONFIG)

    data["cumulative_demand"] = data["predicted_demand"].cumsum()
    distribution_column, cumulative_column = st.columns(2)
    with distribution_column:
        histogram = px.histogram(
            data,
            x="predicted_demand",
            nbins=min(15, max(3, len(data))),
            title="Daily-demand distribution",
            labels={"predicted_demand": "Predicted orders"},
            color_discrete_sequence=["#0ea5e9"],
        )
        st.plotly_chart(style_figure(histogram), width="stretch", config=PLOT_CONFIG)
    with cumulative_column:
        cumulative = px.area(
            data,
            x="date",
            y="cumulative_demand",
            title="Cumulative forecast demand",
            labels={"date": "Date", "cumulative_demand": "Cumulative orders"},
            color_discrete_sequence=["#10b981"],
        )
        cumulative.update_traces(line=dict(width=3))
        st.plotly_chart(style_figure(cumulative), width="stretch", config=PLOT_CONFIG)

    section_heading("Forecast table", "Review or download the exact values returned by FastAPI.")
    table_data = data[["date", "predicted_demand", "cumulative_demand"]].copy()
    table_data.columns = ["Date", "Predicted demand", "Cumulative demand"]
    st.dataframe(
        table_data,
        hide_index=True,
        width="stretch",
        column_config={
            "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Predicted demand": st.column_config.NumberColumn(format="%.2f"),
            "Cumulative demand": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download forecast CSV",
        data=table_data.to_csv(index=False).encode("utf-8"),
        file_name=f"demand_forecast_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

history = st.session_state.forecast_history
if history:
    section_heading("Session scenario comparison", "Compare forecasts generated during this browser session.")
    history_frame = pd.DataFrame(history)
    history_frame["Scenario"] = [f"Run {index + 1}" for index in range(len(history_frame))]
    comparison = px.bar(
        history_frame,
        x="Scenario",
        y="Total demand",
        color="Product",
        hover_data=["Generated", "Horizon", "Average demand"],
        title="Forecast totals by session run",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(style_figure(comparison), width="stretch", config=PLOT_CONFIG)
    if st.button("Clear forecast history", help="Clears only this browser session's comparison history."):
        st.session_state.forecast_history = []
        st.rerun()

with st.expander("Method and interpretation notes"):
    st.markdown(
        """
        - The backend currently forecasts from recent observed order volume and returns only future dates and predicted demand.
        - Historical observations, confidence intervals, trend, and seasonal components are not returned, so this page does not invent them.
        - Session comparisons are temporary and disappear when the Streamlit session ends.
        - Treat a zero forecast as a prompt to verify the product identifier and available source data.
        """
    )
