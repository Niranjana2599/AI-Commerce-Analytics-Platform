"""Shared natural-language entry point for the existing agent API."""

import streamlit as st

from utils.api_client import api_request


def render_agent_entry(prefix: str, *, show_optional_context: bool = False) -> None:
    """Ask the existing supervisor and present only concise execution evidence."""
    question_key = f"{prefix}_question"
    with st.form(f"{prefix}_business_question_form"):
        question = st.text_area(
            "Ask a business question",
            key=question_key,
            placeholder="How can we increase revenue during the upcoming festival?",
            max_chars=2_000,
            height=105,
            label_visibility="collapsed",
        )
        customer_id = None
        review_text = None
        if show_optional_context:
            with st.expander("Optional context for a known customer or review"):
                customer_id = st.text_input("Customer ID", max_chars=128)
                review_text = st.text_area("Review text", max_chars=5_000, height=70)
        submitted = st.form_submit_button("Analyze", type="primary", width="stretch")

    if submitted:
        if not question.strip():
            st.warning("Enter a business question to continue.")
        else:
            with st.spinner("The supervisor is selecting and checking the required tools…"):
                result, error = api_request(
                    "POST",
                    "/agent/ask",
                    json={
                        "question": question.strip(),
                        "customer_id": (customer_id or "").strip() or None,
                        "review_text": (review_text or "").strip() or None,
                    },
                    timeout=120,
                )
            if error:
                st.error(error)
            elif result:
                _show_agent_result(result)


def _show_agent_result(result: dict) -> None:
    st.markdown("### Answer")
    st.write(result.get("answer") or "No answer returned.")

    evaluation = result.get("evaluation") or {}
    evidence = result.get("evidence") or []
    successful = evaluation.get(
        "successful_results",
        sum(item.get("status") in {"success", "fallback"} for item in evidence),
    )
    intent = result.get("intent", "unknown")
    status = "Blocked" if intent == "blocked" else "Completed" if evaluation.get("task_success") else "Limited"
    fallbacks = [
        item.get("fallback_tool") or item.get("tool")
        for item in evidence if item.get("fallback_used")
    ]
    columns = st.columns(5)
    columns[0].metric("Intent", str(intent).replace("_", " ").title())
    columns[1].metric("Status", status)
    columns[2].metric("Evidence", f"{successful} tool results")
    columns[3].metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
    columns[4].metric("Fallback", ", ".join(fallbacks) if fallbacks else "None")

    if evidence:
        tools = []
        for item in evidence:
            mark = "✓" if item.get("status") == "success" else "⚠"
            label = _tool_label(item.get("tool", "tool"))
            if item.get("fallback_used"):
                label += " (fallback)"
            tools.append(f"{mark} {label}")
        st.markdown("**Tools used**  \n" + " → ".join(tools))
    if result.get("sources"):
        st.caption("Knowledge sources: " + ", ".join(sorted(set(result["sources"]))))

    with st.expander("How the agent analyzed it"):
        for index, item in enumerate(evidence, start=1):
            label = _tool_label(item.get("tool", "tool"))
            step_status = "Fallback" if item.get("fallback_used") else item.get("status", "unknown").title()
            st.markdown(
                f"**{index}. {label}** · {step_status} · "
                f"{item.get('elapsed_ms', 0):.0f} ms · attempt {item.get('attempt', 1)}"
            )
        if result.get("anomalies"):
            st.caption("Execution controls: " + ", ".join(
                str(item.get("anomaly_type", "recorded")).replace("_", " ")
                for item in result["anomalies"]
            ))
        if result.get("errors"):
            st.caption("Some requested evidence was unavailable; the answer may be partial.")


def _tool_label(tool: str) -> str:
    """Make registry identifiers readable without changing the API contract."""
    labels = {
        "customer_cohort_tool": "Customer cohort",
        "customer_rfm_tool": "RFM analysis",
        "clv_prediction_tool": "CLV prediction",
        "historical_value_tool": "Historical value proxy",
        "churn_prediction_tool": "Churn prediction",
        "rfm_risk_tool": "RFM risk proxy",
        "recommendation_tool": "Recommendation",
        "business_analytics_tool": "Business analytics",
        "demand_forecast_tool": "Demand forecast",
        "anomaly_detection_tool": "Anomaly detection",
        "sentiment_tool": "Review sentiment",
        "rag_search": "Knowledge retrieval",
    }
    return labels.get(tool, tool.replace("_", " ").title())
