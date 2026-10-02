"""LangGraph commerce analyst backed by the FastAPI agent endpoint."""

import streamlit as st

from utils.api_client import api_request
from utils.ui import page_hero, setup_page

setup_page("Agentic Analyst")
page_hero(
    "Multi-step business analysis",
    "Agentic Commerce Analyst",
    "Ask a business question. The supervisor selects from existing analytics, ML, recommendation, forecasting, sentiment, anomaly, and RAG services.",
)

with st.form("agent_question_form"):
    question = st.text_area(
        "Business question",
        placeholder="What is the total revenue? Did anything unusual happen to orders?",
        max_chars=2_000,
        height=110,
    )
    customer_id = st.text_input("Customer ID (optional)", max_chars=128)
    review_text = st.text_area("Review text (optional, for sentiment questions)", max_chars=5_000, height=70)
    submitted = st.form_submit_button("Analyze", type="primary", width="stretch")

if submitted:
    if not question.strip():
        st.warning("Enter a business question to continue.")
    else:
        with st.spinner("Planning and querying commerce services..."):
            result, error = api_request(
                "POST",
                "/agent/ask",
                json={"question": question.strip(), "customer_id": customer_id.strip() or None,"review_text":review_text.strip() or None},
                timeout=120,
            )
        if error:
            st.error(error)
        elif result:
            st.markdown("### Analysis")
            st.write(result.get("answer", "No answer returned."))
            columns = st.columns(4)
            columns[0].metric("Intent", result.get("intent", "unknown"))
            columns[1].metric("Tool calls", len(result.get("tools_used", [])))
            columns[2].metric("Iterations", result.get("iterations", 0))
            columns[3].metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
            if result.get("tools_used"):
                st.caption("Tools: " + " → ".join(result["tools_used"]))
            if result.get("sources"):
                st.markdown("**Retrieved sources:** " + ", ".join(sorted(set(result["sources"]))))
            if result.get("errors"):
                with st.expander("Tool errors and fallbacks"):
                    st.write(result["errors"])
            with st.expander("Safe execution trace", expanded=True):
                if result.get("decision_summary"):
                    st.caption(f"Decision: {result['decision_summary']}")
                for index, step in enumerate(result.get("evidence", []), start=1):
                    status = "fallback" if step.get("fallback_used") else step.get("status", "unknown")
                    st.markdown(
                        f"**{index}. {step.get('tool')}** · {status} · "
                        f"{step.get('elapsed_ms', 0):.0f} ms · attempt {step.get('attempt', 1)}"
                    )
                    if step.get("fallback_used"):
                        st.caption(f"Fallback for {step.get('primary_tool')}: {step.get('fallback_tool')} — {step.get('fallback_reason')}")
                    if step.get("output"):
                        st.json(step["output"])
                st.markdown("**Decision and synthesis trace**")
                st.dataframe(result.get("trace", []), hide_index=True, width="stretch")
                st.markdown("**Evaluation**")
                st.json(result.get("evaluation", {}))

with st.expander("How this differs from the RAG chatbot"):
    st.markdown(
        "The RAG chatbot retrieves indexed documents for knowledge-base questions. "
        "This supervisor can route to structured analytics or existing model services, "
        "combine multiple tool results, and include RAG when relevant. Tool selection and execution are bounded by the backend registry."
    )
