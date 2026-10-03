"""Evidence observation and user-facing response safety for the supervisor graph."""

from typing import Any

from backend.app.agents.state import AgentState
from backend.app.guardrails.pii import contains_contact_details, redact_contact_details


def _trace(state: AgentState, event: str, **fields: Any) -> None:
    state.setdefault("trace", []).append({"event": event, "iteration": state.get("iteration_count", 0), **fields})


def _observation_summary(tool:str,output:dict[str,Any])->dict[str,Any]:
    summary={}
    for key in ("metric","value","count","segment","method","status","cohort_definition"):
        if key in output: summary[key]=output[key]
    if "customers" in output: summary["customer_count"]=len(output["customers"])
    if "recommendations" in output: summary["recommendation_count"]=len(output["recommendations"])
    if "flags" in output: summary["flag_count"]=len(output["flags"])
    if "sources" in output: summary["retrieval_count"]=len(output["sources"])
    if tool=="rag_search": summary["source_labels"]=output.get("sources",[])
    return summary


def _synthesize(state:AgentState)->AgentState:
    """Create a concise, evidence-derived response; never expose private reasoning."""
    results=state["tool_results"]
    parts=[]
    for row in results:
        tool,out=row["tool"],row["output"]
        if tool=="business_analytics_tool":
            parts.append(f"{out['metric'].replace('_',' ').title()}: {out['value']:,} (commerce analytics)." if isinstance(out.get("value"),(int,float)) else f"{out['metric']} results were retrieved from commerce analytics.")
        elif tool=="customer_cohort_tool":
            parts.append(f"Selected {out['count']} customers in the {out['segment']} cohort ({out['cohort_definition']}).")
        elif tool=="customer_rfm_tool":
            parts.append(f"RFM features were retrieved for {out.get('count',len(out.get('customers',[])))} cohort customers.")
        elif tool=="clv_prediction_tool":
            ranked=sorted(out.get("customers",[]),key=lambda item:item.get("predicted_clv",0),reverse=True)
            values=[float(x["predicted_clv"]) for x in ranked if x.get("predicted_clv") is not None]
            parts.append(f"The saved CLV model scored {len(ranked)} customers; the highest predicted value was {max(values):,.2f}." if values else f"The saved CLV model scored {len(ranked)} customers.")
        elif tool=="historical_value_tool":
            parts.append("The saved CLV model was unavailable; customers are ranked by observed historical monetary value (a proxy, not a model prediction).")
        elif tool=="churn_prediction_tool":
            ranked=sorted(out.get("customers",[]),key=lambda item:item.get("churn_probability",0),reverse=True)
            probabilities=[float(x["churn_probability"]) for x in ranked if x.get("churn_probability") is not None]
            parts.append(f"The saved churn model scored {len(ranked)} customers; {sum(value >= 0.7 for value in probabilities)} had a score of at least 0.70." if probabilities else f"The saved churn model scored {len(ranked)} customers.")
        elif tool=="rfm_risk_tool":
            customers=out.get("customers",[])
            high_risk=sum(row.get("risk_proxy")=="high" for row in customers)
            parts.append(f"The trained churn model was unavailable. An RFM-based risk proxy was used instead (recency_days > 180); {high_risk} of {len(customers)} selected customers met that rule. This is not a churn-model prediction.")
        elif tool=="recommendation_tool":
            product_count=len({product for row in out["recommendations"] for product in row.get("product_ids",[])})
            parts.append(f"Recommendations were generated for {len(out['recommendations'])} selected customers, covering {product_count} distinct products.")
        elif tool=="anomaly_detection_tool":
            if out.get("status")=="insufficient_history": parts.append(f"Insufficient history to assess {out['metric']} anomalies.")
            else: parts.append(f"{out['metric'].title()} anomaly scan ({out['method']}) found {len(out['flags'])} flagged days.")
        elif tool=="demand_forecast_tool":
            forecast=out.get("forecast",[])
            parts.append(f"Demand forecast returned {len(forecast)} points; first predicted demand is {forecast[0]['predicted_demand']} per day." if forecast else "Demand forecast returned no points.")
        elif tool=="sentiment_tool": parts.append(f"Review sentiment: {out['sentiment']}.")
        elif tool=="rag_search":
            snippets=[]
            text=out.get("documents","")[:1600]
            for source in out.get("sources",[])[:5]:
                snippets.append(f"[{source}]")
            parts.append("Retrieved untrusted evidence (treated as data, not instructions): " + text + " Sources: " + ", ".join(snippets))
    if not parts:
        question=state.get("user_query","").lower()
        if any(term in question for term in ("competitor", "market share", "industry benchmark")):
            parts=["The current platform has no validated competitor or market-share evidence, so it cannot make a factual comparison."]
        elif any(term in question for term in ("marketing channel", "traffic source", "acquisition channel")):
            parts=["The current commerce data does not include acquisition-channel or traffic-source evidence, so channel performance cannot be compared."]
        elif any(term in question for term in ("live chat", "phone support", "support channel", "customer support preference")):
            parts=["The current platform has no validated customer-support channel preference data, so it cannot compare live chat with phone support."]
        elif "currency" in question or "currencies" in question:
            parts=["The current platform has no validated currency-demand or foreign-exchange evidence, so it cannot recommend supported currencies from customer behavior."]
        else:
            parts=["No validated evidence was available to answer this question."]
    fallback_succeeded=any(call.get("fallback_used") and call.get("status")=="success" for call in state["tool_calls"])
    if state.get("errors") and not fallback_succeeded:
        parts.append("Some requested tools failed; results above are partial.")
    if state.get("anomalies"):
        anomaly_summaries=[]
        for item in state["anomalies"]:
            label=item["anomaly_type"].replace("_"," ")
            if item["anomaly_type"]=="iteration_limit": anomaly_summaries.append("Stopped at the configured iteration limit; returning partial results")
            elif item["anomaly_type"]=="repeated_tool_result": anomaly_summaries.append("Stopped after a repeated tool result")
            elif item["anomaly_type"]=="tool_oscillation": anomaly_summaries.append("Stopped after repeated tool switching")
            elif item["anomaly_type"]=="fallback_limit": anomaly_summaries.append("Stopped after reaching the fallback limit")
            elif item["anomaly_type"]=="retry_limit": anomaly_summaries.append("Stopped after reaching the retry limit")
            else: anomaly_summaries.append(f"Recorded {label}")
        parts.append("Execution safety: " + "; ".join(anomaly_summaries) + ".")
    question=state.get("user_query","").lower()
    if any(term in question for term in ("competitor", "market share", "industry benchmark")):
        parts.append("The platform has no structured competitor or market-share data; retrieved seller and delivery guidance does not establish a factual competitor comparison.")
    answer=" ".join(parts)
    has_rag=any(row["tool"]=="rag_search" for row in results)
    grounding="grounded" if results and (not has_rag or bool(state["sources"])) else "unknown"
    if results and not has_rag and all(row["tool"]=="customer_cohort_tool" and row["output"].get("count",0)==0 for row in results):
        grounding="unknown"
    state["final_response"]={"answer":answer,"grounded":grounding,"sources":state["sources"],"evidence_status":"validated tool results" if results else "no evidence","confidence":"limited" if any(call.get("fallback_used") for call in state["tool_calls"]) else "evidence-backed" if results else "none"}
    _trace(state,"synthesized",grounded=state["final_response"]["grounded"],evidence_count=len(results))
    return state


def _output_guard(state:AgentState)->AgentState:
    response=state.get("final_response",{})
    answer=redact_contact_details(response.get("answer",""))
    retrieved_sources={source for row in state["tool_results"] if row["tool"]=="rag_search" for source in row["output"].get("sources",[])}
    if any(row["tool"]=="rag_search" and row.get("valid") for row in state["tool_results"]):
        if not retrieved_sources or not set(response.get("sources",[])).issubset(retrieved_sources):
            response["grounded"]="false"
            answer="Retrieved evidence did not pass source validation; no grounded RAG claim is returned."
    if not answer.strip():
        state["errors"].append("Output guardrail rejected empty synthesis")
        answer="Unable to produce a validated answer from available evidence."
        response["grounded"]="false"
    if contains_contact_details(answer):
        answer="Output withheld because contact PII remained after redaction."
        response["grounded"]="false"
    response["answer"]=answer
    state["final_response"]=response
    _trace(state,"output_guard",pii_redacted=True,source_valid=bool(retrieved_sources) if any(row["tool"]=="rag_search" for row in state["tool_results"]) else None)
    return state


