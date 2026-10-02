"""Result-aware observe → decide → act LangGraph supervisor."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from backend.app.agents.planner import understand_request
from backend.app.agents.selector import select_next_action
from backend.app.agents.state import AgentState
from backend.app.guardrails.injection import detect_prompt_injection
from backend.app.guardrails.pii import contains_contact_details, redact_contact_details
from backend.app.monitoring.agent_metrics import observe_agent_run
from backend.app.tools.registry import TOOL_REGISTRY, execute, validate_inputs

LOGGER=logging.getLogger(__name__)
MAX_ITERATIONS=8
MAX_RETRIES=2
MAX_REPEATED_SIGNATURES=2
MAX_TOOL_LATENCY_MS=15_000


def _trace(state:AgentState,event:str,**fields:Any)->None:
    state.setdefault("trace",[]).append({"event":event,"iteration":state.get("iteration_count",0),**fields})


def _guard(state:AgentState)->AgentState:
    question=state.get("user_query","")
    if not question.strip() or len(question)>2_000:
        state["guardrail_result"]={"status":"blocked","reason":"Question must be non-empty and at most 2,000 characters."}
    elif detect_prompt_injection(question):
        state["guardrail_result"]={"status":"blocked","reason":"Prompt-injection pattern detected by the demo input filter."}
    elif contains_contact_details(question) or contains_contact_details(state.get("customer_id") or ""):
        state["guardrail_result"]={"status":"blocked","reason":"Email/phone-like PII must not be sent to the local planner. Use the platform customer identifier."}
    elif contains_contact_details(state.get("review_text") or ""):
        state["guardrail_result"]={"status":"blocked","reason":"Review text containing email/phone-like PII was blocked before sentiment analysis."}
    elif question.strip().lower() in {"help","tell me something useful","do something","analyze this"}:
        state["guardrail_result"]={"status":"clarification","reason":"Please specify a business metric, customer segment, forecast, review, recommendation, or knowledge-base question."}
    else:
        state["guardrail_result"]={"status":"accepted","reason":"basic demo checks passed"}
    _trace(state,"guard",status=state["guardrail_result"]["status"])
    if state["guardrail_result"]["status"]=="blocked":
        state["final_response"]={"answer":state["guardrail_result"]["reason"],"grounded":"unknown","confidence":"none"}
    return state


def _understand(state:AgentState)->AgentState:
    started=time.perf_counter()
    intent,objectives,mode,calls=understand_request(state["user_query"],state.get("customer_id"),state.get("planner_mode","auto"))
    if state.get("review_text") and "sentiment" not in objectives:
        objectives.append("sentiment")
    state.update(intent=intent,objectives=objectives,planner_mode=mode,llm_calls=state.get("llm_calls",0)+calls,
                 llm_latency_ms=state.get("llm_latency_ms",0)+(time.perf_counter()-started)*1000 if calls else state.get("llm_latency_ms",0),
                 plan={"intent":intent,"objectives":objectives,"strategy":"select one tool, observe its result, then decide again"})
    state["messages"].append({"role":"assistant","content":f"Request categorized as {intent}; objectives: {', '.join(objectives)}."})
    state["memory_context"]={"scope":"request_only","completed_tools":[]}
    _trace(state,"understand",intent=intent,objectives=objectives,planner=mode)
    return state


def _select(state:AgentState)->AgentState:
    if state.get("next_action") is not None:
        action=state["next_action"]
        mode=state.get("selector_mode","rules")
        calls=0
    else:
        started=time.perf_counter()
        action,mode,calls=select_next_action(state,state.get("selector_strategy","auto"))
        state["llm_calls"]=state.get("llm_calls",0)+calls
        state["selector_mode"]=mode
        if calls: state["llm_latency_ms"]=state.get("llm_latency_ms",0)+(time.perf_counter()-started)*1000
    state["next_action"]=action
    if action is None:
        state["decision"]="finish"
        state["decision_summary"]="Requested evidence has been collected or no safe next tool is available."
        _trace(state,"decision",decision="finish",summary=state["decision_summary"],selector=mode)
        return state
    if state.get("iteration_count",0)>=state.get("max_iterations",MAX_ITERATIONS):
        state["anomalies"].append({"anomaly_type":"iteration_limit","threshold":state["max_iterations"],"observed_value":state["iteration_count"],"action_taken":"stop and synthesize partial evidence"})
        state["decision"]="finish"
        state["decision_summary"]="Stopped at the configured iteration limit; partial evidence is preserved."
        state["next_action"]=None
        _trace(state,"decision",decision="iteration_limit",summary=state["decision_summary"])
        return state
    state["current_action"]=action
    state["current_tool"]=action["tool"]
    state["attempt_count"]=0
    state["fallback_used"]=False
    state["fallback_for"]=None
    state["tool_valid"]=False
    state["decision"]="continue"
    _trace(state,"tool_selected",tool=action["tool"],selector=mode,action_summary=f"Run {action['tool']} using validated request/evidence fields.")
    return state


def _execute(state:AgentState)->AgentState:
    action=state["current_action"]
    spec=TOOL_REGISTRY[action["tool"]]
    state["attempt_count"]=state.get("attempt_count",0)+1
    state["iteration_count"]=state.get("iteration_count",0)+1
    started=time.perf_counter()
    input_validated=False
    try:
        validate_inputs(spec,action.get("input",{}))
        input_validated=True
        result=execute(action["tool"],action.get("input",{}))
        duration=(time.perf_counter()-started)*1_000
        call={"tool":action["tool"],"input":action.get("input",{}),"status":"success","output":result,"input_validated":True,
              "attempt":state["attempt_count"],"elapsed_ms":round(duration,2),"fallback_used":bool(state.get("fallback_used")),
              "primary_tool":(state.get("fallback_for") or {}).get("tool"),"fallback_tool":action["tool"] if state.get("fallback_used") else None,
              "fallback_reason":(state.get("fallback_for") or {}).get("fallback_reason"),"error":None}
        state["tool_valid"]=True
    except Exception as error:
        duration=(time.perf_counter()-started)*1_000
        call={"tool":action["tool"],"input":action.get("input",{}),"status":"failed","output":None,"input_validated":input_validated,
              "attempt":state["attempt_count"],"elapsed_ms":round(duration,2),"fallback_used":bool(state.get("fallback_used")),
              "primary_tool":(state.get("fallback_for") or {}).get("tool"),"fallback_tool":action["tool"] if state.get("fallback_used") else None,"fallback_reason":(state.get("fallback_for") or {}).get("fallback_reason"),
              "error_type":type(error).__name__,"error":str(error)[:500],"retryable":not isinstance(error,(ValueError,TypeError,KeyError))}
        state["tool_valid"]=False
        state["errors"].append(f"{action['tool']} attempt {state['attempt_count']}: {type(error).__name__}: {str(error)[:300]}")
        if action["tool"]=="rag_search":
            state["anomalies"].append({"anomaly_type":"empty_retrieval","threshold":1,"observed_value":0,"action_taken":"mark retrieved evidence invalid; do not synthesize a grounded answer"})
        LOGGER.warning("Agent tool failed request_id=%s tool=%s attempt=%s error=%s",state["request_id"],action["tool"],state["attempt_count"],type(error).__name__)
    state["tool_calls"].append(call)
    if state.get("fallback_used") and state["attempt_count"]==1:
        state["fallback_count"]+=1
    _trace(state,"tool_result",tool=action["tool"],status=call["status"],attempt=state["attempt_count"],elapsed_ms=call["elapsed_ms"],fallback_used=call["fallback_used"])
    return state


def _validate(state:AgentState)->AgentState:
    last=state["tool_calls"][-1]
    state["tool_valid"]=last["status"]=="success"
    if state["tool_valid"] and last["tool"]=="rag_search":
        if not last["output"].get("sources") or not last["output"].get("documents","").strip():
            state["tool_valid"]=False
            last["status"]="invalid_output"
            last["error"]="Retrieval evidence or source metadata was empty."
            state["errors"].append(f"{last['tool']}: empty retrieval evidence")
            state["anomalies"].append({"anomaly_type":"empty_retrieval","threshold":1,"observed_value":0,"action_taken":"mark evidence invalid; do not synthesize a grounded answer"})
    _trace(state,"result_validation",tool=last["tool"],valid=state["tool_valid"])
    return state


def _validation_route(state:AgentState)->Literal["observe","retry","fallback","stop"]:
    if state["tool_valid"]: return "observe"
    if state.get("iteration_count",0)>=state.get("max_iterations",MAX_ITERATIONS):
        state["anomalies"].append({"anomaly_type":"iteration_limit","threshold":state["max_iterations"],"observed_value":state["iteration_count"],"action_taken":"stop after failed tool; preserve partial evidence"})
        return "stop"
    if state.get("fallback_used"):
        return "stop"
    spec=TOOL_REGISTRY[state["current_tool"]]
    if state["tool_calls"][-1].get("retryable") and state.get("attempt_count",0)<=min(spec.max_retries,state.get("max_retries",MAX_RETRIES)):
        return "retry"
    if spec.fallback_tool:
        return "fallback"
    return "stop"


def _retry(state:AgentState)->AgentState:
    state["retry_count"]+=1
    if state["retry_count"]>4:
        state["anomalies"].append({"anomaly_type":"retry_limit","threshold":4,"observed_value":state["retry_count"],"action_taken":"stop retries and preserve partial evidence"})
        state["decision"]="finish"
    _trace(state,"retry",tool=state["current_tool"],retry_count=state["retry_count"],attempt=state["attempt_count"]+1)
    return state


def _make_fallback(state:AgentState)->AgentState:
    failed_action=state["current_action"]
    primary=state["current_tool"]
    fallback_name=TOOL_REGISTRY[primary].fallback_tool
    if not fallback_name:
        state["decision"]="finish"
        return state
    inputs=failed_action.get("input",{})
    customers=inputs.get("customers")
    customer_id=inputs.get("customer_id")
    if customers is None:
        cohort=next((row["output"].get("customers") for row in reversed(state["tool_results"])
                    if row["tool"] in {"customer_cohort_tool","customer_rfm_tool"} and row.get("valid")),None)
        customers=cohort
    fallback_input={"customers":customers} if customers else ({"customer_id":customer_id} if customer_id else {})
    state["fallback_for"]={"tool":primary,"fallback_reason":state["tool_calls"][-1].get("error","primary tool unavailable")}
    try:
        from backend.app.tools.registry import validate_inputs
        validate_inputs(TOOL_REGISTRY[fallback_name],fallback_input)
    except Exception as error:
        state["anomalies"].append({"anomaly_type":"fallback_unavailable","threshold":1,"observed_value":0,"action_taken":"controlled stop: fallback inputs unavailable"})
        state["errors"].append(f"Fallback {fallback_name} unavailable: {str(error)[:300]}")
        return state
    state["current_action"]={"tool":fallback_name,"input":fallback_input}
    state["current_tool"]=fallback_name
    state["fallback_used"]=True
    state["attempt_count"]=0
    _trace(state,"fallback_selected",primary_tool=primary,fallback_tool=fallback_name,reason=state["fallback_for"]["fallback_reason"][:200])
    return state


def _observe(state:AgentState)->AgentState:
    call=state["tool_calls"][-1]
    signature=json.dumps({"tool":call["tool"],"input":call["input"],"output":call["output"]},sort_keys=True,default=str)
    count=state["seen_signatures"].get(signature,0)+1
    state["seen_signatures"][signature]=count
    if count>=MAX_REPEATED_SIGNATURES:
        state["anomalies"].append({"anomaly_type":"repeated_tool_result","threshold":MAX_REPEATED_SIGNATURES,"observed_value":count,"action_taken":"stop repeated tool loop"})
    names=[row["tool"] for row in state["tool_calls"] if row["status"]=="success"]
    if len(names)>=4 and names[-4]==names[-2] and names[-3]==names[-1]:
        state["anomalies"].append({"anomaly_type":"tool_oscillation","threshold":4,"observed_value":names[-4:],"action_taken":"stop alternating tool cycle"})
    if call["elapsed_ms"]>MAX_TOOL_LATENCY_MS:
        state["anomalies"].append({"anomaly_type":"tool_latency_spike","threshold":MAX_TOOL_LATENCY_MS,"observed_value":call["elapsed_ms"],"action_taken":"record latency anomaly"})
    result_summary=_observation_summary(call["tool"],call["output"])
    if call["tool"]=="churn_prediction_tool":
        probabilities=[float(row["churn_probability"]) for row in call["output"].get("customers",[]) if row.get("churn_probability") is not None]
        result_summary["high_risk_count"]=sum(probability>=0.7 for probability in probabilities)
    if call["fallback_used"]:
        state["tool_results"].append({"tool":call["tool"],"output":call["output"],"valid":True,"fallback_used":True,"fallback_for":state["fallback_for"]["tool"]})
    else:
        state["tool_results"].append({"tool":call["tool"],"output":call["output"],"valid":True,"fallback_used":False,"fallback_for":None})
    output=call["output"]
    observation={"tool":call["tool"],"status":"success","summary":result_summary,"iteration":state.get("iteration_count",len(state["tool_calls"]))}
    state["observations"].append(observation)
    state["sources"].extend(output.get("sources",[]) if isinstance(output,dict) else [])
    state["sources"]=list(dict.fromkeys(state["sources"]))
    state["memory_context"]["completed_tools"].append(call["tool"])
    state["messages"].append({"role":"tool","name":call["tool"],"content":json.dumps(observation,default=str)})
    _trace(state,"observation",tool=call["tool"],summary=observation["summary"],fallback_used=call["fallback_used"])
    if count>=MAX_REPEATED_SIGNATURES or state.get("decision")=="finish" or state.get("anomalies") and state["anomalies"][-1]["anomaly_type"] in {"retry_limit","tool_oscillation"}:
        state["decision"]="finish"
        state["decision_summary"]="Stopped a repeated/oscillating tool loop or execution retry anomaly."
    if sum(bool(row.get("fallback_used")) for row in state["tool_calls"])>=3:
        state["anomalies"].append({"anomaly_type":"fallback_limit","threshold":3,"observed_value":3,"action_taken":"stop after excessive fallbacks"})
        state["decision"]="finish"
        state["decision_summary"]="Stopped after reaching the fallback safety limit."
    return state


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


def _reason(state:AgentState)->AgentState:
    if state.get("decision")=="finish" and state.get("decision_summary"):
        return state
    started=time.perf_counter()
    action,mode,calls=select_next_action(state,state.get("selector_strategy","auto"))
    state["llm_calls"]+=calls
    if calls: state["llm_latency_ms"]=state.get("llm_latency_ms",0)+(time.perf_counter()-started)*1000
    state["selector_mode"]=mode
    if action is None:
        state["decision"]="finish"
        state["decision_summary"]="The requested evidence is complete; no further tool is required."
        state["next_action"]=None
    else:
        state["decision"]="continue"
        state["decision_summary"]=f"Observed {state['tool_results'][-1]['tool']} result; next step is {action['tool']}."
        state["next_action"]=action
    _trace(state,"reasoned_decision",decision=state["decision"],summary=state["decision_summary"],next_tool=(action or {}).get("tool"),selector=mode)
    return state


def _after_reason(state:AgentState)->Literal["select","synthesize"]:
    if state.get("decision")=="finish" or state.get("anomalies") and state["anomalies"][-1]["anomaly_type"] in {"repeated_tool_result","fallback_limit"}:
        return "synthesize"
    return "select"


def _after_validation(state:AgentState)->str:
    return _validation_route(state)


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
            preview=[{"customer_id":x.get("customer_unique_id"),"predicted_clv":round(float(x["predicted_clv"]),2)} for x in ranked[:5] if x.get("predicted_clv") is not None]
            parts.append(f"The saved CLV model scored {len(ranked)} customers. Highest predicted values: {preview}.")
        elif tool=="historical_value_tool":
            parts.append("The saved CLV model was unavailable; customers are ranked by observed historical monetary value (a proxy, not a model prediction).")
        elif tool=="churn_prediction_tool":
            ranked=sorted(out.get("customers",[]),key=lambda item:item.get("churn_probability",0),reverse=True)
            preview=[{"customer_id":x.get("customer_unique_id"),"churn_probability":round(float(x["churn_probability"]),3)} for x in ranked[:5] if x.get("churn_probability") is not None]
            parts.append(f"The saved churn model scored {len(ranked)} customers. Highest scores: {preview}.")
        elif tool=="rfm_risk_tool":
            preview=[{"customer_id":x.get("customer_unique_id"),"risk_proxy":x["risk_proxy"],"recency_days":x.get("recency_days")} for x in out.get("customers",[])[:5]]
            parts.append(f"The trained churn model was unavailable. An RFM-based risk proxy was used instead (recency_days > 180); this is not a churn-model prediction. Examples: {preview}.")
        elif tool=="recommendation_tool":
            parts.append(f"Recommendations were generated for {len(out['recommendations'])} selected customers: {out['recommendations']}.")
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


def _evaluate(state:AgentState)->AgentState:
    result=state["final_response"]
    succeeded={call["tool"] for call in state["tool_calls"] if call["status"]=="success"}
    recovered=set(succeeded)
    recovered.update(call["primary_tool"] for call in state["tool_calls"] if call.get("fallback_used") and call.get("status")=="success" and call.get("primary_tool"))
    unresolved_failures=[call for call in state["tool_calls"] if call["status"]!="success" and call["tool"] not in recovered]
    state["evaluation"]={"tool_calls":len(state["tool_calls"]),"successful_results":len(state["tool_results"]),
        "input_schema_accuracy":sum(bool(call.get("input_validated")) for call in state["tool_calls"])/max(len(state["tool_calls"]),1),
        "errors":len(state["errors"]),"retries":state["retry_count"],"fallbacks":state["fallback_count"],
        "iterations":state["iteration_count"],"grounded":result.get("grounded","unknown"),
        "anomalies":len(state["anomalies"]),"task_success":bool(state["tool_results"]) and not unresolved_failures and result.get("grounded")!="false"}
    _trace(state,"evaluation",**state["evaluation"])
    return state


def _finish_failure(state:AgentState)->AgentState:
    guard_status=state.get("guardrail_result",{}).get("status")
    if guard_status in {"blocked","clarification"}:
        state["intent"]="blocked" if guard_status=="blocked" else "clarification"
        state["final_response"]={"answer":("Request blocked: " if guard_status=="blocked" else "Clarification needed: ")+state["guardrail_result"]["reason"],"grounded":"unknown","sources":[],"evidence_status":"none","confidence":"none"}
        state["evaluation"]={"task_success":False,"blocked":guard_status=="blocked","clarification":guard_status=="clarification","tool_calls":0,"grounded":"unknown"}
        _trace(state,"input_policy",status=guard_status,reason=state["guardrail_result"]["reason"])
        return state
    state["decision"]="finish"
    state["decision_summary"]="No further safe action is available; retaining partial evidence and tool errors."
    return state


_builder=StateGraph(AgentState)
for _name,_node in (("guard",_guard),("understand",_understand),("select",_select),("execute",_execute),("validate",_validate),("retry",_retry),("fallback",_make_fallback),("observe",_observe),("reason",_reason),("synthesize",_synthesize),("output_guard",_output_guard),("evaluate",_evaluate),("stop",_finish_failure)):
    _builder.add_node(_name,_node)
_builder.add_edge(START,"guard")
_builder.add_conditional_edges("guard",lambda s:"stop" if s.get("guardrail_result",{}).get("status")!="accepted" else "understand",{"stop":"stop","understand":"understand"})
_builder.add_edge("understand","select")
_builder.add_conditional_edges("select",lambda s:"execute" if s.get("next_action") else "synthesize",{"execute":"execute","synthesize":"synthesize"})
_builder.add_edge("execute","validate")
_builder.add_conditional_edges("validate",_after_validation,{"observe":"observe","retry":"retry","fallback":"fallback","stop":"stop"})
_builder.add_conditional_edges("retry",lambda s:"stop" if s.get("decision")=="finish" else "execute",{"stop":"stop","execute":"execute"})
_builder.add_edge("fallback","execute")
_builder.add_edge("observe","reason")
_builder.add_conditional_edges("reason",_after_reason,{"select":"select","synthesize":"synthesize"})
_builder.add_edge("synthesize","output_guard")
_builder.add_edge("output_guard","evaluate")
_builder.add_edge("evaluate",END)
_builder.add_conditional_edges("stop",lambda s:"evaluate" if s.get("guardrail_result",{}).get("status") in {"blocked","clarification"} else "synthesize",{"evaluate":"evaluate","synthesize":"synthesize"})
agent_graph=_builder.compile()


def _public_output(tool:str,output:dict[str,Any]|None)->dict[str,Any]|None:
    if not isinstance(output,dict): return None
    keys={"customer_cohort_tool":("segment","count","cohort_definition"),"customer_rfm_tool":("count","note"),
          "business_analytics_tool":("metric","value","rows"),"anomaly_detection_tool":("metric","method","baseline_median","observations","flags","status"),
          "demand_forecast_tool":("forecast",),"sentiment_tool":("sentiment",),"rag_search":("documents","sources"),
          "recommendation_tool":("recommendations",),"churn_prediction_tool":("customers","model"),"clv_prediction_tool":("customers","model"),
          "rfm_risk_tool":("customers","method","is_model_prediction"),"historical_value_tool":("customers","method","is_model_prediction")}.get(tool,())
    safe={key:output[key] for key in keys if key in output}
    if "customers" in safe:
        safe["customers"]=[{key:row[key] for key in ("customer_unique_id","customer_id","recency_days","frequency","monetary","predicted_clv","churn_probability","risk_proxy","prediction") if key in row} for row in safe["customers"][:20]]
    if "recommendations" in safe:
        safe["recommendations"]=[{key:row[key] for key in ("customer_id","product_ids") if key in row} for row in safe["recommendations"]]
    if "documents" in safe: safe["documents"]=redact_contact_details(str(safe["documents"])[:1600])
    return safe


def _public_input_summary(inputs:dict[str,Any])->dict[str,Any]:
    summary={key:value for key,value in inputs.items() if key in {"metric","segment","limit","days","k"}}
    for key in ("customers","customer_ids"):
        if key in inputs: summary[f"{key}_count"]=len(inputs[key])
    if "features" in inputs: summary["feature_names"]=sorted(inputs["features"].keys())
    if "customer_id" in inputs: summary["customer_id"]="[REDACTED]"
    if "review" in inputs: summary["review"]="[REDACTED]"
    if "query" in inputs: summary["query"]="[REDACTED]"
    return summary


def run_agent(question:str,customer_id:str|None=None,review_text:str|None=None,*,planner_mode:str="auto",selector_mode:str="auto",max_iterations:int=MAX_ITERATIONS)->dict[str,Any]:
    request_id=str(uuid4())
    state:AgentState={"request_id":request_id,"user_query":question,"customer_id":customer_id,"review_text":review_text,
        "messages":[{"role":"user","content":question}],"objectives":[],"tool_calls":[],"tool_results":[],"observations":[],
        "errors":[],"retry_count":0,"attempt_count":0,"fallback_count":0,"iteration_count":0,"max_iterations":max(1,min(int(max_iterations),MAX_ITERATIONS)),
        "max_retries":MAX_RETRIES,"memory_context":{},"guardrail_result":{},"sources":[],"anomalies":[],"evaluation":{},"trace":[],"review_text":review_text,
        "seen_signatures":{},"llm_calls":0,"llm_latency_ms":0.0,"planner_mode":planner_mode,"selector_mode":"auto","selector_strategy":selector_mode,"fallback_used":False}
    started=time.perf_counter()
    result=agent_graph.invoke(state,config={"recursion_limit":8*MAX_ITERATIONS+20})
    latency_ms=round((time.perf_counter()-started)*1_000,2)
    result["latency_ms"]=latency_ms
    _trace(result,"monitoring",latency_ms=latency_ms,llm_calls=result.get("llm_calls",0),token_usage="unavailable_from_current_runtime")
    observe_agent_run(result,latency_ms)
    LOGGER.info("agent_request_complete request_id=%s intent=%s tools=%s latency_ms=%s",request_id,result.get("intent","blocked"),[item["tool"] for item in result["tool_calls"]],latency_ms)
    trace=json.loads(json.dumps(result.get("trace",[]),default=str))
    safe_calls=[]
    for item in result.get("tool_calls",[]):
        safe_calls.append({"tool":item["tool"],"status":item["status"],"input_redacted":True,
            "input_fields":sorted(item.get("input",{}).keys()),"input_summary":_public_input_summary(item.get("input",{})),"output":_public_output(item["tool"],item.get("output")),
            "attempt":item["attempt"],"elapsed_ms":item["elapsed_ms"],"fallback_used":item["fallback_used"],
            "primary_tool":item.get("primary_tool"),"fallback_tool":item.get("fallback_tool"),
            "fallback_reason":redact_contact_details(str(item.get("fallback_reason") or "")) or None,
            "error":redact_contact_details(str(item.get("error") or "")) or None,"input_validated":item.get("input_validated",False)})
    safe_calls=json.loads(json.dumps(safe_calls,default=str))
    response=result.get("final_response",{})
    return {"request_id":request_id,"answer":response.get("answer",""),"intent":result.get("intent","blocked"),
        "tools_used":[item["tool"] for item in safe_calls],"iterations":result.get("iteration_count",0),"latency_ms":latency_ms,
        "fallback_used":result.get("fallback_count",0)>0,"errors":[redact_contact_details(str(error)) for error in result.get("errors",[])],"sources":[redact_contact_details(str(source)) for source in result.get("sources",[])],
        "evidence":safe_calls,"evaluation":result.get("evaluation",{}),"trace":trace,"anomalies":result.get("anomalies",[]),
        "grounded":response.get("grounded","unknown"),"confidence":response.get("confidence","none"),
        "decision_summary":result.get("decision_summary",""),"llm_calls":result.get("llm_calls",0),"llm_latency_ms":round(result.get("llm_latency_ms",0),2),
        "planner_mode":result.get("planner_mode","rules"),"selector_mode":result.get("selector_mode","rules"),
        "estimated_cost_usd":None,"token_usage":None}
