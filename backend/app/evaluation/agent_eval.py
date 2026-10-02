"""Runtime benchmark for rules, LLM-planned, and result-aware selector modes."""

from __future__ import annotations

from statistics import mean
from typing import Any, Callable

from backend.app.evaluation.golden_dataset import GOLDEN_DATASET


def evaluate_runtime_cases(run:Callable[...,dict[str,Any]],cases:list[dict[str,Any]]|None=None,*,planner_mode:str="rules",selector_mode:str="rules")->dict[str,Any]:
    """Execute the same golden requests and score tool selection plus actual outcomes."""
    cases=cases or [case for case in GOLDEN_DATASET if "question" in case and not case.get("operation")]
    rows=[]
    for case in cases:
        result=run(case["question"],review_text=case.get("review_text"),max_iterations=case.get("max_iterations",8),planner_mode=planner_mode,selector_mode=selector_mode)
        tools=result.get("tools_used",[])
        expected_tools=case.get("expected_tools") or case.get("expected_tool_order") or ([case["first_tool"]] if case.get("first_tool") else [])
        first_input=result.get("evidence",[{}])[0].get("input_summary",{}) if result.get("evidence") else {}
        expected_input=case.get("expected_first_input",{})
        blocked=result.get("intent")=="blocked"
        expected_result=case.get("expected_result")
        expected_intent=case.get("intent")
        actual_result=result.get("evaluation",{}).get("task_success")
        final_answer=result.get("answer","")
        if expected_result is None:
            result_correct=(actual_result is True) if case.get("must_succeed") else (blocked==bool(case.get("blocked")) if case.get("blocked") is not None else True)
        elif isinstance(expected_result,str):
            result_correct=expected_result in final_answer
        else:
            result_correct=actual_result is expected_result
        rows.append({"case_id":case["id"],"intent_correct":not case.get("intent") or result.get("intent")==case["intent"],
            "first_tool_correct":not case.get("first_tool") or bool(tools) and tools[0]==case["first_tool"],
            "tool_argument_correct":all(first_input.get(key)==value for key,value in expected_input.items()),
            "tool_arguments_valid":all(item.get("input_validated",False) for item in result.get("evidence",[]) if item["status"]=="success"),
            "task_completed":(bool(actual_result) and result_correct) if case.get("must_succeed") else (result_correct and not case.get("blocked",False)),
            "expected_tools":expected_tools,"actual_tools":tools,"tool_order":tools,
            "tool_order_correct":not expected_tools or tools==expected_tools,
            "plan_success":not expected_tools or tools==expected_tools,"expected_result":expected_result,
            "actual_result":actual_result,"result_correct":result_correct,"task_success":(bool(actual_result) and result_correct) if case.get("must_succeed") else (result_correct and not case.get("blocked",False)),
            "intent_correct":not expected_intent or result.get("intent")==expected_intent,
            "guardrail":result.get("trace",[]) and next((event.get("status") for event in result["trace"] if event.get("event")=="guard"),None),
            "final_answer":final_answer,"tool_calls":len(result.get("evidence",[])),"iterations":result.get("iterations",0),
            "retry_count":result.get("evaluation",{}).get("retries",0),"fallback_used":result.get("fallback_used",False),
            "fallback_expected":case.get("fallback_expected"),"fallback_correct":None if "fallback_expected" not in case else result.get("fallback_used",False)==case["fallback_expected"],
            "latency_ms":result.get("latency_ms",0),
            "failure_rate":int(not result.get("evaluation",{}).get("task_success",False)),"grounded":result.get("grounded","unknown"),
            "planner_mode":result.get("planner_mode"),"selector_mode":result.get("selector_mode")})
    n=max(len(rows),1)
    return {"cases":rows,"metrics":{"intent_accuracy":mean(row["intent_correct"] for row in rows) if rows else 0,
        "execution_task_success_rate":mean(row["task_success"] for row in rows) if rows else 0,
        "first_tool_accuracy":mean(row["first_tool_correct"] for row in rows) if rows else 0,
        "tool_argument_accuracy":mean(row["tool_argument_correct"] for row in rows) if rows else 0,
        "schema_validity_rate":mean(row["tool_arguments_valid"] for row in rows) if rows else 0,
        "task_completion_rate":mean(row["task_completed"] for row in rows) if rows else 0,
        "plan_success":mean(row["plan_success"] for row in rows) if rows else 0,
        "tool_order_accuracy":mean(row["tool_order_correct"] for row in rows) if rows else 0,
        "retry_rate":sum(row["retry_count"]>0 for row in rows)/n,"fallback_rate":sum(row["fallback_used"] for row in rows)/n,
        "average_tool_calls":mean(row["tool_calls"] for row in rows) if rows else 0,
        "average_iterations":mean(row["iterations"] for row in rows) if rows else 0,
        "average_latency_ms":mean(row["latency_ms"] for row in rows) if rows else 0,
        "failure_rate":mean(row["failure_rate"] for row in rows) if rows else 0,
        "fallback_correctness":mean([row["fallback_correct"] for row in rows if row["fallback_correct"] is not None]) if any(row["fallback_correct"] is not None for row in rows) else None}}


def benchmark_agent_versions(run:Callable[...,dict[str,Any]],cases:list[dict[str,Any]]|None=None)->dict[str,Any]:
    """V1 deterministic planner; V2 LLM planner+rules selector; V3 LLM planner+result-aware loop."""
    return {
        "V1_deterministic_planner":evaluate_runtime_cases(run,cases,planner_mode="rules",selector_mode="rules"),
        "V2_llm_planner_rules_selector":evaluate_runtime_cases(run,cases,planner_mode="llm",selector_mode="rules"),
        "V3_result_aware_agent":evaluate_runtime_cases(run,cases,planner_mode="llm",selector_mode="llm"),
    }
