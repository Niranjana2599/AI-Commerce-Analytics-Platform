"""HTTP API for the LangGraph commerce analyst."""

from fastapi import APIRouter, HTTPException

from backend.app.agents.supervisor import run_agent
from backend.app.schemas.contracts import AgentRequest, AgentResponse

router = APIRouter(tags=["Agentic Analyst"])


@router.post("/agent/ask", response_model=AgentResponse)
def ask_agent(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(**run_agent(request.question, request.customer_id, request.review_text))
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Agent request failed; see server logs for request diagnostics.") from error
