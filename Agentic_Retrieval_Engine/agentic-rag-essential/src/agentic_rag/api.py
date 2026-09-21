from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .orchestrator import AgenticRetrievalOrchestrator, ConversationMemory, ToolRegistry, ToolSpec


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    trace_id: str
    confidence_achieved: bool
    iterations_used: int
    planning_mode: str
    reflection_mode: str
    latency_seconds: float
    sources: list[str]


def _demo_sql(_: str):
    return [{
        "content": "Acme Corp has high churn risk and declining account activity.",
        "source": "customer_db",
        "score": 0.91,
    }]


def _demo_vector(_: str):
    return [{
        "content": "Acme Corp has two unresolved support escalations related to service reliability.",
        "source": "support_tickets",
        "score": 0.88,
    }]


def create_app(orchestrator: AgenticRetrievalOrchestrator | None = None) -> FastAPI:
    if orchestrator is None:
        registry = ToolRegistry([
            ToolSpec("sql_db", "Structured customer/account data.", _demo_sql, max_results=6),
            ToolSpec("vector_search", "Unstructured documents and support tickets.", _demo_vector, max_results=8),
        ])
        agent = None
        if settings.openai_api_key:
            from .agents import OpenAIStructuredAgent
            agent = OpenAIStructuredAgent(
                settings.openai_api_key,
                model=settings.openai_model,
                timeout=settings.openai_timeout_seconds,
            )
        orchestrator = AgenticRetrievalOrchestrator(
            registry,
            agent=agent,
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
            max_evidence_chunks=settings.max_evidence_chunks,
            tool_timeout_seconds=settings.tool_timeout_seconds,
            max_retries=settings.max_retries,
            memory=ConversationMemory(),
        )

    app = FastAPI(title=settings.app_name, version="1.0.0")

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/tools")
    def tools() -> dict:
        return {
            "tools": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "max_results": spec.max_results,
                }
                for spec in orchestrator.registry.available()
            ]
        }

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            result = orchestrator.run(request.query)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return QueryResponse(
            answer=result.answer,
            trace_id=result.trace_id,
            confidence_achieved=result.confidence_achieved,
            iterations_used=result.iterations_used,
            planning_mode=result.planning_mode,
            reflection_mode=result.reflection_mode,
            latency_seconds=result.latency_seconds,
            sources=list(result.sources),
        )

    return app


app = create_app()
