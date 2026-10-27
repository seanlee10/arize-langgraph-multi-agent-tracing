"""FastAPI entry point for the research-agent service (the system's front door)."""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import anthropic
from fastapi import FastAPI, HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, field_validator

from lg_tracing import request_context, setup_tracing
from research_agent.graph import RECURSION_LIMIT, DecisionAgentError, build_research_graph
from research_agent.web_search import make_web_search_tool

DEFAULT_MODEL = "claude-opus-5-5"


class InvokeRequest(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z.\-]{1,10}$")
    session_id: str | None = None
    user_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def normalise_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class InvokeResponse(BaseModel):
    ticker: str
    decision: Literal["buy", "hold", "sell"]
    confidence: float
    rationale: str
    research_brief: str
    trace_id: str


def create_app(graph: CompiledStateGraph, *, lifespan=None) -> FastAPI:
    app = FastAPI(title="research-agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(body: InvokeRequest) -> dict[str, Any]:
        with request_context(
            session_id=body.session_id or str(uuid.uuid4()),
            user_id=body.user_id or "demo-user",
            request_id=str(uuid.uuid4()),
            ticker=body.ticker,
        ):
            try:
                state = await graph.ainvoke(
                    {"messages": [HumanMessage(f"Research {body.ticker}")], "ticker": body.ticker},
                    config={"recursion_limit": RECURSION_LIMIT},
                )
            except DecisionAgentError as exc:
                raise HTTPException(
                    status_code=502, detail={"error": str(exc), "trace_id": exc.trace_id}
                ) from exc
            except GraphRecursionError as exc:
                raise HTTPException(
                    status_code=502, detail={"error": "research hit the step limit", "trace_id": ""}
                ) from exc
            except anthropic.APIError as exc:
                raise HTTPException(status_code=502, detail={"error": str(exc), "trace_id": ""}) from exc
        return {
            "ticker": body.ticker,
            "research_brief": state["research_brief"],
            "trace_id": state["trace_id"],
            **state["decision"],
        }

    return app


def build_model() -> ChatAnthropic:
    # No temperature/top_p/top_k, no forced tool_choice, thinking left on: all 400 on this model.
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


def create_app_from_env() -> FastAPI:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    provider = setup_tracing("research-agent")
    graph = build_research_graph(
        build_model(),
        web_search=make_web_search_tool(os.getenv("WEBSEARCH_MCP_URL", "http://localhost:8000/mcp")),
        decision_url=os.getenv("DECISION_AGENT_URL", "http://localhost:8002"),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        provider.shutdown()

    return create_app(graph, lifespan=lifespan)
