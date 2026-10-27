"""FastAPI entry point for the decision-agent service."""

import os
from contextlib import asynccontextmanager
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from decision_agent.graph import DecisionNotSubmitted, build_decision_graph, run_decision
from decision_agent.tools import INDICATOR_TOOLS
from lg_tracing import continue_request_context, setup_tracing

DEFAULT_MODEL = "claude-opus-5-5"


class DecideRequest(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z.\-]{1,10}$")
    research_brief: str

    @field_validator("ticker", mode="before")
    @classmethod
    def normalise_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class DecideResponse(BaseModel):
    decision: Literal["buy", "hold", "sell"]
    confidence: float
    rationale: str


def create_app(
    *, model: BaseChatModel, tools: list[BaseTool] = INDICATOR_TOOLS, lifespan=None
) -> FastAPI:
    app = FastAPI(title="decision-agent", lifespan=lifespan)
    graph = build_decision_graph(model, tools)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/invoke", response_model=DecideResponse)
    async def invoke(body: DecideRequest, request: Request) -> dict:
        # Headers carry traceparent + baggage from the research agent.
        with continue_request_context(request.headers, ticker=body.ticker):
            try:
                return await run_decision(graph, body.ticker, body.research_brief)
            except (DecisionNotSubmitted, anthropic.APIError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

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
    provider = setup_tracing("decision-agent")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        provider.shutdown()

    return create_app(model=build_model(), lifespan=lifespan)
