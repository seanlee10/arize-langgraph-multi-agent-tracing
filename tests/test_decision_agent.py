import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from decision_agent.app import create_app
from lg_tracing import inject_carrier, request_context
from tests.fakes import ScriptedChatModel, tool_call
from tests.helpers import only_span

BODY = {"ticker": "NVDA", "research_brief": "b"}


@tool
def force_index(ticker: str, period: int = 13) -> dict:
    """Fake force index."""
    return {"trend": "bullish"}


@tool("force_index")
def failing(ticker: str) -> dict:
    """Fake force index that fails."""
    raise RuntimeError("upstream data feed down")


def submit(id: str = "c2", **overrides):
    args = {"decision": "buy", "confidence": 0.7, "rationale": "FI up"} | overrides
    return tool_call("submit_decision", args, id=id)


def caller_headers(session_id: str) -> tuple[dict[str, str], trace.Span]:
    with request_context(session_id=session_id, user_id="u5", request_id="r", ticker="NVDA"):
        with trace.get_tracer("lg_tracing").start_as_current_span("caller") as caller:
            return inject_carrier(), caller


async def post(app, json, headers=None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://decision") as client:
        return await client.post("/invoke", json=json, headers=headers or {})


async def test_invoke_continues_trace_from_headers(exporter):
    model = ScriptedChatModel(
        responses=[tool_call("force_index", {"ticker": "NVDA"}), submit()]
    )
    headers, caller = caller_headers("s5")
    resp = await post(create_app(model=model, tools=[force_index]), BODY, headers)
    assert resp.status_code == 200
    assert resp.json() == {"decision": "buy", "confidence": 0.7, "rationale": "FI up"}
    root = only_span(exporter, "decision_graph")
    assert root.parent.span_id == caller.get_span_context().span_id
    assert only_span(exporter, "force_index").status.status_code == StatusCode.OK
    inner = [s for s in exporter.get_finished_spans() if s.name != "caller"]
    assert inner and all(s.attributes["session.id"] == "s5" for s in inner)


async def test_direct_call_without_traceparent_starts_own_trace(exporter):
    model = ScriptedChatModel(responses=[submit(id="c1")])
    resp = await post(create_app(model=model, tools=[force_index]), BODY)
    assert resp.status_code == 200
    root = only_span(exporter, "decision_graph")
    assert root.parent is None
    assert root.attributes["user.id"] == "anonymous"


async def test_model_never_submits_returns_502(exporter):
    model = ScriptedChatModel(responses=[AIMessage("I think buy.")])
    resp = await post(create_app(model=model, tools=[force_index]), BODY)
    assert resp.status_code == 502


async def test_endless_tool_loop_returns_502(exporter):
    model = ScriptedChatModel(
        responses=[tool_call("force_index", {"ticker": "NVDA"}, id=f"c{i}") for i in range(30)]
    )
    resp = await post(create_app(model=model, tools=[force_index]), BODY)
    assert resp.status_code == 502


async def test_tool_error_goes_back_to_model(exporter):
    seen: list[list] = []

    class Recording(ScriptedChatModel):
        def _generate(self, messages, *args, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    model = Recording(responses=[tool_call("force_index", {"ticker": "NVDA"}), submit()])
    resp = await post(create_app(model=model, tools=[failing]), BODY)
    assert resp.status_code == 200
    tool_messages = [m for m in seen[-1] if isinstance(m, ToolMessage)]
    assert [m.status for m in tool_messages] == ["error"]
    assert "upstream data feed down" in tool_messages[0].content
    assert only_span(exporter, "force_index").status.status_code == StatusCode.ERROR


@pytest.mark.parametrize("ticker", ["nvda ", " brk.b"])
async def test_ticker_is_normalised(exporter, ticker):
    model = ScriptedChatModel(responses=[submit(id="c1")])
    resp = await post(
        create_app(model=model, tools=[force_index]), {"ticker": ticker, "research_brief": "b"}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("ticker", ["", "NVDA!", "TOOLONGTICKER", "123"])
async def test_bad_ticker_returns_422(exporter, ticker):
    model = ScriptedChatModel(responses=[])
    resp = await post(
        create_app(model=model, tools=[force_index]), {"ticker": ticker, "research_brief": "b"}
    )
    assert resp.status_code == 422


async def test_concurrent_requests_keep_their_own_session(exporter):
    headers_a, caller_a = caller_headers("A")
    headers_b, caller_b = caller_headers("B")

    def app_for():
        model = ScriptedChatModel(
            responses=[tool_call("force_index", {"ticker": "NVDA"}), submit()]
        )
        return create_app(model=model, tools=[force_index])

    await asyncio.gather(
        post(app_for(), BODY, headers_a), post(app_for(), BODY, headers_b)
    )
    for session, caller in (("A", caller_a), ("B", caller_b)):
        trace_id = caller.get_span_context().trace_id
        inner = [
            s
            for s in exporter.get_finished_spans()
            if s.context.trace_id == trace_id and s.name != "caller"
        ]
        assert any(s.name == "decision_graph" for s in inner)
        assert all(s.attributes["session.id"] == session for s in inner)


async def test_health():
    app = create_app(model=ScriptedChatModel(responses=[]), tools=[force_index])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://decision") as client:
        resp = await client.get("/health")
    assert resp.json() == {"status": "ok"}
