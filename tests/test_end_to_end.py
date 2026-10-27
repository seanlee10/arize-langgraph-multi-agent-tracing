"""One POST to the research agent must yield ONE connected trace across all three services.

Research agent -> (MCP web_search) -> websearch server, and
research agent -> (HTTP + traceparent/baggage) -> decision agent, all in-process.
"""

import json

import httpx
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from opentelemetry.trace import StatusCode

import decision_agent.app as decision_app_module
import research_agent.app as research_app_module
from lg_tracing import TAGS
from lg_tracing.provider import ALLOWED_TRACERS
from research_agent.graph import build_research_graph
from research_agent.web_search import make_web_search_tool
from tests.fakes import ScriptedChatModel, tool_call
from tests.helpers import only_span
from websearch_mcp.server import create_server


@tool
def force_index(ticker: str, period: int = 13) -> dict:
    """Fake force index."""
    return {"trend": "bullish"}


async def fake_search(query: str, max_results: int) -> list[dict]:
    return [{"title": f"{query} beats", "url": "https://example.com", "content": "c"}]


async def test_one_request_produces_one_connected_trace(exporter):
    decision_model = ScriptedChatModel(
        responses=[
            tool_call("force_index", {"ticker": "NVDA"}),
            tool_call(
                "submit_decision",
                {"decision": "buy", "confidence": 0.7, "rationale": "FI up"},
                id="c2",
            ),
        ]
    )
    decision_app = decision_app_module.create_app(model=decision_model, tools=[force_index])

    research_model = ScriptedChatModel(
        responses=[
            tool_call("web_search", {"query": "NVDA"}),
            AIMessage("notes"),
            AIMessage("Brief: ok"),
        ]
    )
    graph = build_research_graph(
        research_model,
        web_search=make_web_search_tool(create_server(fake_search)),
        decision_url="http://decision",
        decision_transport=httpx.ASGITransport(app=decision_app),
    )
    research_app = research_app_module.create_app(graph)

    transport = httpx.ASGITransport(app=research_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://research") as client:
        resp = await client.post(
            "/invoke",
            json={"ticker": "nvda", "session_id": "sess-42", "user_id": "analyst-1"},
        )

    assert resp.status_code == 200 and resp.json()["decision"] == "buy"
    spans = exporter.get_finished_spans()
    assert {s.context.trace_id for s in spans} == {int(resp.json()["trace_id"], 16)}
    for s in spans:
        assert s.attributes["session.id"] == "sess-42", s.name
        assert s.attributes["user.id"] == "analyst-1", s.name
        assert json.loads(s.attributes["metadata"])["ticker"] == "NVDA", s.name
        assert list(s.attributes["tag.tags"]) == TAGS, s.name
        assert s.status.status_code is not StatusCode.ERROR, s.name
        assert s.instrumentation_scope.name in ALLOWED_TRACERS, s.name
    research_root = only_span(exporter, "research_graph")
    assert research_root.parent is None
    decision_root = only_span(exporter, "decision_graph")
    assert decision_root.parent.span_id == only_span(exporter, "call_decision").context.span_id
    server = only_span(exporter, "web_search", scope="lg_tracing")
    client = only_span(exporter, "web_search", scope="openinference.instrumentation.langchain")
    assert server.parent.span_id == client.context.span_id
