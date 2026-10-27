import httpx
import pytest
from opentelemetry.trace import StatusCode
from langchain_core.messages import AIMessage, ToolMessage

from research_agent.app import create_app
from research_agent.graph import build_research_graph
from research_agent.web_search import make_web_search_tool
from tests.fakes import ScriptedChatModel, tool_call
from tests.helpers import only_span
from websearch_mcp.server import create_server

DECISION = {"decision": "hold", "confidence": 0.5, "rationale": "r"}


async def fake_search(query: str, max_results: int) -> list[dict]:
    return [{"title": f"{query} beats", "url": "https://example.com", "content": "c"}]


def make_app(model, *, search=fake_search, decision_status=200, decision_body=None, recorded=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if recorded is not None:
            recorded.update(request.headers)
        if decision_status != 200:
            return httpx.Response(decision_status, text="down")
        if decision_body is not None:
            return httpx.Response(200, content=decision_body)
        return httpx.Response(200, json=DECISION)

    graph = build_research_graph(
        model,
        web_search=make_web_search_tool(create_server(search)),
        decision_url="http://decision",
        decision_transport=httpx.MockTransport(handler),
    )
    return create_app(graph)


def happy_model(**kwargs):
    return ScriptedChatModel(
        responses=[
            tool_call("web_search", {"query": "NVDA"}),
            AIMessage("notes"),
            AIMessage("Brief: ok"),
        ]
    )


async def post(app, json):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://research") as client:
        return await client.post("/invoke", json=json)


async def test_happy_path_returns_contract_and_trace_id(exporter):
    resp = await post(make_app(happy_model()), {"ticker": "NVDA"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "ticker": "NVDA",
        "decision": "hold",
        "confidence": 0.5,
        "rationale": "r",
        "research_brief": "Brief: ok",
        "trace_id": body["trace_id"],
    }
    assert int(body["trace_id"], 16) == only_span(exporter, "research_graph").context.trace_id


async def test_decision_call_carries_call_decision_span_as_parent(exporter):
    recorded: dict[str, str] = {}
    resp = await post(make_app(happy_model(), recorded=recorded), {"ticker": "NVDA"})
    assert resp.status_code == 200
    parent_id = recorded["traceparent"].split("-")[2]
    assert parent_id == format(only_span(exporter, "call_decision").context.span_id, "016x")
    assert "session_id=" in recorded["baggage"]


async def test_mcp_server_span_is_child_of_client_tool_span(exporter):
    await post(make_app(happy_model()), {"ticker": "NVDA"})
    client = only_span(exporter, "web_search", scope="openinference.instrumentation.langchain")
    server = only_span(exporter, "web_search", scope="lg_tracing")
    assert server.parent.span_id == client.context.span_id
    assert server.context.trace_id == client.context.trace_id


async def test_mcp_failure_reaches_model_as_tool_error(exporter):
    seen: list[list] = []

    class Recording(ScriptedChatModel):
        def _generate(self, messages, *args, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    async def broken(query: str, max_results: int) -> list[dict]:
        raise RuntimeError("tavily down")

    model = Recording(
        responses=[
            tool_call("web_search", {"query": "NVDA"}),
            AIMessage("notes"),
            AIMessage("Brief: ok"),
        ]
    )
    resp = await post(make_app(model, search=broken), {"ticker": "NVDA"})
    assert resp.status_code == 200
    tool_messages = [m for m in seen[1] if isinstance(m, ToolMessage)]
    assert [m.status for m in tool_messages] == ["error"]
    # The MCP server deliberately masks the raw exception text.
    assert "Error executing tool web_search" in tool_messages[0].content
    client = only_span(exporter, "web_search", scope="openinference.instrumentation.langchain")
    assert client.status.status_code == StatusCode.ERROR


async def test_decision_agent_down_returns_502_with_trace_id(exporter):
    resp = await post(make_app(happy_model(), decision_status=503), {"ticker": "NVDA"})
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert int(detail["trace_id"], 16) == only_span(exporter, "research_graph").context.trace_id
    assert detail["error"]


@pytest.mark.parametrize("body", ["not json", '{"decision": "hold"}'])
async def test_malformed_decision_response_returns_502_with_trace_id(exporter, body):
    resp = await post(make_app(happy_model(), decision_body=body), {"ticker": "NVDA"})
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert int(detail["trace_id"], 16) == only_span(exporter, "research_graph").context.trace_id


async def test_lowercase_ticker_is_normalised(exporter):
    resp = await post(make_app(happy_model()), {"ticker": " nvda "})
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "NVDA"


@pytest.mark.parametrize("ticker", ["", "NVDA!", "TOOLONGTICKER", "123"])
async def test_bad_ticker_returns_422(exporter, ticker):
    resp = await post(make_app(ScriptedChatModel(responses=[])), {"ticker": ticker})
    assert resp.status_code == 422


async def test_health():
    app = make_app(ScriptedChatModel(responses=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://research") as client:
        resp = await client.get("/health")
    assert resp.json() == {"status": "ok"}
