import json

from mcp.client import Client
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from lg_tracing import inject_carrier, request_context
from lg_tracing.provider import ALLOWED_TRACERS
from tests.helpers import only_span
from websearch_mcp.server import create_server


async def fake_search(query: str, max_results: int) -> list[dict]:
    return [{"title": f"{query} beats", "url": "https://example.com", "content": "c"}][:max_results]


async def test_server_span_continues_callers_trace(exporter):
    with request_context(session_id="s9", user_id="u9", request_id="r", ticker="NVDA"):
        with trace.get_tracer("lg_tracing").start_as_current_span("caller") as caller:
            async with Client(create_server(fake_search)) as c:
                result = await c.call_tool("web_search", {"query": "NVDA"}, meta=inject_carrier())

    assert json.loads(result.content[0].text)[0]["title"] == "NVDA beats"
    server = only_span(exporter, "web_search", scope="lg_tracing")
    assert server.context.trace_id == caller.get_span_context().trace_id
    assert server.parent.span_id == caller.get_span_context().span_id
    assert server.attributes["session.id"] == "s9"
    assert server.attributes["openinference.span.kind"] == "TOOL"
    assert server.attributes["tool.name"] == "web_search"
    assert json.loads(server.attributes["input.value"]) == {"query": "NVDA", "max_results": 5}
    assert json.loads(server.attributes["output.value"])[0]["title"] == "NVDA beats"
    assert server.status.status_code is StatusCode.OK


async def test_no_mcp_sdk_spans_recorded(exporter):
    async with Client(create_server(fake_search)) as c:
        await c.call_tool("web_search", {"query": "x"})

    scopes = {s.instrumentation_scope.name for s in exporter.get_finished_spans()}
    assert scopes and scopes <= ALLOWED_TRACERS


async def test_search_failure_marks_server_span_error(exporter):
    async def broken(query: str, max_results: int) -> list[dict]:
        raise RuntimeError("tavily down")

    async with Client(create_server(broken)) as c:
        result = await c.call_tool("web_search", {"query": "x"})

    assert result.is_error
    span = only_span(exporter, "web_search", scope="lg_tracing")
    assert span.status.status_code is StatusCode.ERROR
    assert "tavily down" in span.status.description
