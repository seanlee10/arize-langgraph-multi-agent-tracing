"""Contract: what LangChainInstrumentor does to a LangGraph run (spec 3.3, risk items 1-3).

The design adds no manual spans around graphs, nodes or tools; it relies on these behaviours.

Finding: the instrumentor creates node/tool spans from LangChain callbacks but never attaches
them as the current OTel span (see _start_trace in its _tracer.py). Inside a node, the plain
current context is still whatever surrounded ainvoke(). So lg_tracing resolves the running
span with openinference.instrumentation.langchain.get_current_span() before injecting the
carrier or reading the trace id.

Tested versions (uv.lock):
    langgraph                                1.2.18
    langchain-core                           1.7.0
    openinference-instrumentation-langchain  0.1.80
    opentelemetry-sdk                        1.45.1
"""

from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from lg_tracing.propagation import continue_request_context, inject_carrier, request_context
from lg_tracing.spans import current_trace_id
from tests.fakes import ScriptedChatModel, tool_call
from tests.helpers import only_span

INSTRUMENTOR_SCOPE = "openinference.instrumentation.langchain"
tracer = trace.get_tracer("lg_tracing")


class State(TypedDict):
    value: str


def parent_id(carrier: dict[str, str]) -> str:
    return carrier["traceparent"].split("-")[2]


async def run_from_upstream(graph, state):
    """Run graph as a downstream service would: continuing a carrier from an upstream span."""
    with request_context(session_id="sess-c", user_id="user-c", request_id="req-c", ticker="NVDA"):
        with tracer.start_as_current_span("upstream") as upstream:
            carrier = inject_carrier()
    with continue_request_context(carrier):
        await graph.ainvoke(state)
    return upstream


def two_node_graph(outbound):
    builder = StateGraph(State)
    builder.add_node("outbound", outbound)
    builder.add_node("finish", lambda state: {"value": state["value"] + "!"})
    builder.add_edge(START, "outbound")
    builder.add_edge("outbound", "finish")
    builder.add_edge("finish", END)
    return builder.compile(name="contract_graph")


def agent_graph(probe):
    model = ScriptedChatModel(responses=[tool_call("probe", {}), AIMessage("done")])

    async def agent(state: MessagesState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([probe]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(name="contract_graph")


def probe_tool(kind: str, captured: list[str]):
    if kind == "async":

        @tool
        async def probe() -> str:
            """Record the traceparent an outbound call would send."""
            captured.append(parent_id(inject_carrier()))
            return "ok"
    else:

        @tool
        def probe() -> str:
            """Record the traceparent an outbound call would send."""
            captured.append(parent_id(inject_carrier()))
            return "ok"

    return probe


def outbound_node(kind: str, captured: list[str]):
    if kind == "async":

        async def outbound(state: State):
            captured.append(parent_id(inject_carrier()))
            return {"value": state["value"]}
    else:

        def outbound(state: State):
            captured.append(parent_id(inject_carrier()))
            return {"value": state["value"]}

    return outbound


def span_hex(span) -> str:
    return format(span.context.span_id, "016x")


def instrumented_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.instrumentation_scope.name == INSTRUMENTOR_SCOPE]


async def test_graph_root_is_child_of_extracted_context(exporter):
    upstream = await run_from_upstream(two_node_graph(outbound_node("async", [])), {"value": "x"})

    root = only_span(exporter, "contract_graph", scope=INSTRUMENTOR_SCOPE)
    assert root.parent.span_id == upstream.get_span_context().span_id
    assert {s.context.trace_id for s in exporter.get_finished_spans()} == {
        upstream.get_span_context().trace_id
    }


@pytest.mark.parametrize("node_kind", ["async", "sync"])
async def test_inject_inside_node_points_at_node_span(exporter, node_kind):
    captured: list[str] = []
    await run_from_upstream(two_node_graph(outbound_node(node_kind, captured)), {"value": "x"})

    assert captured == [span_hex(only_span(exporter, "outbound", scope=INSTRUMENTOR_SCOPE))]


@pytest.mark.parametrize("tool_kind", ["async", "sync"])
async def test_inject_inside_toolnode_tool_points_at_tool_span(exporter, tool_kind):
    captured: list[str] = []
    await run_from_upstream(agent_graph(probe_tool(tool_kind, captured)), {"messages": [HumanMessage("go")]})

    assert captured == [span_hex(only_span(exporter, "probe", scope=INSTRUMENTOR_SCOPE))]


async def test_current_trace_id_inside_node_is_graph_trace(exporter):
    trace_ids: list[str] = []

    def outbound(state: State):
        trace_ids.append(current_trace_id())
        return {"value": state["value"]}

    await run_from_upstream(two_node_graph(outbound), {"value": "x"})

    root = only_span(exporter, "contract_graph", scope=INSTRUMENTOR_SCOPE)
    assert trace_ids == [format(root.context.trace_id, "032x")]


async def test_request_attributes_reach_every_instrumented_span(exporter):
    await run_from_upstream(agent_graph(probe_tool("async", [])), {"messages": [HumanMessage("go")]})

    spans = instrumented_spans(exporter)
    assert spans
    for s in spans:
        assert s.attributes["session.id"] == "sess-c", s.name
        assert s.attributes["user.id"] == "user-c", s.name


async def test_instrumented_spans_have_explicit_status(exporter):
    await run_from_upstream(agent_graph(probe_tool("async", [])), {"messages": [HumanMessage("go")]})
    not_ok = [(s.name, s.status.status_code) for s in instrumented_spans(exporter) if s.status.status_code != StatusCode.OK]
    assert not_ok == []

    exporter.clear()

    def outbound(state: State):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await run_from_upstream(two_node_graph(outbound), {"value": "x"})
    assert only_span(exporter, "outbound", scope=INSTRUMENTOR_SCOPE).status.status_code == StatusCode.ERROR
    assert only_span(exporter, "contract_graph", scope=INSTRUMENTOR_SCOPE).status.status_code == StatusCode.ERROR
