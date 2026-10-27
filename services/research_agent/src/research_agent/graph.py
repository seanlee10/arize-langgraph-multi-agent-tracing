"""LangGraph research flow: search the web via MCP, summarize, then ask decision-agent for a call."""

from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from lg_tracing import current_trace_id, inject_carrier

RESEARCH_PROMPT = (
    "You are an equity research agent. Use the web_search tool to gather recent news, earnings "
    "and analyst opinion about the requested ticker. Search as many times as you need, then "
    "reply without calling a tool once you have enough material."
)
SUMMARIZE_PROMPT = (
    "Write a concise research brief on the ticker from the research material provided: key facts, "
    "catalysts, risks and overall sentiment. Do not give a buy/hold/sell recommendation."
)
DECISION_TIMEOUT_SECONDS = 120


class DecisionAgentError(Exception):
    def __init__(self, message: str, *, trace_id: str):
        super().__init__(message)
        self.trace_id = trace_id


class ResearchState(MessagesState):
    ticker: str
    research_brief: str
    decision: dict
    trace_id: str


def build_research_graph(
    model: BaseChatModel,
    *,
    web_search: BaseTool,
    decision_url: str,
    decision_transport: httpx.AsyncBaseTransport | None = None,
) -> CompiledStateGraph:
    researcher = model.bind_tools([web_search])

    async def research(state: ResearchState) -> dict[str, Any]:
        reply = await researcher.ainvoke([SystemMessage(RESEARCH_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def after_research(state: ResearchState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "summarize"

    async def summarize(state: ResearchState) -> dict[str, Any]:
        # Pass the material as text: replaying tool_use/thinking blocks without the same tools 400s.
        results = [m.text for m in state["messages"] if isinstance(m, ToolMessage)]
        notes = state["messages"][-1].text
        material = "\n\n".join(
            [f"Ticker: {state['ticker']}", "Search results:", *results, "Research notes:", notes]
        )
        reply = await model.ainvoke([SystemMessage(SUMMARIZE_PROMPT), HumanMessage(material)])
        return {"research_brief": str(reply.text)}

    async def call_decision(state: ResearchState) -> dict[str, Any]:
        # inject_carrier() must run inside this node so decision-agent parents to this span.
        trace_id = current_trace_id()
        try:
            async with httpx.AsyncClient(transport=decision_transport) as client:
                response = await client.post(
                    f"{decision_url}/invoke",
                    json={"ticker": state["ticker"], "research_brief": state["research_brief"]},
                    headers=inject_carrier(),
                    timeout=DECISION_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                decision = response.json()
                missing = {"decision", "confidence", "rationale"} - set(decision)
                if missing:
                    raise ValueError(f"decision-agent response missing {sorted(missing)}")
                return {"decision": decision, "trace_id": trace_id}
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise DecisionAgentError(str(exc), trace_id=trace_id) from exc

    builder = StateGraph(ResearchState)
    builder.add_node("research", research)
    builder.add_node("tools", ToolNode([web_search], handle_tool_errors=True))
    builder.add_node("summarize", summarize)
    builder.add_node("call_decision", call_decision)
    builder.add_edge(START, "research")
    builder.add_conditional_edges("research", after_research, ["tools", "summarize"])
    builder.add_edge("tools", "research")
    builder.add_edge("summarize", "call_decision")
    builder.add_edge("call_decision", END)
    return builder.compile(name="research_graph")
