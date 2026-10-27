"""LangGraph agent: the model calls indicator tools, then finishes by calling submit_decision."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from decision_agent.tools import INDICATOR_TOOLS, submit_decision

SYSTEM_PROMPT = (
    "You are a trading decision agent. You receive a ticker and a research brief. "
    "Use the technical-indicator and insider-trade tools to test the brief against market data, "
    "then finish by calling submit_decision with decision (buy, hold or sell), a confidence "
    "between 0 and 1, and a short rationale. You must end by calling submit_decision."
)
RECURSION_LIMIT = 25


class DecisionNotSubmitted(Exception):
    """The agent ended without a successful submit_decision call."""


def build_decision_graph(
    model: BaseChatModel, tools: list[BaseTool] = INDICATOR_TOOLS
) -> CompiledStateGraph:
    model_with_tools = model.bind_tools([*tools, submit_decision])

    async def agent(state: MessagesState) -> dict[str, Any]:
        reply = await model_with_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def after_agent(state: MessagesState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: MessagesState) -> str:
        # The AI message that requested these tools sits just before the tool results.
        ai_message = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        submit_ids = {call["id"] for call in ai_message.tool_calls if call["name"] == "submit_decision"}
        # A rejected submit (e.g. bad args) goes back to the model like any other tool error.
        submitted = any(
            isinstance(m, ToolMessage) and m.tool_call_id in submit_ids and m.status != "error"
            for m in state["messages"]
        )
        return END if submitted else "agent"

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([*tools, submit_decision], handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, ["tools", END])
    builder.add_conditional_edges("tools", after_tools, ["agent", END])
    return builder.compile(name="decision_graph")


async def run_decision(graph: CompiledStateGraph, ticker: str, research_brief: str) -> dict[str, Any]:
    prompt = f"Ticker: {ticker}\n\nResearch brief:\n{research_brief}"
    try:
        state = await graph.ainvoke(
            {"messages": [HumanMessage(prompt)]}, config={"recursion_limit": RECURSION_LIMIT}
        )
    except GraphRecursionError as exc:
        raise DecisionNotSubmitted("agent hit the step limit without submitting") from exc

    messages = state["messages"]
    accepted = {m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.status != "error"}
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call["name"] == "submit_decision" and call["id"] in accepted:
                args = call["args"]
                return {
                    "decision": args["decision"],
                    "confidence": args["confidence"],
                    "rationale": args["rationale"],
                }
    raise DecisionNotSubmitted("agent finished without calling submit_decision")
