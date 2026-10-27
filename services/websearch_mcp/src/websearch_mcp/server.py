"""MCP server exposing a traced `web_search` tool backed by Tavily."""

import json
import os
from collections.abc import Awaitable, Callable
from typing import Annotated

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from lg_tracing import continue_request_context, set_output, setup_tracing, tool_span

SearchFn = Callable[[str, int], Awaitable[list[dict]]]

WEB_SEARCH_DESCRIPTION = (
    "Search the web for recent financial news, earnings, analyst opinions and other information "
    "about a company or stock. Returns a JSON list of {title, url, content} results."
)
MAX_CONTENT_CHARS = 1500


def create_server(search: SearchFn) -> MCPServer:
    server = MCPServer("websearch")

    @server.tool(name="web_search", description=WEB_SEARCH_DESCRIPTION)
    async def web_search(
        query: Annotated[str, Field(description="Search query")],
        ctx: Context,
        max_results: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> str:
        # The caller put traceparent + baggage into the request's _meta.
        with continue_request_context(ctx.request_context.meta or {}):
            with tool_span("web_search", {"query": query, "max_results": max_results}) as span:
                results = await search(query, max_results)
                set_output(span, results)
                return json.dumps(results)

    return server


async def tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = await client.search(query=query, max_results=max_results, topic="finance")
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": (item.get("content") or "")[:MAX_CONTENT_CHARS],
        }
        for item in response.get("results", [])
    ]


def main() -> None:
    if not os.getenv("TAVILY_API_KEY"):
        raise SystemExit("TAVILY_API_KEY is required")
    provider = setup_tracing("websearch-mcp")
    try:
        create_server(tavily_search).run(
            transport="streamable-http", host="0.0.0.0", port=int(os.getenv("PORT", "8000"))
        )
    finally:
        provider.shutdown()


if __name__ == "__main__":
    main()
