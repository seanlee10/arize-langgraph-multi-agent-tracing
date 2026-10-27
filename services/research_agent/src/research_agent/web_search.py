"""LangChain tool that calls the remote websearch MCP server, propagating trace context in `_meta`."""

from typing import Annotated

from langchain_core.tools import BaseTool, ToolException, tool
from mcp.client import Client
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from lg_tracing import inject_carrier


def make_web_search_tool(server: str | MCPServer, *, timeout_seconds: float = 30.0) -> BaseTool:
    @tool("web_search")
    async def web_search(
        query: Annotated[str, Field(description="Search query")],
        max_results: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> str:
        """Search the web for recent financial news, earnings, analyst opinions and other information about a company or stock. Returns a JSON list of {title, url, content} results."""
        # Called from inside the tool, so the carrier parents the server span to this tool span.
        async with Client(server) as client:
            result = await client.call_tool(
                "web_search",
                {"query": query, "max_results": max_results},
                read_timeout_seconds=timeout_seconds,
                meta=inject_carrier(),
            )
        text = "".join(block.text for block in result.content if getattr(block, "text", None))
        if result.is_error:
            raise ToolException(text or "web_search failed")
        return text

    web_search.handle_tool_error = True
    return web_search
