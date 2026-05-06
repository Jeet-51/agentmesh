"""
Wikipedia MCP tool server — port 8012.

Exposes two MCP tools:
  search(query, n_results)           — search Wikipedia and return page titles + excerpts
  get_summary(page_title, sentences) — fetch the opening summary of a specific page

Transport: MCP over SSE using a pure ASGI app (mcp>=1.24 compatible).
"""

from __future__ import annotations

import json
import os

import structlog
import uvicorn
import wikipediaapi
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

log = structlog.get_logger(__name__)

_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AgentMesh/1.0 (portfolio project; contact via GitHub)",
)

server = Server("wikipedia-tool")
# Disable DNS rebinding protection — safe for Docker internal networking where
# containers connect by hostname (e.g., wikipedia-tool:8012).
transport = SseServerTransport(
    "/messages/",
    security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search",
            description=(
                "Search Wikipedia for pages related to a query. "
                "Returns a list of page titles with their opening excerpt."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "n_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-10). Defaults to 5.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_summary",
            description=(
                "Fetch the introductory summary of a specific Wikipedia page. "
                "Use the exact page title returned by the search tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_title": {
                        "type": "string",
                        "description": "Exact Wikipedia page title.",
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Approximate sentences to return (1-20). Defaults to 8.",
                        "default": 8,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["page_title"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    args = arguments or {}
    log.info("wikipedia.tool.call", tool=name, args=args)

    if name == "search":
        result = _search(args.get("query", ""), int(args.get("n_results", 5)))
    elif name == "get_summary":
        result = _get_summary(args.get("page_title", ""), int(args.get("sentences", 8)))
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _search(query: str, n_results: int) -> dict:
    if not query:
        return {"error": "query is required"}
    try:
        import requests
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": min(n_results, 10),
                "namespace": 0,
                "format": "json",
            },
            timeout=10,
            headers={"User-Agent": "AgentMesh/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        titles, descriptions, urls = data[1], data[2], data[3]
        results = [
            {"title": t, "excerpt": d or "(no excerpt)", "url": u}
            for t, d, u in zip(titles, descriptions, urls)
        ]
        log.info("wikipedia.search.ok", query=query, count=len(results))
        return {"query": query, "results": results, "source": "wikipedia"}
    except Exception as exc:
        log.error("wikipedia.search.error", query=query, error=str(exc))
        return {"error": str(exc), "query": query}


def _get_summary(page_title: str, sentences: int) -> dict:
    if not page_title:
        return {"error": "page_title is required"}
    try:
        page = _wiki.page(page_title)
        if not page.exists():
            return {"error": f"Page '{page_title}' does not exist.", "page_title": page_title}
        full_text = page.summary
        trimmed = ". ".join(full_text.split(". ")[:sentences])
        if not trimmed.endswith("."):
            trimmed += "."
        log.info("wikipedia.get_summary.ok", page_title=page_title, chars=len(trimmed))
        return {
            "page_title": page.title,
            "summary": trimmed,
            "url": page.fullurl,
            "source": "wikipedia",
        }
    except Exception as exc:
        log.error("wikipedia.get_summary.error", page_title=page_title, error=str(exc))
        return {"error": str(exc), "page_title": page_title}


# ---------------------------------------------------------------------------
# Pure ASGI app — direct routing preserves scope["root_path"] = "" so that
# SseServerTransport.connect_sse() advertises "/messages/" (not "/sse/messages/")
# to the MCP client as the POST endpoint.
# DNS rebinding protection is disabled via TransportSecuritySettings above so
# Docker container hostnames (wikipedia-tool:8012) are accepted.
# ---------------------------------------------------------------------------

async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        await receive()
        await send({"type": "lifespan.startup.complete"})
        await receive()
        await send({"type": "lifespan.shutdown.complete"})
        return

    path   = scope.get("path", "")
    method = scope.get("method", "GET")

    if path == "/sse" and method == "GET":
        async with transport.connect_sse(scope, receive, send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    elif path.startswith("/messages/"):
        await transport.handle_post_message(scope, receive, send)

    elif path == "/health":
        response = JSONResponse({"status": "ok", "service": "wikipedia-mcp-tool", "port": 8012})
        await response(scope, receive, send)

    else:
        response = JSONResponse({"error": "not found"}, status_code=404)
        await response(scope, receive, send)


if __name__ == "__main__":
    uvicorn.run(
        "wikipedia_tool:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8012)),
        reload=False,
    )
