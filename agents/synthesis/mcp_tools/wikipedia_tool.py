"""
Wikipedia MCP tool server — port 8012.

Exposes two MCP tools:
  search(query, n_results)         — search Wikipedia and return page titles + excerpts
  get_summary(page_title, sentences) — fetch the opening summary of a specific page

Transport: MCP over SSE (GET /sse + POST /messages).
Uses the wikipedia-api library (no API key required).
"""

from __future__ import annotations

import json
import os

import structlog
import uvicorn
import wikipediaapi
from fastapi import FastAPI, Request
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport

log = structlog.get_logger(__name__)

# Wikipedia API client — set a descriptive user-agent as required by Wikimedia policy.
_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AgentMesh/1.0 (portfolio project; contact via GitHub)",
)

server = Server("wikipedia-tool")
transport = SseServerTransport("/messages/")


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
                "Returns a list of page titles with their opening excerpt. "
                "Use this to discover relevant Wikipedia articles before fetching full summaries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-10). Defaults to 5.",
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
                "Use the exact page title returned by the search tool. "
                "Returns the opening paragraphs plus the page URL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_title": {
                        "type": "string",
                        "description": "Exact Wikipedia page title (as returned by the search tool).",
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Approximate number of sentences to return (1-20). Defaults to 8.",
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
    """Search Wikipedia using the MediaWiki API via wikipedia-api."""
    if not query:
        return {"error": "query is required"}
    try:
        import requests

        # Use the MediaWiki opensearch API for title suggestions.
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

        titles: list[str] = data[1]
        descriptions: list[str] = data[2]
        urls: list[str] = data[3]

        results = [
            {
                "title": title,
                "excerpt": desc or "(no excerpt)",
                "url": url,
            }
            for title, desc, url in zip(titles, descriptions, urls)
        ]

        log.info("wikipedia.search.ok", query=query, count=len(results))
        return {"query": query, "results": results, "source": "wikipedia"}

    except Exception as exc:
        log.error("wikipedia.search.error", query=query, error=str(exc))
        return {"error": str(exc), "query": query}


def _get_summary(page_title: str, sentences: int) -> dict:
    """Fetch the introductory summary of a Wikipedia page."""
    if not page_title:
        return {"error": "page_title is required"}
    try:
        page = _wiki.page(page_title)

        if not page.exists():
            log.warning("wikipedia.get_summary.not_found", page_title=page_title)
            return {
                "error": f"Page '{page_title}' does not exist on Wikipedia.",
                "page_title": page_title,
            }

        # Trim to approximately `sentences` sentences by splitting on ". ".
        full_text = page.summary
        sentence_list = full_text.split(". ")
        trimmed = ". ".join(sentence_list[:sentences])
        if not trimmed.endswith("."):
            trimmed += "."

        log.info(
            "wikipedia.get_summary.ok",
            page_title=page_title,
            chars=len(trimmed),
        )
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
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Wikipedia MCP Tool", version="1.0.0")


@app.get("/sse")
async def sse_endpoint(request: Request) -> None:
    async with transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


@app.post("/messages/")
async def handle_messages(request: Request) -> None:
    await transport.handle_post_message(
        request.scope, request.receive, request._send
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "wikipedia-mcp-tool", "port": 8012}


if __name__ == "__main__":
    uvicorn.run(
        "wikipedia_tool:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8012)),
        reload=False,
    )
