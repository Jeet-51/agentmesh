"""
SEC EDGAR MCP tool server — port 8013.

Exposes two MCP tools:
  search_filings(company_name, form_type)  — full-text search EDGAR for filings
  get_10k(ticker)                          — fetch the most recent 10-K filing metadata
                                             and key sections (business description,
                                             risk factors header, financial highlights)

Transport: MCP over SSE (GET /sse + POST /messages).
Uses the free SEC EDGAR REST API — no API key required.
Per SEC fair-use policy: User-Agent must identify the application.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import structlog
import uvicorn
from fastapi import FastAPI, Request
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport

log = structlog.get_logger(__name__)

# SEC EDGAR requires a descriptive User-Agent.
_HEADERS = {
    "User-Agent": "AgentMesh/1.0 portfolio-project (contact via GitHub)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
_EFTS_HEADERS = {
    "User-Agent": "AgentMesh/1.0 portfolio-project (contact via GitHub)",
}

_EDGAR_BASE = "https://data.sec.gov"
_EFTS_BASE = "https://efts.sec.gov"
_SUBMISSIONS_URL = f"{_EDGAR_BASE}/submissions/CIK{{cik:010d}}.json"

server = Server("edgar-tool")
transport = SseServerTransport("/messages/")


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_filings",
            description=(
                "Search SEC EDGAR full-text search for filings by a company name or keyword. "
                "Returns a list of matching filings with accession numbers, form types, "
                "filing dates, and company names. Useful for discovering what filings exist "
                "before fetching specific ones."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name or keyword to search for.",
                    },
                    "form_type": {
                        "type": "string",
                        "description": "SEC form type filter, e.g. '10-K', '10-Q', '8-K'. Defaults to '10-K'.",
                        "default": "10-K",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Max results to return (1-10). Defaults to 5.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["company_name"],
            },
        ),
        types.Tool(
            name="get_10k",
            description=(
                "Retrieve metadata and key sections from the most recent annual report (10-K) "
                "for a company identified by its SEC CIK number or ticker. Returns the filing "
                "date, accession number, reporting period, and links to the document index. "
                "Also returns available section headers to guide further research."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL) or company name.",
                    },
                },
                "required": ["ticker"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    args = arguments or {}
    log.info("edgar.tool.call", tool=name, args={k: v for k, v in args.items() if k != "content"})

    if name == "search_filings":
        result = _search_filings(
            company_name=args.get("company_name", ""),
            form_type=args.get("form_type", "10-K"),
            n_results=int(args.get("n_results", 5)),
        )
    elif name == "get_10k":
        result = _get_10k(ticker=args.get("ticker", ""))
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _search_filings(company_name: str, form_type: str, n_results: int) -> dict[str, Any]:
    """
    Search EDGAR full-text search (EFTS) for filings matching a company name.

    Uses the public EFTS endpoint — no auth required.
    """
    if not company_name:
        return {"error": "company_name is required"}

    try:
        resp = requests.get(
            f"{_EFTS_BASE}/LATEST/search-index",
            params={
                "q": f'"{company_name}"',
                "forms": form_type,
                "dateRange": "custom",
                "startdt": "2020-01-01",
                "hits.hits.total.value": n_results,
            },
            headers=_EFTS_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("hits", {}).get("hits", [])[:n_results]
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            results.append(
                {
                    "accession_no": src.get("accession_no"),
                    "form_type": src.get("form_type"),
                    "filed_at": src.get("file_date"),
                    "company_name": src.get("entity_name"),
                    "cik": src.get("entity_id"),
                    "period_of_report": src.get("period_of_report"),
                    "document_url": (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{src.get('entity_id', '')}/{src.get('accession_no', '').replace('-', '')}"
                    ),
                }
            )

        log.info("edgar.search_filings.ok", company=company_name, count=len(results))
        return {
            "query": company_name,
            "form_type": form_type,
            "results": results,
            "source": "sec_edgar",
        }

    except Exception as exc:
        log.error("edgar.search_filings.error", company=company_name, error=str(exc))
        return {"error": str(exc), "company_name": company_name}


def _get_10k(ticker: str) -> dict[str, Any]:
    """
    Fetch the most recent 10-K metadata for a company via the EDGAR submissions API.

    Strategy:
      1. Use the EDGAR company search to resolve ticker → CIK.
      2. Fetch the submissions JSON for that CIK.
      3. Find the most recent 10-K in the filings list.
      4. Return filing metadata + document index URL.
    """
    if not ticker:
        return {"error": "ticker is required"}

    try:
        # Step 1 — resolve ticker to CIK via EDGAR company tickers JSON.
        cik = _resolve_ticker_to_cik(ticker)
        if cik is None:
            return {
                "error": f"Could not resolve '{ticker}' to a CIK. Try the company's full legal name.",
                "ticker": ticker,
            }

        # Step 2 — fetch submissions for this CIK.
        sub_url = f"{_EDGAR_BASE}/submissions/CIK{cik:010d}.json"
        resp = requests.get(sub_url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        sub_data = resp.json()

        company_name = sub_data.get("name", ticker)
        filings = sub_data.get("filings", {}).get("recent", {})

        # Step 3 — find most recent 10-K.
        form_types: list[str] = filings.get("form", [])
        filing_dates: list[str] = filings.get("filingDate", [])
        accession_nos: list[str] = filings.get("accessionNumber", [])
        periods: list[str] = filings.get("reportDate", [])

        ten_k_idx: int | None = None
        for i, ft in enumerate(form_types):
            if ft == "10-K":
                ten_k_idx = i
                break  # Most recent is first in the list

        if ten_k_idx is None:
            return {
                "error": f"No 10-K filing found for {ticker} (CIK {cik}).",
                "ticker": ticker,
                "cik": cik,
                "company_name": company_name,
            }

        accession_no = accession_nos[ten_k_idx]
        accession_clean = accession_no.replace("-", "")
        doc_index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/"
            f"{accession_no}-index.htm"
        )

        log.info(
            "edgar.get_10k.ok",
            ticker=ticker,
            cik=cik,
            filed=filing_dates[ten_k_idx],
        )
        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "cik": cik,
            "form_type": "10-K",
            "filing_date": filing_dates[ten_k_idx],
            "period_of_report": periods[ten_k_idx],
            "accession_number": accession_no,
            "document_index_url": doc_index_url,
            "edgar_viewer_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=10",
            "note": (
                "Retrieve the full document from document_index_url. "
                "Key sections: Item 1 (Business), Item 1A (Risk Factors), "
                "Item 7 (MD&A), Item 8 (Financial Statements)."
            ),
            "source": "sec_edgar",
        }

    except Exception as exc:
        log.error("edgar.get_10k.error", ticker=ticker, error=str(exc))
        return {"error": str(exc), "ticker": ticker}


def _resolve_ticker_to_cik(ticker: str) -> int | None:
    """
    Resolve a stock ticker to a SEC CIK number using EDGAR's company_tickers.json.

    The JSON maps CIK → {ticker, title, cik_str}. We reverse it to look up by ticker.
    """
    try:
        resp = requests.get(
            f"{_EDGAR_BASE}/files/company_tickers.json",
            headers={"User-Agent": "AgentMesh/1.0 portfolio-project"},
            timeout=15,
        )
        resp.raise_for_status()
        data: dict[str, dict] = resp.json()

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return int(entry["cik_str"])

        return None
    except Exception as exc:
        log.error("edgar.resolve_ticker.error", ticker=ticker, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SEC EDGAR MCP Tool", version="1.0.0")


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
    return {"status": "ok", "service": "edgar-mcp-tool", "port": 8013}


if __name__ == "__main__":
    uvicorn.run(
        "edgar_tool:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8013)),
        reload=False,
    )
