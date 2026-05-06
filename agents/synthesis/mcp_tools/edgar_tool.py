"""
SEC EDGAR MCP tool server — port 8013.

Exposes two MCP tools:
  search_filings(company_name, form_type)  — full-text search EDGAR for filings
  get_10k(ticker)                          — fetch the most recent 10-K with key sections

Transport: MCP over SSE using a pure ASGI app (mcp>=1.24 compatible).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests
import structlog
import uvicorn
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# HTTP headers (no Host header — let requests set it per URL)
# ---------------------------------------------------------------------------

# SEC EDGAR policy: User-Agent must include a contact email address.
# Format: "{app-name}/{version} {contact-email}"
_SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "AgentMesh/1.0 agentmesh-portfolio@example.com",
)

_DATA_HEADERS = {
    "User-Agent": _SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}
_EFTS_HEADERS = {
    "User-Agent": _SEC_USER_AGENT,
    "Accept": "application/json",
}
_EDGAR_BASE = "https://data.sec.gov"
_EFTS_BASE  = "https://efts.sec.gov"

server = Server("edgar-tool")
# Disable DNS rebinding protection — safe for Docker internal networking where
# containers connect by hostname (e.g., edgar-tool:8013).
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
            name="search_filings",
            description=(
                "Search SEC EDGAR full-text search for filings by company name or keyword. "
                "Returns matching filings with company name, form type, filing date, and "
                "direct clickable URLs to each filing on SEC.gov."
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
                        "description": "SEC form type, e.g. '10-K', '10-Q', '8-K'.",
                        "default": "10-K",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Max results (1-10).",
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
                "Retrieve the most recent 10-K annual report for a company by ticker symbol. "
                "Returns filing metadata, direct SEC.gov document URL, and excerpts from "
                "key sections: Business Overview (Item 1) and Risk Factors (Item 1A)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, NVDA, MSFT).",
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
    log.info("edgar.tool.call", tool=name, args=args)

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
# HTTP helper — retry on 429, 1 s delay between calls
# ---------------------------------------------------------------------------


def _sec_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 3,
    timeout: int = 20,
) -> requests.Response:
    """GET with automatic retry on HTTP 429 and 1-second inter-call delay."""
    _hdrs = headers if headers is not None else _DATA_HEADERS
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=_hdrs, timeout=timeout)
            if resp.status_code == 429:
                wait = 2 * (attempt + 1)
                log.warning("edgar.rate_limited", url=url, attempt=attempt, wait=wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(1)  # polite 1-second delay between SEC API calls
            return resp
        except requests.exceptions.HTTPError:
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise exc
    raise RuntimeError(f"SEC API rate limit exceeded after {max_retries} retries for {url}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _search_filings(company_name: str, form_type: str, n_results: int) -> dict[str, Any]:
    """Search EDGAR full-text search index for filings matching company_name."""
    if not company_name:
        return {"error": "company_name is required"}
    try:
        resp = _sec_get(
            f"{_EFTS_BASE}/LATEST/search-index",
            params={
                "q":         f'"{company_name}"',
                "forms":     form_type,
                "dateRange": "custom",
                "startdt":   "2024-01-01",
                "enddt":     "2026-12-31",
                "_size":     n_results,
            },
            headers=_EFTS_HEADERS,
        )
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])[:n_results]

        results = []
        for hit in hits:
            src = hit.get("_source", {})

            # The EFTS document _id is "{accession_no}:{filename}" or just the
            # accession number.  Split on ":" and take only the accession part.
            hit_id  = hit.get("_id", "").strip()
            raw_acc = (
                hit_id.split(":")[0]          # strip ":filename" suffix from EFTS id
                or src.get("accession_no", "")
                or src.get("accession-no", "")
            ).strip()

            # CIK: check several field-name variants the EFTS API may use,
            # then derive from the first 10 digits of the accession number.
            raw_cik_str = (
                str(src.get("entity_id") or src.get("cik") or src.get("entity_cik") or "")
            ).strip()
            if not raw_cik_str and raw_acc:
                # Accession format: XXXXXXXXXX-YY-NNNNNN — first segment is CIK
                raw_cik_str = raw_acc.split("-")[0].lstrip("0") or raw_acc.split("-")[0]

            raw_cik   = raw_cik_str
            acc_clean = raw_acc.replace("-", "")

            # Build direct filing index URL
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{raw_cik}/{acc_clean}/{raw_acc}-index.htm"
                if raw_cik and raw_acc
                else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type={form_type}&dateb=&owner=include&count=10&search_text="
            )

            results.append({
                "company_name":     src.get("entity_name", company_name),
                "form_type":        src.get("form_type", form_type),
                "filing_date":      src.get("file_date"),
                "period_of_report": src.get("period_of_report"),
                "cik":              raw_cik,
                "accession_no":     raw_acc,
                "filing_url":       filing_url,
                "edgar_viewer_url": (
                    f"https://www.sec.gov/cgi-bin/browse-edgar"
                    f"?action=getcompany&CIK={raw_cik}&type={form_type}&dateb=&owner=include&count=10"
                    if raw_cik else None
                ),
            })

        log.info("edgar.search_filings.ok", company=company_name, count=len(results))
        return {
            "query":     company_name,
            "form_type": form_type,
            "total":     len(results),
            "results":   results,
            "source":    "sec_edgar",
        }

    except Exception as exc:
        log.error("edgar.search_filings.error", company=company_name, error=str(exc))
        return {"error": str(exc), "company_name": company_name, "source": "sec_edgar"}


def _get_10k(ticker: str) -> dict[str, Any]:
    """Fetch the most recent 10-K for a ticker including key section excerpts."""
    if not ticker:
        return {"error": "ticker is required"}
    ticker_up = ticker.upper()
    try:
        # ── Step 1: Resolve ticker → CIK ───────────────────────────────────────
        cik = _resolve_ticker_to_cik(ticker_up)
        if cik is None:
            return {"error": f"Could not resolve '{ticker_up}' to a CIK.", "ticker": ticker_up}

        cik_padded = f"{cik:010d}"

        # ── Step 2: Get submissions JSON for this company ───────────────────────
        sub_resp  = _sec_get(f"{_EDGAR_BASE}/submissions/CIK{cik_padded}.json", headers=_DATA_HEADERS)
        sub_data  = sub_resp.json()
        company_name = sub_data.get("name", ticker_up)
        filings   = sub_data.get("filings", {}).get("recent", {})

        form_types    = filings.get("form", [])
        filing_dates  = filings.get("filingDate", [])
        accession_nos = filings.get("accessionNumber", [])
        periods       = filings.get("reportDate", [])

        ten_k_idx = next((i for i, ft in enumerate(form_types) if ft == "10-K"), None)
        if ten_k_idx is None:
            return {"error": f"No 10-K found for {ticker_up}.", "ticker": ticker_up, "cik": cik}

        accession_no  = accession_nos[ten_k_idx]
        acc_clean     = accession_no.replace("-", "")
        filing_date   = filing_dates[ten_k_idx]
        period        = periods[ten_k_idx]

        # ── Step 3: Build direct URLs ───────────────────────────────────────────
        filing_index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{accession_no}-index.htm"
        )
        edgar_viewer_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik_padded}&type=10-K&dateb=&owner=include&count=10"
        )

        # ── Step 4: Fetch filing index to find primary document ─────────────────
        primary_doc_url = None
        sections = {}
        try:
            idx_resp = _sec_get(filing_index_url, headers=_DATA_HEADERS)
            # Look for the primary 10-K document link (usually an .htm file)
            primary_doc_url = _find_primary_document(
                idx_resp.text, cik, acc_clean, filing_index_url
            )
        except Exception as exc:
            log.warning("edgar.get_10k.index_fetch_failed", ticker=ticker_up, error=str(exc))

        # ── Step 5: Extract key sections from primary document ──────────────────
        if primary_doc_url:
            try:
                doc_resp = _sec_get(primary_doc_url, headers=_DATA_HEADERS)
                sections = _extract_10k_sections(doc_resp.text)
            except Exception as exc:
                log.warning("edgar.get_10k.doc_fetch_failed", ticker=ticker_up, error=str(exc))

        log.info("edgar.get_10k.ok", ticker=ticker_up, cik=cik, filed=filing_date)
        result: dict[str, Any] = {
            "ticker":           ticker_up,
            "company_name":     company_name,
            "cik":              cik,
            "form_type":        "10-K",
            "filing_date":      filing_date,
            "period_of_report": period,
            "accession_number": accession_no,
            "filing_index_url": filing_index_url,
            "edgar_viewer_url": edgar_viewer_url,
            "source":           "sec_edgar",
        }
        if primary_doc_url:
            result["primary_document_url"] = primary_doc_url
        if sections:
            result["key_sections"] = sections

        return result

    except Exception as exc:
        log.error("edgar.get_10k.error", ticker=ticker, error=str(exc))
        return {"error": str(exc), "ticker": ticker_up, "source": "sec_edgar"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_ticker_to_cik(ticker: str) -> int | None:
    """Resolve a stock ticker to its SEC CIK integer."""
    try:
        # company_tickers.json lives on www.sec.gov, not data.sec.gov
        resp = _sec_get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_DATA_HEADERS,
        )
        data: dict[str, dict] = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return int(entry["cik_str"])
        return None
    except Exception as exc:
        log.error("edgar.resolve_ticker.error", ticker=ticker, error=str(exc))
        return None


def _find_primary_document(
    index_html: str, cik: int, acc_clean: str, index_url: str
) -> str | None:
    """
    Parse the filing index HTML to find the primary 10-K document URL.

    Strategy (in order):
      1. Find a table row where the document type column contains "10-K"
         (the EDGAR index table has columns: Seq | Description | Document | Type | Size)
      2. Fallback: first .htm file that is not an exhibit and not an index page.
    """
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/"

    # ── Strategy 1: parse the EDGAR filing index table ──────────────────────
    # The table rows look like:
    #   <td>10-K</td> ... <a href="/Archives/edgar/data/.../nvda-20260125.htm">
    # We scan each <tr> and look for one whose type cell is "10-K".
    row_pattern  = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
    href_pattern = re.compile(r'href="(/Archives/edgar/data/[^"]+\.htm)"', re.IGNORECASE)

    for row_match in row_pattern.finditer(index_html):
        row_text = row_match.group(1)
        # Look for a cell that is exactly the form type "10-K"
        if re.search(r"<td[^>]*>\s*10-K\s*</td>", row_text, re.IGNORECASE):
            href = href_pattern.search(row_text)
            if href:
                path = href.group(1)
                if "index" not in path.lower():
                    return f"https://www.sec.gov{path}"

    # ── Strategy 2: first .htm in this accession dir that is not an exhibit ──
    # Exhibit filenames typically contain "ex" followed by a number (e.g. ex231, xex191).
    # The main 10-K usually matches the ticker + date pattern, e.g. nvda-20260125.htm.
    exhibit_re = re.compile(r"[_-]?ex\d|xex\d", re.IGNORECASE)
    htm_files  = re.findall(
        rf'/Archives/edgar/data/{cik}/{acc_clean}/([^"\'>\s]+\.htm)',
        index_html,
        re.IGNORECASE,
    )
    for fname in htm_files:
        fname_lower = fname.lower()
        if "index" in fname_lower:
            continue
        if exhibit_re.search(fname_lower):
            continue
        return f"{base}{fname}"

    return None


def _extract_10k_sections(html: str) -> dict[str, str]:
    """
    Extract Item 1 (Business) and Item 1A (Risk Factors) text from a 10-K HTML document.
    Each section is capped at 2000 characters.
    """
    MAX_CHARS = 2000

    # Strip HTML tags for text extraction
    clean = re.sub(r"<[^>]+>", " ", html)
    clean = re.sub(r"&nbsp;", " ", clean)
    clean = re.sub(r"&#\d+;", " ", clean)
    clean = re.sub(r"&amp;", "&", clean)
    clean = re.sub(r"&lt;", "<", clean)
    clean = re.sub(r"&gt;", ">", clean)
    clean = re.sub(r"\s{3,}", "  ", clean)

    sections: dict[str, str] = {}

    # Patterns to locate Item 1 and Item 1A
    item1_patterns = [
        r"item\s*1[\.\s]*(?:business|overview)",
        r"part\s*i.*?item\s*1[\.\s]*business",
    ]
    item1a_patterns = [
        r"item\s*1a[\.\s]*(?:risk\s*factors?)",
        r"item\s*1\s*a[\.\s]*risk",
    ]
    item2_patterns = [
        r"item\s*2[\.\s]*(?:properties?|unresolved)",
        r"item\s*1b",
    ]

    def _extract_between(text: str, start_patterns: list[str], end_patterns: list[str]) -> str | None:
        start = None
        for pat in start_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                start = m.end()
                break
        if start is None:
            return None
        end = len(text)
        for pat in end_patterns:
            m = re.search(pat, text[start:], re.IGNORECASE)
            if m:
                end = start + m.start()
                break
        excerpt = text[start:end].strip()
        # Clean up extra whitespace
        excerpt = re.sub(r"\s+", " ", excerpt)
        return excerpt[:MAX_CHARS] + ("…" if len(excerpt) > MAX_CHARS else "")

    item1_text = _extract_between(clean, item1_patterns, item1a_patterns + item2_patterns)
    if item1_text and len(item1_text) > 100:
        sections["business_overview_item1"] = item1_text

    item1a_text = _extract_between(clean, item1a_patterns, item2_patterns)
    if item1a_text and len(item1a_text) > 100:
        sections["risk_factors_item1a"] = item1a_text

    return sections


# ---------------------------------------------------------------------------
# Pure ASGI app — direct routing preserves scope["root_path"] = "" so that
# SseServerTransport.connect_sse() advertises "/messages/" (not "/sse/messages/")
# to the MCP client as the POST endpoint.
# DNS rebinding protection is disabled via TransportSecuritySettings above so
# Docker container hostnames (edgar-tool:8013) are accepted.
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
        response = JSONResponse({"status": "ok", "service": "edgar-mcp-tool", "port": 8013})
        await response(scope, receive, send)

    else:
        response = JSONResponse({"error": "not found"}, status_code=404)
        await response(scope, receive, send)


if __name__ == "__main__":
    uvicorn.run(
        "edgar_tool:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8013)),
        reload=False,
    )
