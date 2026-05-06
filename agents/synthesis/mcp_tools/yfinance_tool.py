"""
YFinance MCP tool server — port 8011.

Exposes two MCP tools:
  get_stock_price(ticker)      — real-time price, volume, market cap, 52-week range
  get_financials(ticker)       — trailing P/E, revenue, earnings, margins, debt/equity

Transport: MCP over SSE using a pure ASGI app (mcp>=1.24 compatible).
FastAPI is NOT used for the SSE/messages routes to avoid the ASGI double-response
RuntimeError that occurs when FastAPI wraps SSE connections.
"""

from __future__ import annotations

import json
import os

import structlog
import uvicorn
import yfinance as yf
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

server = Server("yfinance-tool")
# Disable DNS rebinding protection — safe for Docker internal networking where
# containers connect by hostname (e.g., yfinance-tool:8011).
transport = SseServerTransport(
    "/messages/",
    security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_stock_price",
            description=(
                "Get the current stock price and key market metrics for a ticker symbol. "
                "Returns price, daily change, volume, market cap, and 52-week range."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL, MSFT, NVDA.",
                    }
                },
                "required": ["ticker"],
            },
        ),
        types.Tool(
            name="get_financials",
            description=(
                "Get key financial metrics for a publicly traded company: "
                "trailing P/E ratio, revenue (TTM), net income, gross margin, "
                "operating margin, debt-to-equity, and return on equity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol.",
                    }
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
    log.info("yfinance.tool.call", tool=name, ticker=args.get("ticker"))

    if name == "get_stock_price":
        result = _get_stock_price(args.get("ticker", ""))
    elif name == "get_financials":
        result = _get_financials(args.get("ticker", ""))
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_number(value: int | float | None, prefix: str = "$") -> str | None:
    """Format large numbers as human-readable strings (e.g. $1.23B, $456.78M)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if abs(v) >= 1e12:
        return f"{prefix}{v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{prefix}{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{prefix}{v / 1e6:.2f}M"
    return f"{prefix}{v:,.0f}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _get_stock_price(ticker: str) -> dict:
    if not ticker:
        return {"error": "ticker is required"}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_pct = (
            round(((price - prev_close) / prev_close) * 100, 2)
            if price and prev_close else None
        )
        return {
            "ticker": ticker.upper(),
            "price": price,
            "currency": info.get("currency", "USD"),
            "change_percent": change_pct,
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "company_name": info.get("longName"),
            "exchange": info.get("exchange"),
            "source": "yfinance",
        }
    except Exception as exc:
        log.error("yfinance.get_stock_price.error", ticker=ticker, error=str(exc))
        return {"error": str(exc), "ticker": ticker}


def _get_financials(ticker: str) -> dict:
    if not ticker:
        return {"error": "ticker is required"}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        revenue      = info.get("totalRevenue")
        net_income   = info.get("netIncomeToCommon")
        free_cf      = info.get("freeCashflow")
        market_cap   = info.get("marketCap")
        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            # Raw values (for programmatic use)
            "revenue_ttm_raw": revenue,
            "net_income_ttm_raw": net_income,
            "free_cashflow_raw": free_cf,
            "market_cap_raw": market_cap,
            # Formatted strings (for LLM readability)
            "revenue_ttm": _fmt_number(revenue),
            "net_income_ttm": _fmt_number(net_income),
            "free_cashflow": _fmt_number(free_cf),
            "market_cap": _fmt_number(market_cap),
            "gross_margins": (
                f"{info.get('grossMargins') * 100:.1f}%"
                if info.get("grossMargins") is not None else None
            ),
            "operating_margins": (
                f"{info.get('operatingMargins') * 100:.1f}%"
                if info.get("operatingMargins") is not None else None
            ),
            "profit_margins": (
                f"{info.get('profitMargins') * 100:.1f}%"
                if info.get("profitMargins") is not None else None
            ),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": (
                f"{info.get('returnOnEquity') * 100:.1f}%"
                if info.get("returnOnEquity") is not None else None
            ),
            "return_on_assets": (
                f"{info.get('returnOnAssets') * 100:.1f}%"
                if info.get("returnOnAssets") is not None else None
            ),
            "beta": info.get("beta"),
            "source": "yfinance",
        }
    except Exception as exc:
        log.error("yfinance.get_financials.error", ticker=ticker, error=str(exc))
        return {"error": str(exc), "ticker": ticker}


# ---------------------------------------------------------------------------
# Pure ASGI app — direct routing preserves scope["root_path"] = "" so that
# SseServerTransport.connect_sse() advertises "/messages/" (not "/sse/messages/")
# to the MCP client as the POST endpoint.
# DNS rebinding protection is disabled via TransportSecuritySettings above so
# Docker container hostnames (yfinance-tool:8011) are accepted.
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
        response = JSONResponse({"status": "ok", "service": "yfinance-mcp-tool", "port": 8011})
        await response(scope, receive, send)

    else:
        response = JSONResponse({"error": "not found"}, status_code=404)
        await response(scope, receive, send)


if __name__ == "__main__":
    uvicorn.run(
        "yfinance_tool:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8011)),
        reload=False,
    )
