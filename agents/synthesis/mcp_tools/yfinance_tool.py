"""
YFinance MCP tool server — port 8011.

Exposes two MCP tools:
  get_stock_price(ticker)      — real-time price, volume, market cap, 52-week range
  get_financials(ticker)       — trailing P/E, revenue, earnings, margins, debt/equity

Transport: MCP over SSE (GET /sse + POST /messages).
The ADK synthesis agent connects via SseServerParams(url="http://yfinance-tool:8011/sse").
"""

from __future__ import annotations

import json
import os

import structlog
import uvicorn
import yfinance as yf
from fastapi import FastAPI, Request
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

server = Server("yfinance-tool")
transport = SseServerTransport("/messages/")


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
# Tool implementations
# ---------------------------------------------------------------------------


def _get_stock_price(ticker: str) -> dict:
    """Fetch real-time price data from yfinance."""
    if not ticker:
        return {"error": "ticker is required"}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_pct = (
            round(((price - prev_close) / prev_close) * 100, 2)
            if price and prev_close
            else None
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
    """Fetch key financial metrics from yfinance."""
    if not ticker:
        return {"error": "ticker is required"}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info

        # Revenue and earnings from income statement (TTM where available).
        revenue = info.get("totalRevenue")
        net_income = info.get("netIncomeToCommon")

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "revenue_ttm": revenue,
            "net_income_ttm": net_income,
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "free_cashflow": info.get("freeCashflow"),
            "beta": info.get("beta"),
            "source": "yfinance",
        }
    except Exception as exc:
        log.error("yfinance.get_financials.error", ticker=ticker, error=str(exc))
        return {"error": str(exc), "ticker": ticker}


# ---------------------------------------------------------------------------
# FastAPI app — MCP over SSE
# ---------------------------------------------------------------------------

app = FastAPI(title="YFinance MCP Tool", version="1.0.0")


@app.get("/sse")
async def sse_endpoint(request: Request) -> None:
    """SSE stream — the ADK agent connects here to use MCP tools."""
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
    """MCP message endpoint — used for client-to-server messages in SSE mode."""
    await transport.handle_post_message(
        request.scope, request.receive, request._send
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "yfinance-mcp-tool", "port": 8011}


if __name__ == "__main__":
    uvicorn.run(
        "yfinance_tool:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8011)),
        reload=False,
    )
