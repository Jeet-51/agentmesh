"""
Google ADK synthesis agent for AgentMesh.

Receives ResearchFindings from the orchestrator, enriches them with:
  - Live stock/financial data via yfinance MCP tool (SSE)
  - Company background via Wikipedia REST API (direct HTTP)
  - SEC filings via EDGAR EFTS API (direct HTTP)
  - Analyst ratings & news via Finnhub API (direct HTTP)

Architecture
------------
SynthesisAgent wraps an ADK LlmAgent. Wikipedia and EDGAR are called directly
as async HTTP functions before synthesis — their results are injected into the
prompt context. Only yfinance remains as an MCP tool (it works reliably).

Graceful degradation
--------------------
Every enrichment call is wrapped in try/except — failures return placeholder
strings. The ADK runner fallback path still produces a report from research
findings alone if Gemini errors.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types as genai_types

from shared.models import (
    Citation,
    Framework,
    ResearchFindings,
    SynthesisReport,
    _new_uuid,
)

log = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# MCP tool server URLs (yfinance only — wikipedia/edgar are now direct HTTP)
# ---------------------------------------------------------------------------

_YFINANCE_URL = os.environ.get("YFINANCE_TOOL_URL", "http://yfinance-tool:8011/sse")

# ---------------------------------------------------------------------------
# Known company → ticker mapping (used for Finnhub calls)
# ---------------------------------------------------------------------------

_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA",
    "tesla": "TSLA", "amazon": "AMZN", "google": "GOOGL",
    "alphabet": "GOOGL", "meta": "META", "netflix": "NFLX",
    "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
    "tsmc": "TSM", "ibm": "IBM", "oracle": "ORCL",
    "salesforce": "CRM", "adobe": "ADBE", "paypal": "PYPL",
    "visa": "V", "mastercard": "MA", "jpmorgan": "JPM",
    "goldman sachs": "GS", "goldman": "GS",
    "berkshire": "BRK-B", "walmart": "WMT", "target": "TGT",
    "costco": "COST", "exxon": "XOM", "chevron": "CVX",
    "johnson & johnson": "JNJ", "johnson": "JNJ",
    "pfizer": "PFE", "moderna": "MRNA", "unitedhealth": "UNH",
    "broadcom": "AVGO", "arm": "ARM", "palantir": "PLTR",
    "snowflake": "SNOW", "cloudflare": "NET", "datadog": "DDOG",
    "spotify": "SPOT", "uber": "UBER", "airbnb": "ABNB",
}

# ---------------------------------------------------------------------------
# Direct async enrichment functions
# ---------------------------------------------------------------------------


async def get_wikipedia_summary(topic: str) -> dict[str, str]:
    """
    Fetch a plain-text summary from the Wikipedia REST API.

    Returns a dict with keys:
      text  — the extract (up to 2000 chars)
      title — "Wikipedia: <resolved topic>" for use as citation title
      url   — canonical Wikipedia article URL for use as citation URL
    """
    def _make_result(resolved: str, text: str) -> dict[str, str]:
        slug = resolved.replace(" ", "_")
        return {
            "text": text,
            "title": f"Wikipedia: {resolved}",
            "url": f"https://en.wikipedia.org/wiki/{slug}",
        }

    try:
        slug = topic.strip().replace(" ", "_")
        base = "https://en.wikipedia.org/api/rest_v1/page/summary"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{base}/{slug}", headers={"Accept": "application/json"})
            resolved = topic.strip()
            if r.status_code == 404:
                # Retry with first two words only
                short = " ".join(topic.strip().split()[:2])
                slug2 = short.replace(" ", "_")
                r = await client.get(f"{base}/{slug2}", headers={"Accept": "application/json"})
                resolved = short
            data = r.json()
            # Use the canonical title returned by Wikipedia when available
            resolved = data.get("title") or resolved
            text = data.get("extract", "")[:2000]
            return _make_result(resolved, text)
    except Exception as e:
        log.warning("enrichment.wikipedia.error", topic=topic, error=str(e))
        return {
            "text": f"Wikipedia unavailable: {e}",
            "title": f"Wikipedia: {topic}",
            "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
        }


async def search_edgar_filings(company: str) -> str:
    """Search SEC EDGAR EFTS for recent 10-K filings for a company."""
    try:
        headers = {"User-Agent": "AgentMesh/1.0 research@agentmesh.ai"}
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q={company.replace(' ', '+')}&forms=10-K"
            f"&dateRange=custom&startdt=2023-01-01&enddt=2026-12-31"
        )
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            hits = r.json().get("hits", {}).get("hits", [])[:3]
        results = []
        for h in hits:
            src = h.get("_source", {})
            names = src.get("display_names") or src.get("entity_name") or ["Unknown"]
            name = names[0] if isinstance(names, list) else names
            file_date = src.get("file_date", "")
            period = src.get("period_of_report", "")
            form = src.get("form_type", "10-K")
            # Build direct filing URL from accession number in _id
            hit_id = h.get("_id", "")
            accession = hit_id.split(":")[0] if ":" in hit_id else hit_id
            results.append(
                f"{name} | Form: {form} | Filed: {file_date} | Period: {period}"
                + (f" | Accession: {accession}" if accession else "")
            )
        return "\n".join(results) if results else "No 10-K filings found in date range."
    except Exception as e:
        log.warning("enrichment.edgar.error", company=company, error=str(e))
        return f"EDGAR unavailable: {e}"


async def get_finnhub_data(ticker: str) -> dict[str, Any]:
    """
    Fetch analyst consensus and recent news from Finnhub.

    Returns a dict with:
      text     — formatted text for prompt injection
      articles — list of {title, url, published_at, source} for dated citations
    """
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not key:
        return {"text": "Finnhub key not configured", "articles": []}
    try:
        today = _utcnow().strftime("%Y-%m-%d")
        async with httpx.AsyncClient(timeout=10) as client:
            rec_r, news_r = await asyncio.gather(
                client.get(
                    f"https://finnhub.io/api/v1/stock/recommendation"
                    f"?symbol={ticker}&token={key}"
                ),
                client.get(
                    f"https://finnhub.io/api/v1/company-news"
                    f"?symbol={ticker}&from=2026-01-01&to={today}&token={key}"
                ),
                return_exceptions=True,
            )

        result_parts: list[str] = []
        article_meta: list[dict[str, str]] = []

        if not isinstance(rec_r, Exception) and rec_r.status_code == 200:
            recs = rec_r.json()
            if recs:
                r = recs[0]
                result_parts.append(
                    f"Analyst consensus ({ticker}): {r.get('signal', 'N/A')} "
                    f"(Strong Buy: {r.get('strongBuy', 0)}, Buy: {r.get('buy', 0)}, "
                    f"Hold: {r.get('hold', 0)}, Sell: {r.get('sell', 0)}, "
                    f"Strong Sell: {r.get('strongSell', 0)})"
                )

        if not isinstance(news_r, Exception) and news_r.status_code == 200:
            raw_articles = news_r.json()[:5]
            if raw_articles:
                result_parts.append(f"Recent news ({ticker}):")
                for a in raw_articles:
                    headline = (a.get("headline") or "").strip()
                    source   = (a.get("source") or "").strip()
                    url      = (a.get("url") or "").strip()
                    ts       = a.get("datetime")   # Unix timestamp int

                    if not headline:
                        continue
                    result_parts.append(f"  - [{source}] {headline}")

                    # Convert Unix timestamp → ISO-8601 for published_at
                    published_at = ""
                    if ts:
                        try:
                            from datetime import timezone as _tz
                            published_at = datetime.fromtimestamp(
                                int(ts), tz=_tz.utc
                            ).strftime("%Y-%m-%dT%H:%M:%SZ")
                        except Exception:
                            pass

                    if url:
                        article_meta.append({
                            "title":        f"{headline} — {source}" if source else headline,
                            "url":          url,
                            "published_at": published_at,
                        })

        text = "\n".join(result_parts) if result_parts else f"No Finnhub data for {ticker}"
        log.info("enrichment.finnhub.done", ticker=ticker, articles=len(article_meta))
        return {"text": text, "articles": article_meta}

    except Exception as e:
        log.warning("enrichment.finnhub.error", ticker=ticker, error=str(e))
        return {"text": f"Finnhub unavailable: {e}", "articles": []}


async def get_news_articles(topic: str) -> dict[str, Any]:
    """
    Fetch recent news articles from NewsAPI for a given topic.

    Returns a dict with:
      text        — formatted bullet list for prompt injection
      articles    — list of {title, url} for per-article citations
      search_url  — the NewsAPI search URL (used as fallback citation link)
    """
    key = os.getenv("NEWS_API_KEY", "").strip()
    search_url = f"https://newsapi.org/v2/everything?q={topic.replace(' ', '+')}&sortBy=relevancy"
    if not key:
        return {"text": "NewsAPI key not configured", "articles": [], "search_url": search_url}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": topic,
                    "sortBy": "relevancy",
                    "pageSize": 10,   # fetch more so we have enough after filtering
                    "language": "en",
                    "apiKey": key,
                },
            )
        data = r.json()
        if r.status_code != 200:
            log.warning("enrichment.newsapi.bad_status", status=r.status_code, body=data)
            return {"text": f"NewsAPI error {r.status_code}", "articles": [], "search_url": search_url}

        _REMOVED = {"[removed]", "removed", ""}
        lines: list[str] = []
        meta: list[dict[str, str]] = []
        for a in data.get("articles", []):
            title  = (a.get("title") or "").strip()
            source = (a.get("source") or {}).get("name", "") or ""
            url    = (a.get("url") or "").strip()
            # Skip placeholder / removed articles
            if not url or title.lower() in _REMOVED or not title:
                continue
            if "removed" in url.lower():
                continue
            lines.append(f"- {title} ({source}) [{url}]")
            meta.append({
                "title":        f"{title} — {source}" if source else title,
                "url":          url,
                "published_at": (a.get("publishedAt") or "").strip(),
            })
            if len(meta) >= 5:
                break

        text = "\n".join(lines) if lines else "No usable news articles found"
        log.info("enrichment.newsapi.done", topic=topic, total=len(data.get("articles", [])), usable=len(meta))
        return {"text": text, "articles": meta, "search_url": search_url}
    except Exception as e:
        log.warning("enrichment.newsapi.error", topic=topic, error=str(e))
        return {"text": f"NewsAPI unavailable: {e}", "articles": [], "search_url": search_url}


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------


def _extract_entities(query: str, findings: list[ResearchFindings]) -> list[tuple[str, str]]:
    """
    Extract (display_name, ticker) pairs from the query and findings text.

    Returns up to 3 entries. Falls back to scanning for ALLCAPS ticker patterns
    if no known company names are found.
    """
    query_lower = query.lower()
    results: list[tuple[str, str]] = []
    seen_tickers: set[str] = set()

    # 1. Check known companies against the query
    for company, ticker in _TICKER_MAP.items():
        if company in query_lower and ticker not in seen_tickers:
            results.append((company.title(), ticker))
            seen_tickers.add(ticker)
        if len(results) >= 3:
            break

    # 2. Scan findings text for $TICKER or plain ALLCAPS patterns
    if len(results) < 3:
        all_text = " ".join(f.findings[:300] for f in findings[:4])
        raw_tickers = re.findall(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b", all_text)
        known_set = set(_TICKER_MAP.values())
        for groups in raw_tickers:
            t = groups[0] or groups[1]
            if t in known_set and t not in seen_tickers:
                for cname, cticker in _TICKER_MAP.items():
                    if cticker == t:
                        results.append((cname.title(), t))
                        seen_tickers.add(t)
                        break
            if len(results) >= 3:
                break

    return results


def _extract_company_names(query: str) -> list[str]:
    """
    Return known company names found in the query (up to 2).

    Returns an empty list when no recognised company is found — callers
    that would produce meaningless results with a generic query (e.g. EDGAR)
    should skip their call entirely in that case.
    """
    query_lower = query.lower()
    names: list[str] = []
    for company in _TICKER_MAP:
        if company in query_lower:
            names.append(company.title())
        if len(names) >= 2:
            break
    return names


# ---------------------------------------------------------------------------
# UPV extraction — User-Provided Values
# ---------------------------------------------------------------------------


def _extract_upvs(query: str) -> list[dict[str, str]]:
    """
    Extract User-Provided Values (UPVs) from the query string.

    Detects explicit numbers the user embedded in their query — these are
    treated as ground truth during synthesis and must not be overwritten by
    retrieved data.

    Handles patterns like:
      "revenue ($111.2B)"        → {metric: "revenue",      value: "$111.2B"}
      "gross margin of 47.8%"    → {metric: "gross margin", value: "47.8%"}
      "18.2% net margin"         → {metric: "net margin",   value: "18.2%"}
      "market cap: $2.93T"       → {metric: "market cap",   value: "$2.93T"}
    """
    upvs: list[dict[str, str]] = []
    seen_values: set[str] = set()

    _STRIP_WORDS = {
        "what", "how", "why", "the", "and", "or", "in", "of", "for",
        "with", "from", "that", "this", "will", "have", "its", "their",
        "also", "both", "each", "any", "all", "a", "an",
    }

    def _clean_metric(raw: str) -> str:
        """Strip punctuation and leading/trailing connector words."""
        m = raw.strip().strip("(").strip(":").strip(",").strip()
        # Drop leading connector words  ("and gross margin" → "gross margin")
        words = m.split()
        while words and words[0].lower() in _STRIP_WORDS:
            words = words[1:]
        # Drop trailing connector words
        while words and words[-1].lower() in _STRIP_WORDS:
            words = words[:-1]
        return " ".join(words)

    def _valid_metric(m: str) -> bool:
        return (
            len(m) >= 3
            and not m[0].isdigit()
            and m.lower().split()[0] not in _STRIP_WORDS
        )

    # Pattern 1: metric (value)  →  "revenue ($111.2B)" or "gross margin (47.8%)"
    # [BTMKbtmk%] is non-optional here so pure digits like (2024) are excluded
    for metric, value in re.findall(
        r"([\w][\w\s]{1,30}?)\s*\(\s*(\$?[\d,.]+\s*[BTMKbtmk%])\s*\)",
        query,
    ):
        metric = _clean_metric(metric)
        value = value.strip()
        if _valid_metric(metric) and value not in seen_values:
            upvs.append({"metric": metric, "value": value})
            seen_values.add(value)

    # Pattern 2: "metric: value"  →  "operating margin: 8.2%"
    for metric, value in re.findall(
        r"([\w][\w\s]{1,28}?)\s*:\s*(\$?[\d,.]+\s*[BTMKbtmk%])",
        query,
    ):
        metric = _clean_metric(metric)
        value = value.strip()
        if _valid_metric(metric) and value not in seen_values:
            upvs.append({"metric": metric, "value": value})
            seen_values.add(value)

    # Pattern 3: "value metric"  →  "18.2% net margin"
    for value, metric in re.findall(
        r"(\$?[\d,.]+\s*[BTMKbtmk%])\s+((?:[\w]+\s*){1,4})",
        query,
    ):
        metric = _clean_metric(metric)
        value = value.strip()
        if _valid_metric(metric) and value not in seen_values:
            upvs.append({"metric": metric, "value": value})
            seen_values.add(value)

    return upvs[:12]  # cap to avoid noise on very long queries


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a senior research analyst. You have been given:
  1. Pre-fetched enrichment data (Wikipedia background, SEC EDGAR filings, Finnhub analyst ratings)
  2. Research findings from a multi-agent crew
  3. Access to live yfinance MCP tools for real-time stock/financial data

Your task: produce a professional intelligence report using ALL available data.

IMPORTANT — USE THE PRE-FETCHED DATA:
- Cite Wikipedia background as [wikipedia] where you reference company history/overview
- Cite EDGAR filing info as [edgar] when discussing official financial filings
- Cite Finnhub data as [yfinance] when referencing analyst ratings or news
- Cite NewsAPI articles as [newsapi] when referencing recent news headlines
- Call yfinance MCP tools (get_stock_price, get_financials) for live prices if needed
- Cite research findings as [research]

REPORT STRUCTURE (use these exact section headers):

## Executive Summary
2-3 sentences capturing the single most important takeaway.

## Market Position & Outlook
Detailed analysis synthesising all research findings into coherent narrative.
Do not repeat findings verbatim — draw conclusions, identify trends, connect dots.

## Key Risks & Challenges
Bullet points of the most material risks, each with 1-2 sentences of context.

## Financial Indicators
Quantitative data from yfinance, EDGAR filings, and research. Include revenue,
growth rates, market cap, margins, analyst consensus ratings where available.

## Conclusion & Recommendation
Clear forward-looking statement and 2-3 actionable recommendations.

DATA HIERARCHY — follow strictly in this order:
  1. USER-PROVIDED VALUES (UPVs) — absolute ground truth, from the query itself
  2. SEC EDGAR official filings — verified regulatory disclosures
  3. yfinance live data — real-time market figures
  4. Finnhub / NewsAPI — analyst ratings and recent news
  5. CrewAI research findings — web-sourced secondary research
  6. Wikipedia — background / encyclopaedic context only

USER-PROVIDED VALUE (UPV) RULES — non-negotiable:
- If the prompt includes a "## USER-PROVIDED VALUES" section, those figures are
  the primary values for the Executive Summary, Financial Indicators, and all
  derivative analysis. Use them exactly as written — never round, adjust, or omit.
- VARIABLE BINDING: Before generating the report, mentally map each UPV to its
  role: query.revenue → report.primary_revenue, query.margin → report.primary_margin.
  Do not let retrieved data overwrite these bindings during generation.
- CONFLICT HANDLING: If any retrieved source (any tier) contradicts a UPV by more
  than 2%, you MUST NOT list the external figure as a factual alternative in the main
  body. Instead, add a final section "## Contradictory Market Data" formatted as:
    > [Source name] reported [external value] for [metric], which differs from the
    > user-provided figure of [UPV]. The user-provided value is used as primary.
  This keeps the main report internally consistent while preserving transparency.
- If no UPV section is present, fall back to the normal source hierarchy above.

WRITING RULES:
- Write in flowing professional prose (except Key Risks — use bullets)
- Synthesise across all sub-tasks — don't repeat each finding in sequence
- Every specific factual claim must note its source: [research], [yfinance], [wikipedia], [edgar]
- Confidence score reflects research quality, not tool availability
- EXACT NUMBERS: In the Financial Indicators section, copy the precise values returned by
  yfinance tools directly into the narrative — never round or paraphrase. Write "$189.47"
  not "~$190" or "approximately $190". If yfinance returns Revenue TTM: $391.04B, write
  exactly $391.04B in the narrative.
- DIRECT ANSWER: The ## Conclusion & Recommendation section MUST open with a sentence that
  directly answers the specific question asked. For comparison queries (compare X vs Y),
  state which company leads and by what margin in the very first sentence using exact figures
  and the company names from the query. Never open with a generic statement.
- PRIORITY SOURCES: When the same fact appears in multiple sources, always cite the most
  authoritative one: SEC EDGAR or government filings first, Reuters/Bloomberg second,
  Yahoo Finance/CNBC third, Wikipedia last. Never cite an unknown blog when a major
  source confirms the same claim.
- HEDGE UNCERTAINTY: Any forward-looking projection, analyst estimate, or unverified claim
  must use hedging language: "analysts expect", "according to [source]", "reportedly",
  "guidance suggests", "may", "could". Never state uncertain or future events as fact.

OUTPUT FORMAT — after the report, output ONE JSON block in ```json ... ```:

```json
{
  "narrative": "<the complete report in markdown, all five sections>",
  "confidence_scores": {
    "overall": <float 0.0-1.0>,
    "research_quality": <float 0.0-1.0>
  },
  "recommended_actions": [
    "<specific actionable recommendation 1>",
    "<specific actionable recommendation 2>",
    "<specific actionable recommendation 3>"
  ],
  "citations": [
    {
      "source_title": "<source name or article title>",
      "source_url": "<full https:// URL — copy from the source lists; never leave empty>",
      "claim": "<specific claim this citation supports>",
      "tool_used": "<yfinance|wikipedia|edgar|crewai_research|newsapi>"
    }
  ]
}
```

CRITICAL — YFINANCE CITATION CLAIM FORMAT:
For every get_stock_price or get_financials MCP tool call, the citation "claim"
field MUST include the actual numerical values returned by the tool. Example:
  "claim": "Stock price: $189.50, Market cap: $2.93T, Revenue TTM: $391.04B, P/E: 28.5x"
NEVER write vague claims like "live data retrieved" or "financial data from yfinance".
Always include the real numbers — this is required for data verification.

confidence_score guidance:
  0.85-1.0  Strong multi-source consensus, specific figures verified across 3+ sources.
  0.70-0.85 Good evidence, claims from multiple independent sources.
  0.55-0.70 Moderate: findings from 1-2 sources, plausible but less verified.
  0.0-0.55  Weak, contradictory, or very thin evidence.
"""

# ---------------------------------------------------------------------------
# SynthesisAgent
# ---------------------------------------------------------------------------


class SynthesisAgent:
    """
    Manages an ADK LlmAgent with yfinance MCP + direct HTTP enrichment.

    Lifecycle:
      agent = SynthesisAgent()
      await agent.initialize()          # call once at startup
      report = await agent.synthesize(...)
      await agent.close()               # call on shutdown
    """

    def __init__(self) -> None:
        self._session_service: InMemorySessionService | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Set up the shared session service.
        MCP connections are created fresh per synthesis call to avoid
        stale SSE connections that cause tool failures.
        """
        log.info("synthesis_agent.initialize.start")
        self._session_service = InMemorySessionService()
        log.info(
            "synthesis_agent.initialize.done",
            direct_tools=["wikipedia", "edgar", "finnhub", "newsapi"],
            mcp_tools="yfinance (connected fresh per-run)",
        )

    async def close(self) -> None:
        """Nothing persistent to clean up — MCP connections are per-run."""
        log.info("synthesis_agent.closed")

    async def _build_runner(self) -> tuple[Runner, MCPToolset | None]:
        """
        Create a fresh ADK Runner with a fresh yfinance MCP connection.
        Returns (runner, toolset) — caller must close the toolset after use.
        Called once per synthesize() invocation to avoid stale SSE connections.
        """
        tools: list[Any] = []
        toolset: MCPToolset | None = None
        connected: list[str] = []

        try:
            toolset = MCPToolset(
                connection_params=SseConnectionParams(url=_YFINANCE_URL),
            )
            mcp_tools = await asyncio.wait_for(toolset.get_tools(), timeout=12.0)
            tools.extend(mcp_tools)
            connected.append("yfinance")
            log.info("synthesis_agent.mcp.connected", tool_server="yfinance", url=_YFINANCE_URL)
        except Exception as exc:
            log.warning(
                "synthesis_agent.mcp.connection_failed",
                tool_server="yfinance",
                url=_YFINANCE_URL,
                error=str(exc),
            )
            toolset = None

        log.info(
            "synthesis_agent.mcp.ready",
            connected_tools=connected,
            mcp_tool_count=len(tools),
        )

        adk_agent = LlmAgent(
            model="gemini-2.5-flash",
            name="synthesis_agent",
            description="Synthesises research findings into a cited intelligence report.",
            instruction=_SYNTHESIS_PROMPT,
            tools=tools,
        )

        runner = Runner(
            agent=adk_agent,
            app_name="agentmesh-synthesis",
            session_service=self._session_service,
        )
        return runner, toolset

    # ------------------------------------------------------------------
    # Enrichment (called before synthesis)
    # ------------------------------------------------------------------

    async def _fetch_enrichment(
        self,
        query: str,
        findings: list[ResearchFindings],
    ) -> dict[str, Any]:
        """
        Call Wikipedia, EDGAR, Finnhub, and NewsAPI in parallel before synthesis.

        Returns a dict with keys:
          wikipedia        — plain text for prompt injection
          wikipedia_meta   — list of {title, url} dicts for citation building
          edgar            — plain text for prompt injection (only when companies found)
          finnhub          — plain text for prompt injection
          newsapi          — plain text for prompt injection
          newsapi_meta     — list of {title, url} dicts for per-article citations
        """
        company_names = _extract_company_names(query)
        entities = _extract_entities(query, findings)  # [(display_name, ticker), ...]

        # If no specific companies found, also try extracting from findings
        if not company_names:
            company_names = [name for name, _ in entities[:2]]

        log.info(
            "synthesis_agent.enrichment.start",
            companies=company_names,
            tickers=[t for _, t in entities],
            skip_edgar=not company_names,
        )

        # Wikipedia — attempt if we have company names
        wiki_coros = [get_wikipedia_summary(name) for name in company_names[:2]]

        # EDGAR — skip entirely when no recognised company found (avoids unrelated filings)
        edgar_coros = (
            [search_edgar_filings(name) for name in company_names[:2]]
            if company_names else []
        )

        # Finnhub — requires a ticker
        finnhub_coros = [get_finnhub_data(ticker) for _, ticker in entities[:2]]

        # NewsAPI — use the raw query (works for both company and topic queries)
        news_topic = company_names[0] if company_names else query[:100]
        news_coros = [get_news_articles(news_topic)]

        all_coros = wiki_coros + edgar_coros + finnhub_coros + news_coros
        results = await asyncio.gather(*all_coros, return_exceptions=True)

        n_wiki    = len(wiki_coros)
        n_edgar   = len(edgar_coros)
        n_finnhub = len(finnhub_coros)
        wiki_raw    = results[:n_wiki]
        edgar_raw   = results[n_wiki:n_wiki + n_edgar]
        finnhub_raw = results[n_wiki + n_edgar:n_wiki + n_edgar + n_finnhub]
        news_raw    = results[n_wiki + n_edgar + n_finnhub:]

        enrichment: dict[str, Any] = {}

        # ── Wikipedia: unpack dict results, build text + metadata ──────
        wiki_texts: list[str] = []
        wiki_meta: list[dict[str, str]] = []
        for r in wiki_raw:
            if isinstance(r, Exception):
                continue
            if isinstance(r, dict):
                text = r.get("text", "")
                if text:
                    wiki_texts.append(text)
                    wiki_meta.append({"title": r["title"], "url": r["url"]})
            elif r:
                wiki_texts.append(str(r))

        if wiki_texts:
            enrichment["wikipedia"]      = "\n\n---\n\n".join(wiki_texts)
            enrichment["wikipedia_meta"] = wiki_meta

        # ── EDGAR: plain text ──────────────────────────────────────────
        edgar_texts = [
            str(r) for r in edgar_raw
            if not isinstance(r, Exception) and r and "No 10-K" not in str(r)
        ]
        if edgar_texts:
            enrichment["edgar"] = "\n\n".join(edgar_texts)

        # ── Finnhub: unpack dict, build text + dated article metadata ─────
        finnhub_texts: list[str] = []
        finnhub_meta: list[dict[str, str]] = []
        for r in finnhub_raw:
            if isinstance(r, Exception):
                continue
            if isinstance(r, dict):
                text = r.get("text", "")
                if text and "not configured" not in text and "unavailable" not in text:
                    finnhub_texts.append(text)
                finnhub_meta.extend(r.get("articles", []))
            elif r and "not configured" not in str(r):
                finnhub_texts.append(str(r))
        if finnhub_texts:
            enrichment["finnhub"]      = "\n\n".join(finnhub_texts)
            enrichment["finnhub_meta"] = finnhub_meta

        # ── NewsAPI: unpack dict results, build text + per-article metadata ──
        news_meta: list[dict[str, str]] = []
        news_texts: list[str] = []
        news_search_url: str = "https://newsapi.org"
        for r in news_raw:
            if isinstance(r, Exception):
                continue
            if isinstance(r, dict):
                text = r.get("text", "")
                articles = r.get("articles", [])
                if text and "not configured" not in text and "unavailable" not in text:
                    news_texts.append(text)
                news_meta.extend(articles)
                if r.get("search_url"):
                    news_search_url = r["search_url"]
            elif r:
                news_texts.append(str(r))

        if news_texts:
            enrichment["newsapi"]            = "\n\n".join(news_texts)
            enrichment["newsapi_meta"]       = news_meta
            enrichment["newsapi_search_url"] = news_search_url

        log.info(
            "synthesis_agent.enrichment.done",
            wiki_chars=len(enrichment.get("wikipedia", "")),
            wiki_sources=len(wiki_meta),
            edgar_chars=len(enrichment.get("edgar", "")),
            edgar_skipped=not company_names,
            finnhub_chars=len(enrichment.get("finnhub", "")),
            news_articles=len(news_meta),
        )

        return enrichment

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        query: str,
        findings: list[ResearchFindings],
        run_id: str,
        trace_id: str,
    ) -> SynthesisReport:
        """
        1. Fetch Wikipedia, EDGAR, Finnhub enrichment in parallel.
        2. Run the ADK agent over findings + enrichment context.
        3. Parse JSON response into SynthesisReport.
        """
        bound_log = log.bind(trace_id=trace_id, run_id=run_id)
        bound_log.info(
            "synthesis_agent.synthesize.start",
            finding_count=len(findings),
        )

        if self._session_service is None:
            raise RuntimeError("SynthesisAgent.initialize() must be called before synthesize().")

        # ── Step 1: Fetch enrichment data ───────────────────────────────
        enrichment = await self._fetch_enrichment(query, findings)

        # ── Step 2: Build prompt with enrichment injected ───────────────
        user_prompt = self._build_prompt(query, findings, enrichment)

        # ── Step 3: Build fresh runner + MCP connection for this run ────
        runner, toolset = await self._build_runner()

        session = await self._session_service.create_session(
            app_name="agentmesh-synthesis",
            user_id=run_id,
        )

        tool_calls_made: list[dict[str, Any]] = []
        final_text: str = ""

        try:
            async for event in runner.run_async(
                user_id=run_id,
                session_id=session.id,
                new_message=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_prompt)],
                ),
            ):
                fn_calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
                for fn_call in fn_calls:
                    tool_calls_made.append(
                        {"tool_name": fn_call.name, "args": dict(fn_call.args) if fn_call.args else {}}
                    )
                    bound_log.debug("synthesis_agent.tool_call", tool=fn_call.name, args=fn_call.args)

                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_text = "".join(
                            p.text for p in event.content.parts if hasattr(p, "text") and p.text
                        )

        except Exception as exc:
            bound_log.error("synthesis_agent.runner.error", error=str(exc))
            return self._fallback_report(query, findings, run_id, str(exc))

        bound_log.info(
            "synthesis_agent.run.complete",
            tool_calls=len(tool_calls_made),
            response_chars=len(final_text),
        )

        # ── Step 4: Close the per-run MCP connection cleanly ────────────
        if toolset is not None:
            try:
                if hasattr(toolset, "close"):
                    await toolset.close()
            except Exception:
                pass  # SSE cleanup errors are cosmetic — ignore them

        report = self._parse_response(
            text=final_text,
            findings=findings,
            run_id=run_id,
            tool_calls=tool_calls_made,
            enrichment=enrichment,
            bound_log=bound_log,
        )

        bound_log.info(
            "synthesis_agent.synthesize.done",
            report_id=report.report_id,
            citations=len(report.citations),
            mcp_tools_called=report.mcp_tools_called,
        )
        return report

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        findings: list[ResearchFindings],
        enrichment: dict[str, str],
    ) -> str:
        """Render the user message with enrichment context + findings inlined."""
        sections: list[str] = [f"## Original Research Query\n{query}\n"]

        # ── User-Provided Values (UPVs) — inject as locked ground truth ─
        upvs = _extract_upvs(query)
        if upvs:
            upv_lines = "\n".join(
                f"  • {u['metric'].strip().title()}: {u['value']}" for u in upvs
            )
            sections.append(
                "## USER-PROVIDED VALUES — TREAT AS ABSOLUTE GROUND TRUTH\n"
                "The user explicitly stated these figures in their query.\n"
                "They OVERRIDE any contradicting data from external sources:\n\n"
                f"{upv_lines}\n\n"
                "BINDING RULES (re-stated for emphasis):\n"
                "  1. Use these exact values in the Executive Summary and Financial"
                " Indicators — never substitute or average them with external data.\n"
                "  2. If any source contradicts a UPV by >2%, do NOT list the external"
                " value as an alternative in the main body. Instead add a\n"
                "     '## Contradictory Market Data' footnote at the end.\n"
                "  3. Build all derivative analysis (e.g. margin calculations,"
                " YoY growth) from these values as the starting point."
            )

        # ── Pre-fetched enrichment data ─────────────────────────────────
        if enrichment.get("wikipedia"):
            sections.append(
                "## PRE-FETCHED: Wikipedia Background\n"
                "(Use this for company overview — cite as [wikipedia])\n\n"
                + enrichment["wikipedia"]
            )

        if enrichment.get("edgar"):
            sections.append(
                "## PRE-FETCHED: SEC EDGAR Filings\n"
                "(Use this for official filing data — cite as [edgar])\n\n"
                + enrichment["edgar"]
            )

        if enrichment.get("finnhub"):
            sections.append(
                "## PRE-FETCHED: Finnhub Analyst Ratings & News\n"
                "(Use this for analyst consensus and recent headlines — cite as [yfinance])\n\n"
                + enrichment["finnhub"]
            )

        if enrichment.get("newsapi"):
            sections.append(
                "## PRE-FETCHED: NewsAPI Recent Articles\n"
                "(Use these for recent news context — cite each article as [newsapi] "
                "with its exact URL from the brackets)\n\n"
                + enrichment["newsapi"]
            )

        # ── Research findings ───────────────────────────────────────────
        for i, f in enumerate(findings, 1):
            source_lines = []
            for j, s in enumerate(f.sources, 1):
                url_str = s.url if s.url else "NO_URL_AVAILABLE"
                source_lines.append(
                    f"  [{j}] TITLE: {s.title}\n"
                    f"      URL: {url_str}\n"
                    f"      SNIPPET: {s.snippet[:180] if s.snippet else '(no snippet)'}"
                )
            sources_text = "\n".join(source_lines) if source_lines else "  (none provided)"

            sections.append(
                f"## Research Finding {i} (sub_task_id: {f.sub_task_id})\n"
                f"**Fact check passed:** {f.fact_check_passed} | "
                f"**Confidence:** {f.confidence_score}\n\n"
                f"{f.findings}\n\n"
                f"**Sources (COPY THESE URLs into your citations JSON):**\n"
                f"{sources_text}\n"
            )

        sections.append(
            "\n---\n"
            "Now use the pre-fetched Wikipedia, EDGAR, Finnhub, and NewsAPI data above "
            "PLUS any live yfinance MCP tool calls to produce the final JSON report "
            "as specified in your instructions.\n\n"
            "REMINDER: For crewai_research citations, copy the EXACT URL from the "
            "sources listed above. For newsapi citations, copy the URL from the "
            "brackets in the NewsAPI articles. Never use an empty string for source_url."
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        text: str,
        findings: list[ResearchFindings],
        run_id: str,
        tool_calls: list[dict[str, Any]],
        enrichment: dict[str, str],
        bound_log: Any,
    ) -> SynthesisReport:
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        raw_json: dict[str, Any] | None = None

        if json_match:
            try:
                raw_json = json.loads(json_match.group(1))
                bound_log.debug("synthesis.parse.via_fenced_json")
            except json.JSONDecodeError as exc:
                bound_log.warning("synthesis.parse.fenced_json_invalid", error=str(exc))

        if raw_json is None:
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    raw_json = json.loads(obj_match.group())
                    bound_log.debug("synthesis.parse.via_regex_json")
                except json.JSONDecodeError as exc:
                    bound_log.warning("synthesis.parse.regex_json_invalid", error=str(exc))

        if raw_json is None:
            bound_log.warning("synthesis.parse.falling_back_to_raw_text")
            return self._build_report_from_text(text, findings, run_id, tool_calls, enrichment)

        return self._build_report_from_json(raw_json, findings, run_id, tool_calls, enrichment)

    def _build_report_from_json(
        self,
        raw: dict[str, Any],
        findings: list[ResearchFindings],
        run_id: str,
        tool_calls: list[dict[str, Any]],
        enrichment: dict[str, str],
    ) -> SynthesisReport:
        citations: list[Citation] = []
        for c in raw.get("citations", []):
            tool_used = c.get("tool_used", "crewai_research")
            if tool_used not in ("yfinance", "wikipedia", "edgar", "crewai_research", "newsapi"):
                tool_used = "crewai_research"
            raw_url = (c.get("source_url") or "").strip()
            if raw_url and raw_url.startswith("http"):
                url = raw_url
            elif raw_url and "." in raw_url:
                url = f"https://{raw_url}"
            else:
                url = None
            citations.append(
                Citation(
                    source_title=c.get("source_title", "Unknown"),
                    source_url=url,
                    claim=c.get("claim", ""),
                    tool_used=tool_used,  # type: ignore[arg-type]
                )
            )

        citations = self._enrich_research_citations(citations, findings)
        citations = self._ensure_citations(citations, tool_calls, findings, enrichment)

        confidence_scores: dict[str, float] = {}
        for k, v in raw.get("confidence_scores", {}).items():
            try:
                confidence_scores[k] = float(v)
            except (TypeError, ValueError):
                pass
        if not confidence_scores:
            confidence_scores["overall"] = 0.7

        return SynthesisReport(
            run_id=run_id,
            narrative=raw.get("narrative", "Report generation produced no narrative."),
            citations=citations,
            confidence_scores=confidence_scores,
            recommended_actions=raw.get("recommended_actions", []),
            mcp_tools_called=list({c["tool_name"] for c in tool_calls}),
        )

    def _build_report_from_text(
        self,
        text: str,
        findings: list[ResearchFindings],
        run_id: str,
        tool_calls: list[dict[str, Any]],
        enrichment: dict[str, str],
    ) -> SynthesisReport:
        citations = self._ensure_citations([], tool_calls, findings, enrichment)
        narrative = text.strip() or "The synthesis agent returned an empty response."
        return SynthesisReport(
            run_id=run_id,
            narrative=narrative,
            citations=citations,
            confidence_scores={"overall": 0.5},
            recommended_actions=["Review raw agent output — structured parsing failed."],
            mcp_tools_called=list({c["tool_name"] for c in tool_calls}),
        )

    def _enrich_research_citations(
        self,
        citations: list[Citation],
        findings: list[ResearchFindings],
    ) -> list[Citation]:
        """Back-fill source URLs for crewai_research citations the LLM left blank."""
        research_sources: list[tuple[str, str | None, str]] = []
        seen_urls: set[str] = set()
        for f in findings:
            for s in f.sources:
                url_key = (s.url or "").strip()
                if url_key and url_key in seen_urls:
                    continue
                if url_key:
                    seen_urls.add(url_key)
                research_sources.append((
                    (s.title or "Web Research").strip(),
                    s.url,
                    (s.snippet or "")[:200],
                ))

        if not research_sources:
            return citations

        src_iter = iter(research_sources)
        enriched: list[Citation] = []
        for c in citations:
            if c.tool_used == "crewai_research" and not c.source_url:
                try:
                    title, url, snippet = next(src_iter)
                    enriched.append(Citation(
                        source_title=title, source_url=url,
                        claim=c.claim or snippet or "Research finding",
                        tool_used="crewai_research",  # type: ignore[arg-type]
                    ))
                except StopIteration:
                    enriched.append(c)
            else:
                enriched.append(c)

        existing_urls = {c.source_url for c in enriched if c.source_url}
        for title, url, snippet in src_iter:
            if len(enriched) >= 10:
                break
            if url and url in existing_urls:
                continue
            if url:
                existing_urls.add(url)
            enriched.append(Citation(
                source_title=title, source_url=url,
                claim=snippet or "Research source",
                tool_used="crewai_research",  # type: ignore[arg-type]
            ))
        return enriched

    def _ensure_citations(
        self,
        existing: list[Citation],
        tool_calls: list[dict[str, Any]],
        findings: list[ResearchFindings],
        enrichment: dict[str, str] | None = None,
    ) -> list[Citation]:
        """Guarantee at least one citation. Inject enrichment-based citations if needed."""
        citations = list(existing)

        # Supplement from runner tool calls (yfinance MCP)
        seen_tools: set[str] = {c.tool_used for c in citations}
        _tool_map = {
            "get_stock_price": "yfinance", "get_financials": "yfinance",
        }
        for tc in tool_calls:
            tool_used = _tool_map.get(tc["tool_name"], "crewai_research")
            if tool_used not in seen_tools:
                ticker = tc.get("args", {}).get("ticker", "")
                citations.append(Citation(
                    source_title=f"{tc['tool_name']}({ticker})" if ticker else tc["tool_name"],
                    source_url=None,
                    claim=f"Live data retrieved via {tc['tool_name']}",
                    tool_used=tool_used,  # type: ignore[arg-type]
                ))
                seen_tools.add(tool_used)

        # Inject enrichment-sourced citations if we have none for those tools
        if enrichment:
            if enrichment.get("wikipedia") and "wikipedia" not in seen_tools:
                # Use per-article metadata when available for accurate title + URL
                wiki_meta: list[dict[str, str]] = enrichment.get("wikipedia_meta", [])
                if wiki_meta:
                    for meta in wiki_meta:
                        citations.append(Citation(
                            source_title=meta["title"],
                            source_url=meta["url"],
                            claim="Company background and overview data",
                            tool_used="wikipedia",  # type: ignore[arg-type]
                        ))
                else:
                    citations.append(Citation(
                        source_title="Wikipedia",
                        source_url="https://en.wikipedia.org",
                        claim="Company background and overview data",
                        tool_used="wikipedia",  # type: ignore[arg-type]
                    ))
                seen_tools.add("wikipedia")

            if enrichment.get("edgar") and "edgar" not in seen_tools:
                citations.append(Citation(
                    source_title="SEC EDGAR",
                    source_url="https://www.sec.gov/cgi-bin/browse-edgar",
                    claim="Official SEC 10-K filing data",
                    tool_used="edgar",  # type: ignore[arg-type]
                ))
                seen_tools.add("edgar")

            if enrichment.get("newsapi") and "newsapi" not in seen_tools:
                # Inject per-article citations when available, otherwise one generic entry
                news_meta: list[dict[str, str]] = enrichment.get("newsapi_meta", [])
                if news_meta:
                    for article in news_meta[:5]:  # cap at 5 article citations
                        citations.append(Citation(
                            source_title=article["title"],
                            source_url=article["url"],
                            published_at=article.get("published_at") or None,
                            claim="Recent news coverage",
                            tool_used="newsapi",  # type: ignore[arg-type]
                        ))
                else:
                    # Fallback: link to the actual search rather than the homepage
                    fallback_url = enrichment.get("newsapi_search_url", "https://newsapi.org")
                    citations.append(Citation(
                        source_title="NewsAPI — Recent articles",
                        source_url=fallback_url,
                        claim="Recent news articles (individual article links unavailable)",
                        tool_used="newsapi",  # type: ignore[arg-type]
                    ))
                seen_tools.add("newsapi")

            # Finnhub dated article citations — always inject when available
            # (independent of seen_tools since Finnhub shares "yfinance" tool_used)
            finnhub_meta: list[dict[str, str]] = enrichment.get("finnhub_meta", [])
            for article in finnhub_meta[:3]:
                if article.get("published_at") and article.get("url"):
                    citations.append(Citation(
                        source_title=article.get("title", "Finnhub News"),
                        source_url=article["url"],
                        published_at=article["published_at"],
                        claim="Recent analyst coverage and financial news",
                        tool_used="yfinance",  # type: ignore[arg-type]
                    ))

        # Fall back to research finding sources
        if not citations:
            seen: set[str] = set()
            for f in findings:
                for s in f.sources:
                    url_key = (s.url or "").strip()
                    if url_key and url_key in seen:
                        continue
                    if url_key:
                        seen.add(url_key)
                    citations.append(Citation(
                        source_title=s.title or "Web Research",
                        source_url=s.url,
                        claim=s.snippet[:200] if s.snippet else f"Research finding (sub-task {f.sub_task_id})",
                        tool_used="crewai_research",  # type: ignore[arg-type]
                    ))

        if not citations:
            citations.append(Citation(
                source_title="Research findings (fallback)",
                source_url=None,
                claim="Synthesised from provided research findings",
                tool_used="crewai_research",
            ))

        return citations

    # ------------------------------------------------------------------
    # Fallback report
    # ------------------------------------------------------------------

    def _fallback_report(
        self,
        query: str,
        findings: list[ResearchFindings],
        run_id: str,
        error: str,
    ) -> SynthesisReport:
        log.warning("synthesis_agent.fallback_report", run_id=run_id, error=error)
        try:
            from google import genai as _genai
            _client = _genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

            findings_block = "\n\n".join(
                f"### Research Finding {i + 1}\n"
                f"Topic confidence: {f.confidence_score}\n\n"
                f"{f.findings}\n\n"
                f"Sources: " + ", ".join(
                    f"[{s.title}]({s.url})" if s.url else s.title
                    for s in f.sources[:4]
                )
                for i, f in enumerate(findings)
            )

            prompt = (
                f"You are a senior research analyst. Write a professional intelligence "
                f"report answering: {query}\n\n"
                f"Base your report on these research findings:\n\n{findings_block}\n\n"
                f"Structure with these exact sections:\n"
                f"## Executive Summary\n## Market Position & Outlook\n"
                f"## Key Risks & Challenges\n## Financial Indicators\n"
                f"## Conclusion & Recommendation\n\n"
                f"Write professional flowing prose. Synthesise across all findings."
            )

            from google.genai import types as _gtypes
            response = _client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=_gtypes.GenerateContentConfig(temperature=0.3, max_output_tokens=2048),
            )
            narrative = response.text or ""
            avg_confidence = (
                sum(f.confidence_score for f in findings) / len(findings)
                if findings else 0.6
            )
            confidence_scores = {"overall": round(avg_confidence, 2)}

        except Exception as fallback_exc:
            log.error("synthesis_agent.fallback_gemini_error", error=str(fallback_exc))
            findings_text = "\n\n".join(
                f"**Finding {i + 1}:** {f.findings}" for i, f in enumerate(findings)
            )
            narrative = (
                f"## Executive Summary\n\nResearch on '{query}' produced "
                f"{len(findings)} finding(s) from independent research agents.\n\n"
                f"## Market Position & Outlook\n\n{findings_text}\n\n"
                f"## Conclusion & Recommendation\n\nReview the findings above."
            )
            confidence_scores = {"overall": 0.6}

        citations = self._ensure_citations([], [], findings)
        return SynthesisReport(
            run_id=run_id,
            narrative=narrative,
            citations=citations,
            confidence_scores=confidence_scores,
            recommended_actions=[
                "Verify key financial figures with a live data source.",
                "Cross-reference findings with latest analyst reports.",
                "Monitor for new developments in the identified risk areas.",
            ],
            mcp_tools_called=[],
        )
