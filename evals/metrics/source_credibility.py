"""
Source Credibility metric — Tier 2.

Ranks each citation's source domain against a tiered credibility hierarchy
and returns the weighted average score across all citations.

Tier 1 (1.00) — Primary sources: government, regulators, central banks
Tier 2 (0.80) — Major financial press: Reuters, Bloomberg, FT, WSJ, AP
Tier 3 (0.65) — Established finance/data: Yahoo Finance, Seeking Alpha,
                 Macrotrends, MarketWatch, CNBC, Forbes, Investopedia
Tier 4 (0.50) — Reference / encyclopaedic: Wikipedia, Britannica
Tier 5 (0.30) — Unknown / unclassified domains
"""

import re
from urllib.parse import urlparse


_TIERS: list[tuple[float, list[str]]] = [
    (1.00, [
        # Government & regulatory — primary sources
        "sec.gov", "federalreserve.gov", "bls.gov", "census.gov",
        "irs.gov", "treasury.gov", "whitehouse.gov", "congress.gov",
        "europa.eu", "ecb.europa.eu", "bis.org", "imf.org",
        "worldbank.org", "oecd.org", "un.org", "who.int",
        "energy.gov", "epa.gov", "ftc.gov", "doj.gov",
        # Official company investor relations pages
        "investor.apple.com", "ir.apple.com",
        "investor.microsoft.com", "ir.microsoft.com",
        "ir.nvidia.com", "investor.nvidia.com",
        "ir.tesla.com", "investor.tesla.com",
        "abc.xyz",  # Alphabet/Google IR
    ]),
    (0.80, [
        # Major financial press & data providers
        "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
        "apnews.com", "economist.com", "businessinsider.com",
        "barrons.com", "morningstar.com", "spglobal.com",
        "moodys.com", "fitchratings.com", "iea.org",
        # Financial data APIs — structured, reliable data
        "finnhub.io", "alphavantage.co", "quandl.com",
        "stlouisfed.org", "fred.stlouisfed.org",
        # Established business news
        "nytimes.com", "washingtonpost.com", "guardian.com",
        "bbc.com", "bbc.co.uk",
    ]),
    (0.65, [
        # Established finance & tech media
        "finance.yahoo.com", "yahoo.com", "seekingalpha.com",
        "macrotrends.net", "marketwatch.com", "cnbc.com",
        "forbes.com", "investopedia.com", "businessquant.com",
        "tradingeconomics.com", "statista.com", "ainvest.com",
        "financialcontent.com", "techcrunch.com",
        "wired.com", "theverge.com", "arstechnica.com",
        "venturebeat.com", "zdnet.com", "cnet.com",
        # Additional financial sources
        "fool.com", "motleyfool.com", "stockanalysis.com",
        "simplywall.st", "wisesheets.io", "finviz.com",
    ]),
    (0.50, [
        # Reference / encyclopaedic
        "wikipedia.org", "britannica.com", "encyclopedia.com",
    ]),
]


def _domain_score(url: str) -> float:
    """Return the credibility score for a single URL."""
    if not url:
        return 0.30
    try:
        host = urlparse(url).netloc.lower()
        host = re.sub(r"^www\.", "", host)
    except Exception:
        return 0.30

    for score, domains in _TIERS:
        if any(host == d or host.endswith("." + d) for d in domains):
            return score
    return 0.30  # Tier 5 — unknown


def compute(citations: list) -> dict:
    """
    Parameters
    ----------
    citations : list of citation dicts with at least a 'source_url' key.
    """
    if not citations:
        return {"score": 0.5, "details": {"note": "No citations provided"}}

    scores = []
    breakdown: dict[str, int] = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0, "unknown": 0}

    for c in citations:
        url = c.get("source_url") or c.get("url") or ""
        s = _domain_score(url)
        scores.append(s)
        if s >= 1.00:   breakdown["tier1"]   += 1
        elif s >= 0.80: breakdown["tier2"]   += 1
        elif s >= 0.65: breakdown["tier3"]   += 1
        elif s >= 0.50: breakdown["tier4"]   += 1
        else:           breakdown["unknown"] += 1

    avg = sum(scores) / len(scores)
    return {
        "score": round(avg, 3),
        "details": {
            "citation_count": len(citations),
            "breakdown": breakdown,
            "avg_credibility": round(avg, 3),
        },
    }
