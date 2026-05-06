"""
Tool activation metric (Tier 3).

Measures what fraction of the 5 available data sources are represented
in the report's citations.  A high score means the multi-agent pipeline
actually leveraged its full toolset.

Expected tools: yfinance, wikipedia, edgar, crewai_research, newsapi
"""

_ALL_TOOLS = {"yfinance", "wikipedia", "edgar", "crewai_research", "newsapi"}


def compute(citations: list) -> dict:
    activated = {
        c.get("tool_used") for c in citations
        if c.get("tool_used") in _ALL_TOOLS
    }
    score = len(activated) / len(_ALL_TOOLS)

    inactive = _ALL_TOOLS - activated

    return {
        "score": round(score, 3),
        "details": {
            "tools_activated": sorted(activated),
            "tools_inactive":  sorted(inactive),
            "activated_count": len(activated),
            "total_possible":  len(_ALL_TOOLS),
        },
    }
