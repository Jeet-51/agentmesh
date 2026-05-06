"""
Answer Relevance metric — Tier 2.

Measures whether the report's conclusion and recommendations actually
address the original query — not just whether query words appear in the
text (that's entity_coverage), but whether the *intent* is answered.

Method
------
1. Extract key action verbs + noun phrases from the query.
2. Check if the conclusion / recommendation section references them.
3. Penalise if the conclusion is purely generic boilerplate.

Scoring
-------
  1.0 — Conclusion directly addresses query's specific ask
  0.7 — Partial coverage; some query intent addressed
  0.5 — Neutral / conclusion not found
  0.3 — Generic boilerplate with no query-specific content
"""

import re


# Generic filler phrases that indicate boilerplate recommendations
_BOILERPLATE = [
    r"monitor (developments|progress|the situation)",
    r"conduct further research",
    r"stay informed",
    r"consult (a |an )?(financial |professional |qualified )?advisor",
    r"consider (your |the )?(risk|portfolio|options)",
    r"due diligence",
    r"past performance (is not|does not)",
]

# Action verbs that signal a query is asking for analysis
_QUERY_ACTIONS = [
    "evaluate", "assess", "compare", "analyze", "analyse",
    "determine", "calculate", "measure", "quantify", "explain",
    "identify", "estimate", "forecast", "predict",
]


def _extract_conclusion(narrative: str) -> str:
    """Extract the conclusion / recommendation section of the narrative."""
    lower = narrative.lower()
    # Try to find conclusion section
    for marker in ["conclusion", "recommendation", "summary", "outlook"]:
        idx = lower.rfind(marker)
        if idx != -1:
            return narrative[idx:]
    # Fall back to last 25% of text
    cutoff = int(len(narrative) * 0.75)
    return narrative[cutoff:]


def _extract_query_nouns(query: str) -> list[str]:
    """Extract meaningful nouns/phrases from the query (min 4 chars, not stopwords)."""
    stopwords = {
        "what", "does", "the", "for", "and", "are", "how", "why",
        "will", "with", "that", "this", "from", "they", "their",
        "have", "been", "into", "over", "also", "both", "each",
        "specifically", "accounting", "impact", "effect",
        # Common query structure words that rarely appear in conclusions
        "comparing", "compare", "between", "versus", "given", "using",
        "including", "regarding", "within", "across", "during", "about",
        "quarter", "fiscal", "year", "period", "current", "recent",
    }
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]{3,}\b", query)
    return [w.lower() for w in words if w.lower() not in stopwords]


def _boilerplate_ratio(text: str) -> float:
    """Return fraction of boilerplate patterns matched."""
    lower = text.lower()
    hits = sum(1 for p in _BOILERPLATE if re.search(p, lower))
    return hits / len(_BOILERPLATE)


def compute(query: str, narrative: str) -> dict:
    """
    Parameters
    ----------
    query     : Original user query.
    narrative : Full report narrative.
    """
    if not narrative or not query:
        return {"score": 0.5, "details": {"note": "Missing query or narrative"}}

    conclusion = _extract_conclusion(narrative)
    query_nouns = _extract_query_nouns(query)

    if not query_nouns:
        return {"score": 0.5, "details": {"note": "No extractable query terms"}}

    # Check how many query nouns appear in the conclusion
    conclusion_lower = conclusion.lower()
    matched = [n for n in query_nouns if n in conclusion_lower]
    coverage = len(matched) / len(query_nouns)

    # Check for generic boilerplate in conclusion
    boilerplate = _boilerplate_ratio(conclusion)

    # Check if query has a specific action verb
    query_lower = query.lower()
    has_action = any(v in query_lower for v in _QUERY_ACTIONS)

    # Score calculation — realistic thresholds for financial reports
    # Conclusions typically reference company names + 1-2 key metrics,
    # not every noun from the query
    if coverage >= 0.5 and boilerplate < 0.3:
        score = 1.0
    elif coverage >= 0.35 and boilerplate < 0.4:
        score = 0.8
    elif coverage >= 0.2:
        score = 0.6 - (boilerplate * 0.2)
    elif coverage >= 0.1:
        score = 0.5
    else:
        score = 0.3

    # Bonus: if specific action verb in query and conclusion addresses it
    if has_action and coverage >= 0.35:
        score = min(1.0, score + 0.1)

    return {
        "score": round(score, 3),
        "details": {
            "query_nouns": len(query_nouns),
            "matched_in_conclusion": len(matched),
            "coverage": round(coverage, 3),
            "boilerplate_ratio": round(boilerplate, 3),
        },
    }
