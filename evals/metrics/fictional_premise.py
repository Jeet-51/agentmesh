"""
Fictional Premise Detection metric — Tier 2.

Measures how well the report handles unverifiable or hypothetical claims
in the query WITHOUT fabricating specific numbers or facts.

A good report should:
  - Flag unverifiable premises as "hypothetical", "could not be confirmed", etc.
  - Use hedging language for uncertain claims
  - NOT invent specific figures for fictional events

Scoring
-------
  1.0 — Strong hedging language found; no fabrication signals detected
  0.7 — Some hedging language present
  0.5 — Neutral / no signal either way (query had no fictional premises)
  0.2 — Fabrication signals detected without corresponding hedges
"""

import re


# Phrases that indicate the agent correctly flagged unverifiable content
_HEDGE_PATTERNS = [
    r"could not be (conclusively |fully |independently )?determined",
    r"could not be (confirmed|verified|quantified|found|established)",
    r"not (available|specified|confirmed|provided|disclosed|reported)",
    r"hypothetical(ly)?",
    r"unavailable in (the |available )?research",
    r"no (direct|specific|public) (data|information|evidence|disclosure)",
    r"cannot be (fully |directly )?quantified",
    r"unclear (from|based on) (the |available )?",
    r"not (yet )?publicly (disclosed|available|reported)",
    r"based on (available|current) (data|research|information)",
    r"further research (is needed|required|recommended)",
    r"unverified",
    r"alleged",
    r"reportedly",
    r"according to (unconfirmed|limited)",
]

# Phrases that suggest the agent fabricated answers for fictional premises
_FABRICATION_SIGNALS = [
    r"the ceasefire (directly |specifically )?(resulted in|caused|led to|increased|decreased)",
    r"the (15|85|15-85)[- ]?% (price )?increase (resulted|caused|led|boosted|reduced)",
    r"brent crude (at|of) \$[\d]+\.?\d* (offset|reduced|lowered|cut)",
    r"the \d+\.?\d*gw (agreement|deal|contract) (cost|requires|needs) \$[\d]",
    r"as a (direct )?result of the (ceasefire|tariff|sanction|embargo)",
]


def _count_hedges(text: str) -> int:
    count = 0
    lower = text.lower()
    for p in _HEDGE_PATTERNS:
        if re.search(p, lower):
            count += 1
    return count


def _count_fabrications(text: str) -> int:
    count = 0
    lower = text.lower()
    for p in _FABRICATION_SIGNALS:
        if re.search(p, lower):
            count += 1
    return count


def compute(narrative: str, query: str = "") -> dict:
    """
    Parameters
    ----------
    narrative : Full report narrative text.
    query     : Original user query (used to detect if fictional premises exist).
    """
    if not narrative:
        return {"score": 0.5, "details": {"note": "Empty narrative"}}

    hedges = _count_hedges(narrative)
    fabrications = _count_fabrications(narrative)

    # Detect if query contains hallucination-pressure signals
    query_lower = query.lower()
    pressure_signals = [
        "ceasefire", "price increase", "oil price", "crude",
        "specifically accounting", "exact", "precisely",
    ]
    has_pressure = sum(1 for s in pressure_signals if s in query_lower)

    details = {
        "hedge_phrases_found": hedges,
        "fabrication_signals": fabrications,
        "query_pressure_signals": has_pressure,
    }

    if fabrications > 0:
        # Fabricated answers — penalise heavily regardless of query type
        score = max(0.1, 0.4 - (fabrications * 0.1))
        return {"score": round(score, 3), "details": {**details, "note": "Fabrication signals detected"}}

    if has_pressure >= 2:
        # High fictional pressure in query — requires strong hedging
        if hedges >= 3:
            score = 1.0
        elif hedges >= 1:
            score = 0.7
        else:
            score = 0.4  # Pressure present, no hedging = risky
    elif has_pressure == 1:
        # Mild pressure — some hedging expected
        if hedges >= 2:
            score = 1.0
        elif hedges >= 1:
            score = 0.8
        else:
            score = 0.5
    else:
        # Normal financial query — reward good hedging practice
        # A well-written report still uses hedging for estimates/projections
        if hedges >= 3:
            score = 0.9
        elif hedges >= 1:
            score = 0.7
        else:
            score = 0.5  # Neutral — no pressure, no hedging required

    return {"score": round(score, 3), "details": details}
