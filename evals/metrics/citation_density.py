"""
Citation density metric (Tier 3).

Measures how many citations back the narrative per 200 words.
Target: at least 1 citation per 200 words (5 per 1 000 words).

Score = min(1.0, citation_count / expected_citations)
where expected_citations = max(3, word_count / 200)
"""

import re


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def compute(narrative: str, citations: list) -> dict:
    wc = _word_count(narrative)
    n_citations = len(citations)

    # Expect at least 1 citation per 200 words, minimum expectation of 3
    expected = max(3, wc / 200)
    score = min(1.0, n_citations / expected)

    return {
        "score": round(score, 3),
        "details": {
            "word_count":       wc,
            "citation_count":   n_citations,
            "expected_min":     round(expected, 1),
            "density_per_1k":   round(n_citations / max(1, wc) * 1000, 2),
        },
    }
