"""
Narrative length & structure metric (Tier 2).

Catches silent failures (empty / very short reports) and verifies that all
five expected report sections are present.

Score breakdown:
  60 % — word-count band score
  40 % — section completeness (all 5 headers present)
"""

import re

_EXPECTED_SECTIONS = [
    "executive summary",
    "market position",
    "key risks",
    "financial indicators",
    "conclusion",
]

# (min_words, max_words_for_full_score, band_score)
_BANDS = [
    (0,    200,  0.05),
    (200,  400,  0.40),
    (400,  700,  0.70),
    (700,  1000, 0.90),
    (1000, 9999, 1.00),
]


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _section_score(text: str) -> tuple[float, list[str], list[str]]:
    lower = text.lower()
    found   = [s for s in _EXPECTED_SECTIONS if s in lower]
    missing = [s for s in _EXPECTED_SECTIONS if s not in lower]
    return len(found) / len(_EXPECTED_SECTIONS), found, missing


def compute(narrative: str) -> dict:
    wc = _word_count(narrative)

    # Band score
    band_score = 0.05
    for lo, hi, score in _BANDS:
        if lo <= wc < hi:
            band_score = score
            break

    sec_score, found, missing = _section_score(narrative)

    overall = round(0.6 * band_score + 0.4 * sec_score, 3)

    return {
        "score": overall,
        "details": {
            "word_count":           wc,
            "band_score":           band_score,
            "section_score":        round(sec_score, 3),
            "sections_found":       found,
            "sections_missing":     missing,
            "flag_too_short":       wc < 200,
        },
    }
