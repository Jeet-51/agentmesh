"""
Hallucination proxy metric — redesigned (Tier 1 fix).

OLD approach: scanned sentence text for source-tag keywords ("yfinance",
"wikipedia", etc.).  Consistently scored ~0.42 because our reports use
structured JSON citations, not inline tags.

NEW approach — citation count vs paragraph count:
  score = min(1.0, n_citations / max(1, n_paragraphs))

Each meaningful paragraph of the narrative should be backed by at least
one citation.  More citations than paragraphs → score 1.0.
Fewer citations than paragraphs → proportional penalty.

Fallback (no citations provided): original keyword scan is used so the
metric still produces a signal even on bare-text reports.
"""

import re

_SOURCE_TAGS = {"research", "yfinance", "edgar", "newsapi", "wikipedia", "finnhub"}


def _count_paragraphs(text: str) -> int:
    """Count meaningful paragraph blocks (> 40 chars)."""
    blocks = [b.strip() for b in re.split(r"\n{2,}", text)]
    return max(1, sum(1 for b in blocks if len(b) > 40))


def _keyword_fallback(report_text: str) -> dict:
    """Original approach — used when no citation objects are available."""
    sentences = [
        s.strip() for s in re.split(r"[.!?]", report_text)
        if len(s.strip()) > 20
    ]
    if not sentences:
        return {
            "score": 0.0,
            "details": {"method": "keyword_fallback", "total_sentences": 0, "cited_sentences": 0},
        }
    cited = [s for s in sentences if any(t in s.lower() for t in _SOURCE_TAGS)]
    return {
        "score": round(len(cited) / len(sentences), 3),
        "details": {
            "method":            "keyword_fallback",
            "total_sentences":   len(sentences),
            "cited_sentences":   len(cited),
            "uncited_sentences": len(sentences) - len(cited),
        },
    }


def compute(report_text: str, citations: list = []) -> dict:
    if not citations:
        return _keyword_fallback(report_text)

    n_paragraphs = _count_paragraphs(report_text)
    n_citations  = len(citations)
    score        = min(1.0, n_citations / n_paragraphs)

    return {
        "score": round(score, 3),
        "details": {
            "method":          "citation_vs_paragraph",
            "n_citations":     n_citations,
            "n_paragraphs":    n_paragraphs,
            "ratio":           round(n_citations / n_paragraphs, 2),
        },
    }
