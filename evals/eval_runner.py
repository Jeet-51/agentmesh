"""
AgentMesh Eval Runner — Tier 1 + Tier 2 + Tier 3.

Metrics
-------
  hallucination        — citation count vs paragraph count  (Tier 1 redesign)
  quantitative         — normalised $ / % figures vs yfinance data  (Tier 1 fix)
  freshness            — avg age of dated citations  (Tier 1 fix)
  diversity            — unique source domains / total sources
  entity_coverage      — named entities from query found in narrative
  narrative_length     — word count + section completeness  (Tier 2)
  source_credibility   — weighted domain authority of citations  (Tier 2 new)
  fictional_premise    — hedging vs fabrication on unverifiable claims  (Tier 2 new)
  answer_relevance     — conclusion addresses the query's specific ask  (Tier 2 new)
  tool_activation      — fraction of 5 tools represented in citations  (Tier 3)
  citation_density     — citations per 200 words  (Tier 3)
  overall              — mean of 11 primary metrics above

  confidence_calibration — gap between self-reported and eval overall  (diagnostic)
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.storage.eval_db import init_db, save_report, save_scores
from evals.metrics import (
    hallucination,
    quantitative,
    freshness,
    diversity,
    entity_coverage,
    narrative_length,
    tool_activation,
    citation_density,
    confidence_calibration,
    source_credibility,
    fictional_premise,
    answer_relevance,
)


def _extract_yfinance_data(citations: list) -> dict:
    """
    Pull numeric values from yfinance citation claims for quantitative
    ground-truth.  Preserves original string form so the fixed
    quantitative metric can normalise B/T/M suffixes itself.
    """
    yf_data: dict = {}
    for c in citations:
        if c.get("tool_used") != "yfinance":
            continue
        claim = c.get("claim", "") or ""
        title = c.get("source_title", f"yf_{len(yf_data)}")
        # Keep raw tokens including $ T B M K % suffixes
        tokens = re.findall(r"\$?[\d,]+\.?\d*[BTMKbtmk%]?", claim)
        for i, tok in enumerate(tokens[:10]):
            clean = tok.strip().lstrip("$").replace(",", "")
            if clean:
                yf_data[f"{title}_{i}"] = clean
    return yf_data


def _extract_query_upvs(query: str) -> dict:
    """
    Extract User-Provided Values (UPVs) from the query and return them
    as a ground-truth dict compatible with quantitative.compute().

    When the user states a specific figure in their query (e.g. "$111.2B"),
    the agent is expected to use that exact number. If the narrative
    matches the UPV, it should score as verified — even if it doesn't
    match yfinance TTM figures (which cover a different period).
    """
    upvs: dict = {}
    # Match numbers with financial suffixes or % that the user explicitly stated
    # Patterns: ($111.2B), (47.8%), revenue: $25.7B, 18.2% margin
    tokens = re.findall(r"\$?[\d,]+\.?\d*\s*[BTMKbtmk%]", query)
    for i, tok in enumerate(tokens[:12]):
        clean = tok.strip().lstrip("$").replace(",", "").replace(" ", "")
        if clean:
            upvs[f"upv_{i}"] = clean
    return upvs


def run_eval(
    report_id: str,
    query: str,
    narrative: str,
    sources: list,
    yfinance_data: dict | None = None,
    confidence: float = 0.0,
    report_json: dict | None = None,
) -> dict:
    """
    Score one completed report and persist results to SQLite.

    Parameters
    ----------
    report_id    : Unique run ID.
    query        : Original user query string.
    narrative    : Full text of the generated report narrative.
    sources      : List of citation dicts (source_url, published_at, tool_used, claim).
    yfinance_data: Optional override for quantitative ground-truth.
    confidence   : Self-reported overall confidence from the report (0-1).
    report_json  : Full raw report dict for storage.
    """
    init_db()

    save_report(
        report_id=report_id,
        query=query,
        report_json=report_json or {},
        sources_json=sources,
        confidence=confidence,
    )

    yf = yfinance_data if yfinance_data is not None else _extract_yfinance_data(sources)

    # Merge UPVs from the query into ground-truth so that numbers the user
    # explicitly provided count as verified when they appear in the narrative.
    # This prevents the metric penalising the agent for correctly using a
    # Q1 figure ($111.2B) that doesn't match yfinance's TTM annual value.
    upvs = _extract_query_upvs(query)
    combined_ground_truth = {**upvs, **yf}  # yfinance takes precedence on key collision

    # ── 11 primary quality metrics ────────────────────────────────────────────
    scores: dict = {
        # Tier 1
        "hallucination":      hallucination.compute(narrative, sources),
        "quantitative":       quantitative.compute(narrative, combined_ground_truth),
        "freshness":          freshness.compute(sources),
        "diversity":          diversity.compute(sources),
        "entity_coverage":    entity_coverage.compute(query, narrative),
        # Tier 2
        "narrative_length":   narrative_length.compute(narrative),
        "source_credibility": source_credibility.compute(sources),
        "fictional_premise":  fictional_premise.compute(narrative, query),
        "answer_relevance":   answer_relevance.compute(query, narrative),
        # Tier 3
        "tool_activation":    tool_activation.compute(sources),
        "citation_density":   citation_density.compute(narrative, sources),
    }

    # Overall = mean of all 11 primary metrics
    overall = sum(m["score"] for m in scores.values()) / len(scores)
    scores["overall"] = {"score": round(overall, 3), "details": {}}

    # ── Diagnostic (not in overall) ───────────────────────────────────────────
    scores["confidence_calibration"] = confidence_calibration.compute(confidence, overall)

    save_scores(report_id, scores)
    return scores
