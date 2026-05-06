"""
LLM-as-judge hallucination metric (Tier 4).

Uses Gemini Flash to assess whether claims in the narrative are genuinely
supported by the provided citations.  This gives a true hallucination
signal rather than the keyword-proxy used by hallucination.py.

Requires GOOGLE_API_KEY in environment or project .env file.
Falls back to 0.5 if the API call fails (preserving pipeline stability).
"""

import json
import os
import re
import sys
import httpx

# ─── API key resolution ───────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Read GOOGLE_API_KEY from env or project .env file."""
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    # Walk up to project root and read .env
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        env_path = os.path.join(here, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().lstrip()
        here = os.path.dirname(here)
    return ""


# ─── Gemini REST call ─────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are an expert research editor and fact-checker.

Below is a research report narrative followed by its citations.
Your task: assess whether the CLAIMS in the narrative are supported by the citations provided.

NARRATIVE (first 2500 chars):
{narrative}

CITATIONS ({n_citations} total):
{citations_text}

Rate on a scale 0.0 to 1.0:
- 1.00: Virtually all specific claims (facts, figures, statements) have clear citation backing
- 0.75: Most claims are cited; a few minor points lack explicit citation
- 0.50: Roughly half the specific claims have citation support
- 0.25: Most claims appear unsupported or citations are vague
- 0.00: Narrative contains many unsupported assertions

Respond with ONLY valid JSON (no markdown fences):
{{"score": 0.XX, "reasoning": "one sentence", "uncited_claims": ["example claim 1", "example claim 2"]}}
"""


def compute(narrative: str, citations: list, query: str) -> dict:
    """
    Synchronous entry point — safe to call from threads and asyncio executors.
    Returns {"score": float, "details": dict}.
    """
    api_key = _get_api_key()
    if not api_key:
        return {
            "score": 0.5,
            "details": {"message": "GOOGLE_API_KEY not found — defaulting to 0.5"},
        }
    if not narrative or len(narrative.strip()) < 100:
        return {
            "score": 0.0,
            "details": {"message": "Narrative too short to judge"},
        }

    # Format citations for the judge prompt
    citations_text = "\n".join(
        f"  [{i+1}] [{c.get('tool_used','?')}] {c.get('source_title','?')}: {c.get('claim','')[:120]}"
        for i, c in enumerate(citations[:15])
    ) or "  (no citations provided)"

    prompt = _JUDGE_PROMPT.format(
        narrative=narrative[:2500],
        n_citations=len(citations),
        citations_text=citations_text,
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=payload)
        r.raise_for_status()
        raw = r.json()
        text = (
            raw.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        # Strip markdown fences if present
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)
        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        return {
            "score": round(score, 3),
            "details": {
                "reasoning":      result.get("reasoning", ""),
                "uncited_claims": result.get("uncited_claims", [])[:5],
                "model":          "gemini-2.0-flash",
            },
        }
    except json.JSONDecodeError as exc:
        return {
            "score": 0.5,
            "details": {"message": f"JSON parse error from judge: {exc}", "raw": text[:200]},
        }
    except Exception as exc:
        return {
            "score": 0.5,
            "details": {"message": f"LLM judge error: {exc}"},
        }
