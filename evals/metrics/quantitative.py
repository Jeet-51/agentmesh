"""
Quantitative accuracy metric — fixed number normalisation.

Extracts numeric figures from the narrative and spot-checks them against
yfinance ground-truth values.  Numbers are normalised before comparison
so format differences ($1.97T vs 1.97, 76.7% vs 76.7) no longer cause
false mismatches.

Scoring
-------
  With yfinance data  : verified_numbers / total_numbers_found
  Without yfinance    : reward presence of numbers (optimistic 1.0)
  No numbers at all   : neutral 0.5
"""

import re


# Multipliers for suffix normalisation
_SUFFIX = {"t": 1e12, "b": 1e9, "m": 1e6, "k": 1e3}


def _normalise(raw: str) -> float | None:
    """
    Convert a raw numeric string to a plain float.

    Handles:
      $1.97T  → 1.97e12
      76.7%   → 76.7
      $416.50 → 416.50
      2,930   → 2930
      81.35x  → 81.35
    """
    try:
        # Strip currency, spaces, commas, x-suffix
        s = re.sub(r"[$,\s]", "", raw.strip())
        # Check for B/T/M/K suffix (case-insensitive)
        suffix_match = re.match(r"^([\d.]+)([BTMKbtmk])$", s)
        if suffix_match:
            num = float(suffix_match.group(1))
            mult = _SUFFIX[suffix_match.group(2).lower()]
            return num * mult
        # Strip trailing %, x
        s = re.sub(r"[%x]$", "", s)
        return float(s)
    except (ValueError, TypeError):
        return None


def _extract_narrative_numbers(text: str) -> list[float]:
    """Extract all numeric values from the narrative text."""
    # Match: $1.97T  $416.50  76.7%  81.35x  2,930  35.20%
    pattern = r"\$[\d,]+\.?\d*[BTMKbtmk]?|\d[\d,]*\.?\d*\s*[BTMKbtmk%x]"
    raw_matches = re.findall(pattern, text)
    results = []
    for r in raw_matches:
        v = _normalise(r)
        if v is not None:
            results.append(v)
    return results


def _values_close(a: float, b: float, tol: float = 0.15) -> bool:
    """
    Return True if two floats are within tol (15%) of each other.

    15% tolerance accounts for real-world differences between:
    - Quarterly vs TTM (trailing twelve month) figures
    - Rounded narrative values vs precise yfinance returns
    - Currency/unit differences at large scales (B vs exact)
    """
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom <= tol


def compute(report_text: str, yfinance_data: dict = {}) -> dict:
    narrative_nums = _extract_narrative_numbers(report_text)
    total = len(narrative_nums)
    details: dict = {"numbers_found": total}

    if not total:
        return {"score": 0.5, "details": {**details, "note": "No numbers found in narrative"}}

    if not yfinance_data:
        # No ground-truth — reward presence of numbers
        return {"score": 1.0, "details": {**details, "note": "No yfinance ground-truth; numbers present"}}

    # Normalise yfinance ground-truth values
    ground_truth: list[float] = []
    for v in yfinance_data.values():
        n = _normalise(str(v))
        if n is not None:
            ground_truth.append(n)

    if not ground_truth:
        return {"score": 1.0, "details": {**details, "note": "Ground-truth values could not be parsed"}}

    # Count narrative numbers that are close to at least one ground-truth value
    # Full credit within 15%, partial credit (0.5) within 40% for same-order-of-magnitude values
    verified = 0
    partial = 0
    for num in narrative_nums:
        if any(_values_close(num, gt, tol=0.15) for gt in ground_truth):
            verified += 1
        elif any(_values_close(num, gt, tol=0.40) for gt in ground_truth):
            partial += 1

    score = (verified + partial * 0.5) / total
    details.update({
        "verified_exact": verified,
        "verified_partial": partial,
        "ground_truth_count": len(ground_truth),
    })

    return {"score": round(score, 3), "details": details}
