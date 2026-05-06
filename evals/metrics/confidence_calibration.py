"""
Confidence calibration metric (Tier 3).

Compares the model's self-reported confidence score against the actual
eval overall score.  A well-calibrated model should report confidence
close to what the eval measures.

score = max(0, 1 - gap / 0.5)
  gap = 0.00 → score = 1.00 (perfectly calibrated)
  gap = 0.25 → score = 0.50 (moderately overconfident)
  gap = 0.50 → score = 0.00 (badly miscalibrated)
"""


def compute(self_reported: float, eval_overall: float) -> dict:
    gap = abs(self_reported - eval_overall)
    score = max(0.0, 1.0 - gap / 0.5)

    direction = (
        "overconfident" if self_reported > eval_overall
        else "underconfident" if self_reported < eval_overall
        else "calibrated"
    )

    return {
        "score": round(score, 3),
        "details": {
            "self_reported":   round(self_reported, 3),
            "eval_overall":    round(eval_overall, 3),
            "gap":             round(gap, 3),
            "direction":       direction,
        },
    }
