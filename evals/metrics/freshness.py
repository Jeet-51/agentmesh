"""
Source freshness metric.

Parses ISO-8601 `published_at` fields from citations and computes how
recent the sources are on average.  A source from today scores 1.0; one
from a year ago scores 0.0; linear in between.
"""

from datetime import datetime


def compute(sources: list) -> dict:
    dates: list[int] = []
    today = datetime.utcnow()

    for source in sources:
        published = source.get("published_at", "") or ""
        if not published:
            continue
        try:
            d = datetime.fromisoformat(published.replace("Z", ""))
            days_old = (today - d).days
            dates.append(max(0, days_old))
        except (ValueError, TypeError):
            pass

    if not dates:
        return {
            "score": 0.5,
            "details": {"message": "No dated sources found — defaulting to 0.5"},
        }

    avg_days = sum(dates) / len(dates)
    score = max(0.0, 1.0 - (avg_days / 365))

    return {
        "score": round(score, 3),
        "details": {
            "avg_days_old":        round(avg_days),
            "freshest_days_old":   min(dates),
            "oldest_days_old":     max(dates),
            "sources_with_dates":  len(dates),
        },
    }
