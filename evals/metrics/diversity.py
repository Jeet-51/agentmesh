"""
Source diversity metric.

Measures what fraction of cited URLs come from distinct domains.
A score of 1.0 means every citation is from a unique domain.
"""

from urllib.parse import urlparse


def compute(sources: list) -> dict:
    domains: list[str] = []

    for s in sources:
        url = s.get("source_url", "") or ""
        if url.startswith("http"):
            try:
                netloc = urlparse(url).netloc
                domain = netloc.replace("www.", "").strip()
                if domain:
                    domains.append(domain)
            except Exception:
                pass

    if not domains:
        return {
            "score": 0.5,
            "details": {"message": "No valid URLs found — defaulting to 0.5"},
        }

    unique = set(domains)
    score = len(unique) / len(domains)

    return {
        "score": round(score, 3),
        "details": {
            "total_sources":  len(domains),
            "unique_domains": len(unique),
            "domains":        sorted(unique),
        },
    }
