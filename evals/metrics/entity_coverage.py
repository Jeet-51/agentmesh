"""
Entity coverage metric.

Extracts named entities (proper nouns / acronyms) from the user query and
checks whether each appears in the narrative.  High coverage means the
report actually addressed what was asked.
"""

import re

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "how", "what", "why", "when", "where", "who",
    "in", "of", "for", "and", "or", "to", "vs",
    "with", "its", "their", "has", "have", "had",
    "will", "would", "could", "should", "can",
}


def extract_entities(query: str) -> list[str]:
    """Return proper nouns and acronyms found in the query."""
    # Capitalised words (Title Case) and ALL-CAPS acronyms (2+ letters)
    words = re.findall(r"\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b", query)
    return [w for w in words if w.lower() not in _STOPWORDS]


def compute(query: str, report_text: str) -> dict:
    entities = extract_entities(query)

    if not entities:
        return {
            "score": 1.0,
            "details": {"message": "No named entities found in query"},
        }

    found   = [e for e in entities if e in report_text]
    missing = [e for e in entities if e not in report_text]
    score   = len(found) / len(entities)

    return {
        "score": round(score, 3),
        "details": {
            "entities": entities,
            "found":    found,
            "missing":  missing,
        },
    }
