import sqlite3
import json
from datetime import datetime
import os

# EVAL_DB_PATH env var is set to /app/evals/agentmesh_evals.db inside Docker.
# Locally it falls back to <project-root>/evals/agentmesh_evals.db.
_evals_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("EVAL_DB_PATH", os.path.join(_evals_dir, "agentmesh_evals.db"))


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id        TEXT PRIMARY KEY,
            query     TEXT,
            report_json  TEXT,
            sources_json TEXT,
            confidence   REAL,
            timestamp    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_scores (
            report_id TEXT,
            metric    TEXT,
            score     REAL,
            details   TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_report(report_id: str, query: str, report_json: dict,
                sources_json: list, confidence: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?,?)",
        (
            report_id,
            query,
            json.dumps(report_json),
            json.dumps(sources_json),
            confidence,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def save_scores(report_id: str, scores: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    ts = datetime.utcnow().isoformat()
    for metric, data in scores.items():
        conn.execute(
            "INSERT INTO eval_scores VALUES (?,?,?,?,?)",
            (
                report_id,
                metric,
                data["score"],
                json.dumps(data.get("details", {})),
                ts,
            ),
        )
    conn.commit()
    conn.close()


def get_all_scores() -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT r.query, r.confidence, r.timestamp,
               e.metric, e.score, e.details
        FROM reports r
        JOIN eval_scores e ON r.id = e.report_id
        ORDER BY r.timestamp DESC
    """).fetchall()
    conn.close()
    return rows


def get_recent_scores(limit: int = 20) -> list:
    """Return (query, timestamp, metric→score) tuples for the last N reports."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT r.id, r.query, r.timestamp,
               e.metric, e.score
        FROM reports r
        JOIN eval_scores e ON r.id = e.report_id
        ORDER BY r.timestamp DESC
        LIMIT ?
    """, (limit * 10,)).fetchall()
    conn.close()
    return rows
