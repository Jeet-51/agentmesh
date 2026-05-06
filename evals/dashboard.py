"""
AgentMesh Eval Dashboard — CLI.

Usage:
    python evals/dashboard.py
    python evals/dashboard.py --limit 20
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.storage.eval_db import init_db, get_all_scores

# ─── ANSI colours ────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"


def _colour_score(score: float) -> str:
    s = f"{score:.2f}"
    if score >= 0.75:
        return f"{_GREEN}{s}{_RESET}"
    if score >= 0.50:
        return f"{_YELLOW}{s}{_RESET}"
    return f"{_RED}{s}{_RESET}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def show(limit: int = 10) -> None:
    init_db()
    rows = get_all_scores()

    if not rows:
        print(f"\n{_YELLOW}No eval results yet.{_RESET}")
        print(f"{_DIM}Run a query at http://localhost:3000, then check back here.{_RESET}\n")
        return

    # Group by (query, timestamp) → metric scores
    by_report: dict = defaultdict(dict)
    meta: dict = {}
    for query, conf, ts, metric, score, details in rows:
        key = (ts, query)
        by_report[key][metric] = score
        meta[key] = (query, conf, ts)

    # Sort newest first, cap at limit
    sorted_keys = sorted(by_report.keys(), key=lambda k: k[0], reverse=True)[:limit]

    # ── Header ───────────────────────────────────────────────────────────────
    col_q = 52
    print(f"\n{_BOLD}{_CYAN}{'═' * 105}{_RESET}")
    print(
        f"{_BOLD}{'AgentMesh Eval Dashboard':^105}{_RESET}"
    )
    print(f"{_BOLD}{_CYAN}{'═' * 105}{_RESET}\n")

    # Detect which metric generations exist in this DB
    all_metric_names = {m for row_m in by_report.values() for m in row_m}
    has_narr   = "narrative_length"   in all_metric_names
    has_tier3  = "tool_activation"    in all_metric_names
    has_new    = "source_credibility" in all_metric_names

    if has_new:
        primary_metrics = [
            "hallucination", "quantitative", "freshness", "diversity",
            "entity_coverage", "narrative_length",
            "source_credibility", "fictional_premise", "answer_relevance",
            "overall",
        ]
        header = (
            f"{'Query':<{col_q}} {'Hall':>6} {'Quant':>6} {'Fresh':>6} "
            f"{'Div':>6} {'Ent':>6} {'Narr':>6} {'Cred':>6} {'Fict':>6} {'Rel':>6} {'Overall':>9}"
        )
        rule_w = 140
    elif has_narr:
        primary_metrics = [
            "hallucination", "quantitative", "freshness",
            "diversity", "entity_coverage", "narrative_length", "overall",
        ]
        header = (
            f"{'Query':<{col_q}} {'Hall':>6} {'Quant':>6} "
            f"{'Fresh':>6} {'Div':>6} {'Ent':>6} {'Narr':>6} {'Overall':>9}"
        )
        rule_w = 113
    else:
        primary_metrics = [
            "hallucination", "quantitative", "freshness",
            "diversity", "entity_coverage", "overall",
        ]
        header = (
            f"{'Query':<{col_q}} {'Hall':>6} {'Quant':>6} "
            f"{'Fresh':>6} {'Div':>6} {'Ent':>6} {'Overall':>9}"
        )
        rule_w = 105

    print(f"{_BOLD}{_WHITE}{header}{_RESET}")
    print(f"{_DIM}{'─' * rule_w}{_RESET}")

    # ── Rows ─────────────────────────────────────────────────────────────────
    for key in sorted_keys:
        query, conf, ts = meta[key]
        m = by_report[key]
        ts_short = ts[:16].replace("T", " ")

        q_display = (query[:col_q - 3] + "...") if len(query) > col_q else query

        scores_str = " ".join(
            f"{_colour_score(m.get(name, 0.0)):>6}" if name != "overall"
            else f"{_colour_score(m.get(name, 0.0)):>9}"
            for name in primary_metrics
        )

        print(f"{q_display:<{col_q}} {scores_str}")

        # Tier 3 diagnostic line
        if has_tier3:
            tools  = m.get("tool_activation", 0.0)
            ciden  = m.get("citation_density", 0.0)
            ccal   = m.get("confidence_calibration", 0.0)
            diag = (
                f"  {_DIM}└ {ts_short}"
                f"  |  Tools:{_RESET}{_colour_score(tools)}"
                f"{_DIM}  CiDen:{_RESET}{_colour_score(ciden)}"
                f"{_DIM}  CCal:{_RESET}{_colour_score(ccal)}"
            )
            print(diag)
        else:
            print(f"{_DIM}  └ {ts_short}{_RESET}")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    if len(sorted_keys) > 1:
        print(f"\n{_DIM}{'─' * rule_w}{_RESET}")
        print(f"{_BOLD}{'Averages across ' + str(len(sorted_keys)) + ' reports':<{col_q}}{_RESET}", end="")
        for name in primary_metrics:
            vals = [by_report[k].get(name, 0.0) for k in sorted_keys if name in by_report[k]]
            avg = sum(vals) / len(vals) if vals else 0.0
            pad = 9 if name == "overall" else 6
            print(f" {_colour_score(avg):>{pad}}", end="")
        print()

        if has_tier3:
            tier3_names = ["tool_activation", "citation_density", "confidence_calibration"]
            tier3_labels = ["Tools", "CiDen", "CCal"]
            print(f"{_DIM}  Diagnostics:{_RESET}", end="")
            for name, label in zip(tier3_names, tier3_labels):
                vals = [by_report[k].get(name, 0.0) for k in sorted_keys if name in by_report[k]]
                avg = sum(vals) / len(vals) if vals else 0.0
                print(f"  {_DIM}{label}:{_RESET}{_colour_score(avg)}", end="")
            print()

    print(f"\n{_DIM}DB: {os.path.abspath(os.environ.get('EVAL_DB_PATH', 'evals/agentmesh_evals.db'))}{_RESET}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentMesh Eval Dashboard")
    parser.add_argument("--limit", type=int, default=10, help="Max reports to show (default: 10)")
    args = parser.parse_args()
    show(limit=args.limit)
