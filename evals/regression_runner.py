"""
AgentMesh Regression Runner — Tier 1.

Automatically submits all hard prompts, auto-approves human checkpoints,
waits for completion, and prints a final pass/fail summary.

Usage:
    python evals/regression_runner.py
    python evals/regression_runner.py --start 3      # resume from prompt #3
    python evals/regression_runner.py --dry-run      # print prompts, don't run

Total time: ~25-30 minutes for all 10 queries.
"""

import json
import asyncio
import httpx
import time
import argparse
import sys
import os
from datetime import datetime, timezone

GATEWAY_URL   = "http://localhost:8000"
PROMPTS_FILE  = os.path.join(os.path.dirname(__file__), "dataset", "hard_prompts.json")

# ─── ANSI colours ─────────────────────────────────────────────────────────────
_R = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[92m"
_YELLOW= "\033[93m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_WHITE = "\033[97m"

def _ok(s):   return f"{_GREEN}{s}{_R}"
def _warn(s): return f"{_YELLOW}{s}{_R}"
def _err(s):  return f"{_RED}{s}{_R}"
def _dim(s):  return f"{_DIM}{s}{_R}"
def _bold(s): return f"{_BOLD}{s}{_R}"

# ─── Single query ─────────────────────────────────────────────────────────────

async def run_single_query(client: httpx.AsyncClient, query: str) -> dict:
    """
    Full lifecycle for one query:
      1. Submit  → get run_id
      2. Sleep 5 s for decomposition
      3. Auto-approve the human checkpoint
      4. Poll until terminal status (max 5 min / 60 × 5 s)
    """

    # Step 1: Submit
    try:
        r = await client.post(f"{GATEWAY_URL}/run", json={"query": query})
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Gateway /run returned {exc.response.status_code}: {exc.response.text[:200]}")

    run_id = r.json()["run_id"]
    print(f"    {_dim('run_id:')} {_cyan(run_id[:12])}...")

    # Step 2: Wait for decomposition
    await asyncio.sleep(5)

    # Step 3: Auto-approve checkpoint (best-effort — may already be approved)
    try:
        checkpoint = await client.get(f"{GATEWAY_URL}/checkpoint/{run_id}")
        if checkpoint.status_code == 200:
            data      = checkpoint.json()
            sub_tasks = data.get("sub_tasks") or []
            approve_r = await client.post(
                f"{GATEWAY_URL}/checkpoint/{run_id}/approve",
                json={"sub_tasks": sub_tasks, "approved": True},
            )
            if approve_r.status_code == 200:
                print(f"    {_ok('OK')} Checkpoint approved ({len(sub_tasks)} sub-tasks)")
            else:
                print(f"    {_warn('!')} Approve returned {approve_r.status_code}")
        elif checkpoint.status_code == 404:
            print(f"    {_dim('(no checkpoint — already approved or not needed)')}")
        else:
            print(f"    {_warn('!')} Checkpoint GET returned {checkpoint.status_code}")
    except Exception as exc:
        print(f"    {_warn('!')} Checkpoint step failed: {exc}")

    # Step 4: Poll until terminal
    for tick in range(60):
        await asyncio.sleep(5)
        try:
            status_r = await client.get(f"{GATEWAY_URL}/run/{run_id}")
            payload  = status_r.json()
        except Exception as exc:
            print(f"    {_warn('!')} Poll error: {exc}")
            continue

        s = payload.get("status", "unknown")

        if s in ("completed", "partial", "failed"):
            colour = _ok if s == "completed" else (_warn if s == "partial" else _err)
            print(f"    {colour('→')} Terminal: {colour(s)}")
            payload["run_id"] = run_id
            return payload

        elapsed = (tick + 1) * 5
        print(f"    {_dim(f'[{elapsed:>3}s]')} {s}…", end="\r", flush=True)

    print()
    return {"status": "timeout", "run_id": run_id}


def _cyan(s): return f"{_CYAN}{s}{_R}"

# ─── Main ──────────────────────────────────────────────────────────────────────

async def main(start_idx: int = 0, dry_run: bool = False) -> None:
    with open(PROMPTS_FILE) as f:
        prompts: list[str] = json.load(f)

    total   = len(prompts)
    subset  = prompts[start_idx:]
    started = datetime.now(timezone.utc)

    print(f"\n{_bold(_CYAN + '=' * 68 + _R)}")
    print(f"{_bold('  AgentMesh Regression Runner'):}")
    print(f"{_bold(_CYAN + '=' * 68 + _R)}")
    print(f"  Prompts file : {PROMPTS_FILE}")
    print(f"  Gateway URL  : {GATEWAY_URL}")
    print(f"  Queries      : {len(subset)} of {total}"
          + (f"  (resuming from #{start_idx + 1})" if start_idx else ""))
    print(f"  Est. time    : ~{len(subset) * 3} – {len(subset) * 3 + 5} min\n")

    if dry_run:
        print(_bold("DRY RUN — queries that would be submitted:"))
        for i, q in enumerate(subset, start=start_idx + 1):
            print(f"  [{i:>2}] {q}")
        print()
        return

    results: list[dict] = []

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Verify gateway is reachable before starting
        try:
            health = await client.get(f"{GATEWAY_URL}/health")
            health.raise_for_status()
        except Exception as exc:
            print(_err(f"✗ Gateway unreachable at {GATEWAY_URL}: {exc}"))
            print(_dim("  Make sure Docker is running: docker compose up -d"))
            sys.exit(1)
        print(f"  {_ok('OK')} Gateway healthy\n")

        for i, query in enumerate(subset, start=start_idx):
            label = f"[{i + 1}/{total}]"
            print(f"{_bold(label)} {query[:65]}")
            t0 = time.monotonic()

            try:
                result = await run_single_query(client, query)

                # ── Retry once on failure / timeout / empty narrative ────────
                bad_status = result.get("status") in ("failed", "timeout", "error")
                empty_report = not (result.get("final_report") or {}).get("narrative", "").strip()
                if bad_status or empty_report:
                    reason = "timeout" if result.get("status") == "timeout" else "failed/empty"
                    print(f"    {_warn(f'! {reason} — retrying in 20 s...')}")
                    await asyncio.sleep(20)
                    result = await run_single_query(client, query)

                elapsed = round(time.monotonic() - t0)
                results.append({
                    "query":   query,
                    "status":  result.get("status", "unknown"),
                    "run_id":  result.get("run_id", ""),
                    "elapsed_s": elapsed,
                })
                print(f"    {_dim(f'Done in {elapsed}s')}")
            except Exception as exc:
                elapsed = round(time.monotonic() - t0)
                print(f"    {_err(f'ERROR: {exc}')}")
                results.append({
                    "query":     query,
                    "status":    "error",
                    "error":     str(exc),
                    "elapsed_s": elapsed,
                })

            # Pause between queries to respect rate limits
            if i < start_idx + len(subset) - 1:
                print(f"    {_dim('Waiting 10 s before next query…')}\n")
                await asyncio.sleep(10)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = round((datetime.now(timezone.utc) - started).total_seconds() / 60, 1)
    passed   = sum(1 for r in results if r["status"] in ("completed", "partial"))
    failed   = sum(1 for r in results if r["status"] == "failed")
    errors   = sum(1 for r in results if r["status"] == "error")
    timeouts = sum(1 for r in results if r["status"] == "timeout")

    print(f"\n{_bold(_CYAN + '=' * 68 + _R)}")
    print(f"{_bold('  Regression Test Complete')}")
    print(f"{_bold(_CYAN + '=' * 68 + _R)}")
    print(f"  Total time   : {total_elapsed} min")
    print(f"  {_ok('Passed')}       : {passed}/{len(results)}")
    if failed:   print(f"  {_err('Failed')}       : {failed}")
    if errors:   print(f"  {_err('Errors')}       : {errors}")
    if timeouts: print(f"  {_warn('Timeouts')}     : {timeouts}")

    print(f"\n  Per-query results:")
    for r in results:
        s = r["status"]
        colour = _ok if s in ("completed","partial") else (_warn if s=="timeout" else _err)
        q_short = r["query"][:55] + ("…" if len(r["query"]) > 55 else "")
        rid = r.get("run_id", "")[:8]
        print(f"    {colour('*')} {q_short:<57} {colour(s):<12} {_dim(rid)}")

    print(f"\n  {_bold('View scores:')}")
    print(f"  {_CYAN}python evals/dashboard.py{_R}\n")

    # Write results JSON for CI / further analysis
    out_path = os.path.join(os.path.dirname(__file__), "regression_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_at": started.isoformat(),
            "total_elapsed_min": total_elapsed,
            "passed": passed,
            "total": len(results),
            "results": results,
        }, f, indent=2)
    print(f"  {_dim('Results saved to:')} {out_path}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all hard prompts end-to-end and collect eval scores."
    )
    parser.add_argument(
        "--start", type=int, default=0, metavar="N",
        help="Resume from prompt index N (0-based, default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts that would be submitted without running them",
    )
    args = parser.parse_args()
    asyncio.run(main(start_idx=args.start, dry_run=args.dry_run))
