"""
Orchestrator A2A server and graph runner.

This module is uvicorn's entry point for the orchestrator service.  It does
three things:

  1. Exposes POST /a2a  — receives new research queries from the gateway
     and starts a new graph run in the background.

  2. Exposes GET/POST /checkpoint/{run_id}  — lets the gateway poll for
     the human-in-the-loop interrupt and submit the user's approval.

  3. Exposes GET /run/{run_id}  — lets the gateway poll for final results
     once the graph has completed.

Graph runner design
-------------------
Starting the graph is a two-phase operation because of the human checkpoint:

  Phase A:  gateway calls POST /a2a
            → graph.ainvoke() runs decompose_query then hits interrupt()
            → runner stores the interrupt payload in CheckpointStore
            → POST /a2a returns immediately with {run_id, status: "awaiting_human"}

  Phase B:  frontend polls GET /checkpoint/{run_id} until pending=true
            → user approves/edits/rejects via POST /checkpoint/{run_id}
            → runner's asyncio.Event is set
            → runner calls graph.ainvoke(Command(resume=...)) to continue
            → graph runs dispatch_tasks → merge_results → END
            → final state is stored in RunStore

Gateway polls GET /run/{run_id} to retrieve the completed result.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel

from agents.orchestrator.checkpoints import (
    CheckpointPayload,
    create_checkpoint_router,
    store as checkpoint_store,
)
from agents.orchestrator.graph import graph
from shared.a2a_server import TaskHandler, create_a2a_app
from shared.models import (
    AgentMessage,
    Framework,
    OrchestratorState,
    ReportStatus,
    TaskCard,
    _new_uuid,
)

log = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# In-memory run result store (keyed by run_id)
# ---------------------------------------------------------------------------


class RunStore:
    """Stores completed OrchestratorState objects keyed by run_id."""

    def __init__(self) -> None:
        self._store: dict[str, OrchestratorState] = {}
        self._lock = asyncio.Lock()

    async def save(self, state: OrchestratorState) -> None:
        async with self._lock:
            self._store[state.run_id] = state

    async def get(self, run_id: str) -> OrchestratorState | None:
        async with self._lock:
            return self._store.get(run_id)


run_store = RunStore()


# ---------------------------------------------------------------------------
# Graph runner
# ---------------------------------------------------------------------------


async def _run_graph(initial_state: OrchestratorState) -> None:
    """
    Background task that drives one full orchestration run.

    Phase A: run until the human_checkpoint interrupt fires.
    Phase B: wait for the user's approval, then resume to completion.
    """
    run_id = initial_state.run_id
    thread_config = {"configurable": {"thread_id": run_id}}
    bound_log = log.bind(run_id=run_id, trace_id=initial_state.trace_id)

    bound_log.info("graph_runner.start")

    # ------------------------------------------------------------------
    # Phase A — run up to the interrupt
    # ------------------------------------------------------------------
    try:
        async for event in graph.astream(
            initial_state.model_dump(mode="json"),
            config=thread_config,
            stream_mode="values",
        ):
            # Each streamed event is the full state after a node completes.
            node_status = event.get("status")
            bound_log.debug("graph_runner.stream_event", status=node_status)

    except GraphInterrupt as interrupt_exc:
        # interrupt() fired inside human_checkpoint.
        interrupt_value: dict[str, Any] = interrupt_exc.args[0] if interrupt_exc.args else {}
        bound_log.info("graph_runner.interrupt_fired", interrupt_keys=list(interrupt_value.keys()))

        payload = CheckpointPayload(
            run_id=run_id,
            trace_id=initial_state.trace_id,
            sub_tasks=interrupt_value.get("sub_tasks", []),
            message=interrupt_value.get("message", "Awaiting human approval"),
        )
        ready_event = await checkpoint_store.register(payload)

        # Block until the user submits their decision via POST /checkpoint/{run_id}.
        bound_log.info("graph_runner.awaiting_human")
        await ready_event.wait()
        bound_log.info("graph_runner.human_responded")

    except Exception as exc:
        bound_log.error("graph_runner.phase_a_error", error=str(exc))
        failed_state = initial_state.model_copy(
            update={"status": ReportStatus.FAILED, "updated_at": _utcnow()}
        )
        await run_store.save(failed_state)
        return

    # ------------------------------------------------------------------
    # Phase B — resume after human approval
    # ------------------------------------------------------------------
    resume_payload = await checkpoint_store.get_resume_payload(run_id)
    if resume_payload is None:
        bound_log.error("graph_runner.no_resume_payload")
        return

    await checkpoint_store.delete(run_id)

    bound_log.info("graph_runner.resuming", approved=resume_payload.get("approved"))

    try:
        final_state_dict: dict[str, Any] = {}
        async for event in graph.astream(
            Command(resume=resume_payload),
            config=thread_config,
            stream_mode="values",
        ):
            final_state_dict = event  # keep the last event (full state)
            bound_log.debug("graph_runner.resume_event", status=event.get("status"))

        final_state = OrchestratorState.model_validate(final_state_dict)
        await run_store.save(final_state)
        bound_log.info(
            "graph_runner.complete",
            status=final_state.status,
            has_report=final_state.final_report is not None,
            has_partial=final_state.partial_result is not None,
        )

    except Exception as exc:
        bound_log.error("graph_runner.phase_b_error", error=str(exc))
        # Attempt to salvage any partial state from the run store.
        existing = await run_store.get(run_id)
        if existing is None:
            failed_state = initial_state.model_copy(
                update={"status": ReportStatus.FAILED, "updated_at": _utcnow()}
            )
            await run_store.save(failed_state)


# ---------------------------------------------------------------------------
# A2A TaskHandler — receives queries from the gateway
# ---------------------------------------------------------------------------


class OrchestratorHandler(TaskHandler):
    """
    A2A handler for the orchestrator service.

    The gateway sends a TaskCard whose payload contains {"query": "..."}.
    This handler starts the graph runner as a background task and returns
    immediately with the run_id so the gateway can poll for results.
    """

    async def handle_task(self, task_card: TaskCard, trace_id: str) -> AgentMessage:
        """Start a new orchestration run and return the run_id immediately."""
        query: str = task_card.payload.get("query", "").strip()
        if not query:
            raise ValueError("TaskCard payload must include a non-empty 'query' field.")

        initial_state = OrchestratorState(
            query=query,
            trace_id=trace_id,
        )
        run_id = initial_state.run_id

        log.info(
            "orchestrator.handle_task",
            run_id=run_id,
            trace_id=trace_id,
            query_preview=query[:100],
        )

        # Fire-and-forget: the graph runs in the background.
        asyncio.create_task(_run_graph(initial_state))

        return AgentMessage(
            trace_id=trace_id,
            protocol_used="a2a",
            sender_framework=Framework.LANGGRAPH,
            message={
                "run_id": run_id,
                "status": ReportStatus.AWAITING_HUMAN.value,
                "message": "Graph started. Poll /checkpoint/{run_id} for the sub-task approval UI.",
            },
        )


# ---------------------------------------------------------------------------
# FastAPI app assembly
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build the orchestrator FastAPI app with all routes mounted."""
    base_app = create_a2a_app(OrchestratorHandler(), title="Orchestrator Agent")

    # Mount checkpoint management routes.
    base_app.include_router(create_checkpoint_router())

    # ---- GET /run/{run_id} — poll for final results --------------------
    @base_app.get("/run/{run_id}")
    async def get_run_result(run_id: str) -> dict[str, Any]:
        """
        Return the current state of a graph run.

        Poll this endpoint after the checkpoint has been resolved.
        Returns status + the final_report or partial_result once complete.
        """
        state = await run_store.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

        result: dict[str, Any] = {
            "run_id": run_id,
            "status": state.status.value,
            "query": state.query,
            "sub_task_count": len(state.sub_tasks),
        }

        if state.final_report is not None:
            result["final_report"] = state.final_report.model_dump(mode="json")

        if state.partial_result is not None:
            result["partial_result"] = state.partial_result.model_dump(mode="json")

        return result

    # ---- GET /runs — list all runs (debug) -----------------------------
    @base_app.get("/runs")
    async def list_runs() -> dict[str, list[str]]:
        return {"run_ids": list(run_store._store.keys())}

    return base_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "a2a_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001)),
        reload=False,
    )
