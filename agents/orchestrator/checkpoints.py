"""
Human-in-the-loop checkpoint management for the AgentMesh orchestrator.

LangGraph's interrupt() mechanism suspends the graph and raises a
GraphInterrupt inside the graph runner.  The runner stores the interrupt
payload here (keyed by run_id) so the gateway can fetch it and display
the sub-tasks to the user.  When the user approves/edits/rejects, the
gateway calls resume_checkpoint() which unblocks the graph via
graph.invoke(Command(resume=payload), config=...).

This module owns two things:
  1. CheckpointStore  — in-memory store mapping run_id → pending interrupt
  2. CheckpointRouter — FastAPI router exposing /checkpoint/{run_id} GET + POST
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Data models for the checkpoint HTTP API
# ---------------------------------------------------------------------------


class CheckpointPayload(BaseModel):
    """Stored when LangGraph interrupt() fires; fetched by the gateway."""

    run_id: str
    trace_id: str
    sub_tasks: list[dict[str, Any]]
    message: str
    created_at: datetime = Field(default_factory=_utcnow)


class ApprovalRequest(BaseModel):
    """
    POST body sent by the frontend (via gateway) to resume the graph.

    Three valid shapes:
      {"approved": true}                              — approve as-is
      {"approved": true, "sub_tasks": [...edited]}    — approve with edits
      {"approved": false, "feedback": "..."}          — reject; re-decompose
    """

    approved: bool
    sub_tasks: list[dict[str, Any]] | None = Field(
        default=None,
        description="Edited sub-tasks. Only present when approved=true and user made changes.",
    )
    feedback: str | None = Field(
        default=None,
        description="Rejection reason. Only meaningful when approved=false.",
    )


class CheckpointStatus(BaseModel):
    """Returned by GET /checkpoint/{run_id}."""

    run_id: str
    pending: bool
    payload: CheckpointPayload | None = None


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


class CheckpointStore:
    """
    In-memory store for pending human-in-the-loop checkpoints.

    Maps run_id → (CheckpointPayload, asyncio.Event).
    The Event is set when the user submits an approval, unblocking the
    graph runner that is awaiting resume.

    Thread-safety: asyncio.Lock guards all mutations so concurrent HTTP
    requests from the gateway cannot corrupt state.
    """

    def __init__(self) -> None:
        # run_id → (payload, ready_event, resume_payload)
        self._store: dict[str, tuple[CheckpointPayload, asyncio.Event, dict[str, Any] | None]] = {}
        self._lock = asyncio.Lock()

    async def register(self, payload: CheckpointPayload) -> asyncio.Event:
        """
        Called by the graph runner when interrupt() fires.

        Stores the interrupt payload and returns an Event the runner awaits.
        The runner is unblocked when the user POSTs an approval.
        """
        event = asyncio.Event()
        async with self._lock:
            self._store[payload.run_id] = (payload, event, None)
        log.info(
            "checkpoint.registered",
            run_id=payload.run_id,
            trace_id=payload.trace_id,
            sub_task_count=len(payload.sub_tasks),
        )
        return event

    async def get(self, run_id: str) -> CheckpointPayload | None:
        """Return the pending payload for a run, or None if not found."""
        async with self._lock:
            entry = self._store.get(run_id)
        return entry[0] if entry else None

    async def resolve(self, run_id: str, resume: dict[str, Any]) -> None:
        """
        Called when the user submits their approval decision.

        Stores the resume payload and sets the Event so the graph runner
        can call graph.invoke(Command(resume=resume), ...).
        """
        async with self._lock:
            entry = self._store.get(run_id)
            if entry is None:
                raise KeyError(f"No pending checkpoint for run_id={run_id}")
            payload, event, _ = entry
            self._store[run_id] = (payload, event, resume)
            event.set()
        log.info(
            "checkpoint.resolved",
            run_id=run_id,
            approved=resume.get("approved"),
        )

    async def get_resume_payload(self, run_id: str) -> dict[str, Any] | None:
        """Return the resume payload after resolve() has been called."""
        async with self._lock:
            entry = self._store.get(run_id)
        return entry[2] if entry else None

    async def delete(self, run_id: str) -> None:
        """Remove a checkpoint entry once the graph has resumed."""
        async with self._lock:
            self._store.pop(run_id, None)
        log.debug("checkpoint.deleted", run_id=run_id)

    def pending_run_ids(self) -> list[str]:
        """Return all run_ids currently awaiting human input."""
        return list(self._store.keys())


# ---------------------------------------------------------------------------
# FastAPI router — mounted by a2a_server.py
# ---------------------------------------------------------------------------

# Module-level singleton shared between the router and the graph runner.
store = CheckpointStore()


def create_checkpoint_router() -> APIRouter:
    """
    Return a FastAPI router with two endpoints:

      GET  /checkpoint/{run_id}  — poll for a pending checkpoint (frontend polls this)
      POST /checkpoint/{run_id}  — submit the human decision to resume the graph
    """
    router = APIRouter(prefix="/checkpoint", tags=["checkpoint"])

    @router.get("/{run_id}", response_model=CheckpointStatus)
    async def get_checkpoint(run_id: str) -> CheckpointStatus:
        """
        Return the pending checkpoint payload for the given run.

        The frontend polls this endpoint after submitting a query.
        When pending=true, the frontend renders the sub-task approval UI.
        """
        payload = await store.get(run_id)
        if payload is None:
            return CheckpointStatus(run_id=run_id, pending=False)
        return CheckpointStatus(run_id=run_id, pending=True, payload=payload)

    @router.post("/{run_id}")
    async def post_checkpoint(run_id: str, body: ApprovalRequest) -> dict[str, str]:
        """
        Submit the human approval/rejection to resume the graph.

        The graph runner is blocked on an asyncio.Event; this endpoint sets
        it so the runner can call graph.invoke(Command(resume=...)).
        """
        resume: dict[str, Any] = {"approved": body.approved}

        if body.approved and body.sub_tasks is not None:
            resume["sub_tasks"] = body.sub_tasks

        if not body.approved and body.feedback:
            resume["feedback"] = body.feedback

        try:
            await store.resolve(run_id, resume)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"No pending checkpoint for run_id={run_id}. It may have already been resolved.",
            )

        log.info(
            "checkpoint.http_resolved",
            run_id=run_id,
            approved=body.approved,
            has_edits=body.sub_tasks is not None,
            has_feedback=bool(body.feedback),
        )
        return {"status": "resumed", "run_id": run_id}

    @router.get("/")
    async def list_pending() -> dict[str, list[str]]:
        """List all run_ids currently awaiting human approval (useful for debugging)."""
        return {"pending_run_ids": store.pending_run_ids()}

    return router
