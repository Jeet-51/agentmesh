"""
AgentMesh API gateway.

Single entry point between the React frontend and the agent network.
All frontend calls go through here; nothing in the browser touches agent
services directly.

Endpoints
---------
POST /run                         Accept query, start orchestration, return run_id immediately.
GET  /run/{run_id}                Poll for current status + result.
GET  /run/{run_id}/stream         SSE stream of live agent activity events.
GET  /checkpoint/{run_id}         Fetch pending sub-tasks awaiting human approval.
POST /checkpoint/{run_id}/approve Submit approval / edits / rejection to resume pipeline.
GET  /health                      Gateway liveness check.

Internal communication
----------------------
- Orchestrator A2A endpoint (POST /a2a): reached via A2AClient for starting runs.
- Orchestrator REST endpoints (GET /run, GET|POST /checkpoint): reached via httpx directly,
  because these are plain REST, not A2A protocol calls.

SSE event model
---------------
A background polling task is started per run.  It polls the orchestrator's
GET /run/{run_id} every POLL_INTERVAL seconds, detects status transitions, and
pushes typed AgentActivityEvent objects into per-run asyncio.Queue instances.
Every connected SSE client has its own Queue; events fan-out to all listeners.

Event types emitted (in order):
  started          → run accepted by orchestrator
  decomposed       → sub-tasks created, awaiting human
  awaiting_human   → same tick; sub-tasks surfaced to frontend
  dispatched       → human approved; research agent called
  research_done    → findings received from CrewAI
  synthesis_done   → report produced by ADK agent
  merged           → orchestrator merged final report (terminal - success)
  partial          → at least one agent failed; partial report available (terminal)
  failed           → catastrophic failure (terminal)
  ping             → keepalive heartbeat (30 s interval)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from shared.a2a_client import A2AClient, A2AClientError
from shared.models import Framework, TaskCard

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        (
            structlog.dev.ConsoleRenderer()
            if os.environ.get("LOG_FORMAT", "").lower() != "json"
            else structlog.processors.JSONRenderer()
        ),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

ORCHESTRATOR_URL: str = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8001")
POLL_INTERVAL: float = float(os.environ.get("GATEWAY_POLL_INTERVAL", "2.5"))
SSE_KEEPALIVE: float = float(os.environ.get("GATEWAY_SSE_KEEPALIVE", "25.0"))
CORS_ORIGINS: list[str] = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# ---------------------------------------------------------------------------
# Pydantic models — request / response contracts
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Payload for POST /run."""

    query: str = Field(..., min_length=3, max_length=2000, description="Research query from user.")


class RunCreateResponse(BaseModel):
    """Immediate response from POST /run — before any agent work completes."""

    run_id: str
    trace_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    """Polling response from GET /run/{run_id}."""

    run_id: str
    trace_id: str
    status: str
    query: str
    sub_tasks: list[dict[str, Any]] | None = None
    findings_count: int = 0
    final_report: dict[str, Any] | None = None
    partial_result: dict[str, Any] | None = None
    events_emitted: int = 0


class CheckpointResponse(BaseModel):
    """Response from GET /checkpoint/{run_id}."""

    run_id: str
    pending: bool
    sub_tasks: list[dict[str, Any]] | None = None
    message: str | None = None


class ApprovalRequest(BaseModel):
    """Payload for POST /checkpoint/{run_id}/approve."""

    approved: bool = Field(..., description="True to approve/approve-with-edits; False to reject.")
    sub_tasks: list[dict[str, Any]] | None = Field(
        default=None,
        description="Edited sub-tasks. Only when approved=True and user changed something.",
    )
    feedback: str | None = Field(
        default=None,
        description="Rejection reason. Only when approved=False.",
    )


class AgentActivityEvent(BaseModel):
    """One SSE event pushed to the frontend."""

    event_type: str
    run_id: str
    trace_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# In-memory run state store
# ---------------------------------------------------------------------------


class _GatewayRun:
    """Per-run state managed by the gateway."""

    def __init__(self, run_id: str, trace_id: str, query: str) -> None:
        self.run_id = run_id
        self.trace_id = trace_id
        self.query = query
        self.status: str = "started"
        self.orchestrator_state: dict[str, Any] = {}
        self.events_emitted: int = 0
        # Each SSE connection gets its own Queue entry.
        self._queues: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        async with self._lock:
            self._queues.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    async def emit(self, event: AgentActivityEvent) -> None:
        """Fan-out one event to all connected SSE queues."""
        self.events_emitted += 1
        payload = event.model_dump(mode="json")
        async with self._lock:
            for q in list(self._queues):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    log.warning(
                        "gateway.sse.queue_full",
                        run_id=self.run_id,
                        event_type=event.event_type,
                    )


class GatewayStore:
    """Thread-safe in-memory store for all active gateway runs."""

    def __init__(self) -> None:
        self._runs: dict[str, _GatewayRun] = {}
        self._lock = asyncio.Lock()

    async def create(self, run_id: str, trace_id: str, query: str) -> _GatewayRun:
        run = _GatewayRun(run_id=run_id, trace_id=trace_id, query=query)
        async with self._lock:
            self._runs[run_id] = run
        return run

    async def get(self, run_id: str) -> _GatewayRun | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def all_run_ids(self) -> list[str]:
        async with self._lock:
            return list(self._runs.keys())


store = GatewayStore()

# ---------------------------------------------------------------------------
# Background orchestrator poller
# ---------------------------------------------------------------------------

# Terminal statuses — poller stops when it sees one of these.
_TERMINAL = {"completed", "partial", "failed"}

# Map orchestrator status → gateway SSE event_type.
_STATUS_EVENT: dict[str, str] = {
    "pending":         "started",
    "awaiting_human":  "awaiting_human",
    "running":         "dispatched",
    "completed":       "merged",
    "partial":         "partial",
    "failed":          "failed",
}


async def _poll_orchestrator(run: _GatewayRun) -> None:
    """
    Long-running background task per run.

    Polls GET /run/{run_id} on the orchestrator every POLL_INTERVAL seconds.
    Detects status transitions and emits AgentActivityEvent to all SSE clients.
    Stops when a terminal status is reached or the orchestrator returns 404.
    """
    url = f"{ORCHESTRATOR_URL.rstrip('/')}/run/{run.run_id}"
    prev_status: str | None = None
    prev_findings_count: int = 0
    bound_log = log.bind(run_id=run.run_id, trace_id=run.trace_id)
    bound_log.info("gateway.poller.start")

    async with httpx.AsyncClient(timeout=10.0) as http:
        while True:
            await asyncio.sleep(POLL_INTERVAL)

            try:
                resp = await http.get(url)
            except httpx.RequestError as exc:
                bound_log.warning("gateway.poller.http_error", error=str(exc))
                continue

            if resp.status_code == 404:
                bound_log.warning("gateway.poller.run_not_found")
                await _emit(run, "failed", {"reason": "Run not found on orchestrator"})
                break

            if resp.status_code != 200:
                bound_log.warning("gateway.poller.unexpected_status", code=resp.status_code)
                continue

            data: dict[str, Any] = resp.json()
            current_status: str = data.get("status", "unknown")
            run.orchestrator_state = data
            run.status = current_status

            # Emit on first observation (even if same as initial).
            if prev_status is None:
                await _emit(run, _STATUS_EVENT.get(current_status, current_status), data)

            # Emit on status transition.
            elif current_status != prev_status:
                bound_log.info(
                    "gateway.poller.transition",
                    from_status=prev_status,
                    to_status=current_status,
                )
                # Emit decomposed event just before awaiting_human so frontend
                # has sub-task data before the checkpoint prompt appears.
                if current_status == "awaiting_human":
                    await _emit(run, "decomposed", {
                        "sub_tasks": data.get("sub_tasks", []),
                        "sub_task_count": data.get("sub_task_count", 0),
                    })
                await _emit(run, _STATUS_EVENT.get(current_status, current_status), data)

            # Detect research_done: running + findings present (even without status change).
            findings_count: int = len(data.get("findings", []) or [])
            if (
                current_status == "running"
                and findings_count > 0
                and findings_count != prev_findings_count
            ):
                await _emit(run, "research_done", {
                    "findings_count": findings_count,
                    "message": f"{findings_count} research finding(s) received.",
                })

            # Detect synthesis_done: final_report just appeared.
            if data.get("final_report") and not prev_status == "completed":
                await _emit(run, "synthesis_done", {
                    "report_id": (data.get("final_report") or {}).get("report_id"),
                    "citation_count": len((data.get("final_report") or {}).get("citations", [])),
                })

            prev_status = current_status
            prev_findings_count = findings_count

            if current_status in _TERMINAL:
                bound_log.info("gateway.poller.terminal", status=current_status)
                break

    bound_log.info("gateway.poller.done")


async def _emit(run: _GatewayRun, event_type: str, data: dict[str, Any]) -> None:
    """Helper to build and fan-out one AgentActivityEvent."""
    event = AgentActivityEvent(
        event_type=event_type,
        run_id=run.run_id,
        trace_id=run.trace_id,
        data=data,
    )
    log.debug("gateway.emit", run_id=run.run_id, event_type=event_type)
    await run.emit(event)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("gateway.startup", orchestrator_url=ORCHESTRATOR_URL)
    yield
    log.info("gateway.shutdown")


app = FastAPI(
    title="AgentMesh Gateway",
    version="1.0.0",
    description="Single entry point for the AgentMesh multi-agent research system.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------


@app.post("/run", response_model=RunCreateResponse, status_code=202)
async def create_run(body: RunRequest, request: Request) -> RunCreateResponse:
    """
    Accept a research query and hand it to the orchestrator.

    Returns immediately with a run_id.  The frontend then polls GET /run/{run_id}
    or connects to GET /run/{run_id}/stream for live updates.

    Returns 503 if the orchestrator is unreachable.
    """
    trace_id = str(uuid.uuid4())
    bound_log = log.bind(trace_id=trace_id, query_preview=body.query[:80])
    bound_log.info("gateway.create_run.start")

    # Build A2A TaskCard for the orchestrator.
    task_card = TaskCard(
        sender_framework=Framework.GATEWAY,
        receiver_framework=Framework.LANGGRAPH,
        context_id=trace_id,
        payload={"query": body.query},
    )

    try:
        async with A2AClient(sender_framework=Framework.GATEWAY) as client:
            # Check orchestrator health first for a friendlier error message.
            healthy = await client.health_check(ORCHESTRATOR_URL)
            if not healthy:
                raise A2AClientError("Orchestrator health check failed", agent_url=ORCHESTRATOR_URL)

            response = await client.send_task(
                agent_url=ORCHESTRATOR_URL,
                task_card=task_card,
                trace_id=trace_id,
            )
    except A2AClientError as exc:
        bound_log.error("gateway.create_run.orchestrator_unreachable", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=(
                f"Orchestrator is unreachable at {ORCHESTRATOR_URL}. "
                "Ensure the orchestrator service is running. "
                f"Detail: {exc}"
            ),
        )

    # Extract run_id from the orchestrator's response message.
    run_id: str = response.message.get("run_id", str(uuid.uuid4()))
    orch_status: str = response.message.get("status", "started")

    bound_log.info("gateway.create_run.accepted", run_id=run_id, orch_status=orch_status)

    # Register the run locally and start the background poller.
    run = await store.create(run_id=run_id, trace_id=trace_id, query=body.query)
    asyncio.create_task(
        _poll_orchestrator(run),
        name=f"poller-{run_id}",
    )

    return RunCreateResponse(
        run_id=run_id,
        trace_id=trace_id,
        status=orch_status,
        message="Run started. Poll /run/{run_id} or stream /run/{run_id}/stream.",
    )


# ---------------------------------------------------------------------------
# GET /run/{run_id}
# ---------------------------------------------------------------------------


@app.get("/run/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """
    Return the current status of a run.

    The frontend polls this every few seconds until status is terminal.
    Also proxies the orchestrator's state so the frontend has the full picture.
    """
    run = await store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    orch = run.orchestrator_state
    findings_count = len(orch.get("findings", []) or [])

    return RunStatusResponse(
        run_id=run_id,
        trace_id=run.trace_id,
        status=run.status,
        query=run.query,
        sub_tasks=orch.get("sub_tasks"),
        findings_count=findings_count,
        final_report=orch.get("final_report"),
        partial_result=orch.get("partial_result"),
        events_emitted=run.events_emitted,
    )


# ---------------------------------------------------------------------------
# GET /run/{run_id}/stream  — SSE
# ---------------------------------------------------------------------------


@app.get("/run/{run_id}/stream")
async def stream_run(run_id: str, request: Request) -> EventSourceResponse:
    """
    Stream live agent activity events for a run via Server-Sent Events.

    The frontend connects once and receives push events as the agents progress.
    The stream closes automatically on a terminal event (merged/partial/failed)
    or when the client disconnects.

    Event schema (JSON-encoded):
      { "event_type": str, "run_id": str, "trace_id": str,
        "timestamp": ISO8601, "data": {...} }
    """
    run = await store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    queue = await run.subscribe()
    log.info("gateway.sse.client_connected", run_id=run_id)

    async def generator() -> AsyncGenerator[dict[str, str], None]:
        try:
            while True:
                # Respect client disconnection.
                if await request.is_disconnected():
                    log.info("gateway.sse.client_disconnected", run_id=run_id)
                    break

                try:
                    payload: dict[str, Any] = await asyncio.wait_for(
                        queue.get(), timeout=SSE_KEEPALIVE
                    )
                    yield {
                        "event": payload["event_type"],
                        "data": json.dumps(payload),
                        "id": f"{run_id}-{payload.get('timestamp', '')}",
                    }
                    # Close stream on terminal events.
                    if payload["event_type"] in ("merged", "partial", "failed"):
                        log.info(
                            "gateway.sse.terminal_event",
                            run_id=run_id,
                            event_type=payload["event_type"],
                        )
                        break

                except asyncio.TimeoutError:
                    # Send keepalive ping so proxies don't close the connection.
                    yield {
                        "event": "ping",
                        "data": json.dumps({"run_id": run_id, "ts": datetime.now(timezone.utc).isoformat()}),
                    }
        finally:
            await run.unsubscribe(queue)
            log.info("gateway.sse.stream_closed", run_id=run_id)

    return EventSourceResponse(generator())


# ---------------------------------------------------------------------------
# GET /checkpoint/{run_id}
# ---------------------------------------------------------------------------


@app.get("/checkpoint/{run_id}", response_model=CheckpointResponse)
async def get_checkpoint(run_id: str) -> CheckpointResponse:
    """
    Fetch the pending sub-tasks awaiting human approval for this run.

    The frontend polls this after receiving an 'awaiting_human' SSE event.
    Proxies directly to the orchestrator's /checkpoint/{run_id} endpoint.
    """
    url = f"{ORCHESTRATOR_URL.rstrip('/')}/checkpoint/{run_id}"
    log.info("gateway.get_checkpoint", run_id=run_id)

    try:
        async with httpx.AsyncClient(timeout=8.0) as http:
            resp = await http.get(url)
    except httpx.RequestError as exc:
        log.error("gateway.get_checkpoint.http_error", run_id=run_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Orchestrator unreachable: {exc}",
        )

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"No checkpoint found for run {run_id}.")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Orchestrator returned {resp.status_code}.",
        )

    data = resp.json()
    return CheckpointResponse(
        run_id=run_id,
        pending=data.get("pending", False),
        sub_tasks=(data.get("payload") or {}).get("sub_tasks"),
        message=(data.get("payload") or {}).get("message"),
    )


# ---------------------------------------------------------------------------
# POST /checkpoint/{run_id}/approve
# ---------------------------------------------------------------------------


@app.post("/checkpoint/{run_id}/approve")
async def approve_checkpoint(run_id: str, body: ApprovalRequest) -> dict[str, Any]:
    """
    Submit the human's decision (approve / approve-with-edits / reject).

    Proxies to the orchestrator's POST /checkpoint/{run_id}, which resumes
    the LangGraph graph via Command(resume=payload).

    After this returns, the pipeline resumes automatically.  The frontend
    should continue listening to the SSE stream or polling /run/{run_id}.
    """
    url = f"{ORCHESTRATOR_URL.rstrip('/')}/checkpoint/{run_id}"
    bound_log = log.bind(run_id=run_id, approved=body.approved)
    bound_log.info("gateway.approve_checkpoint")

    # Mark locally for event emission.
    run = await store.get(run_id)

    # Build the request body for the orchestrator's ApprovalRequest model.
    orch_body: dict[str, Any] = {"approved": body.approved}
    if body.approved and body.sub_tasks is not None:
        orch_body["sub_tasks"] = body.sub_tasks
    if not body.approved and body.feedback:
        orch_body["feedback"] = body.feedback

    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.post(url, json=orch_body)
    except httpx.RequestError as exc:
        bound_log.error("gateway.approve_checkpoint.http_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Orchestrator unreachable: {exc}")

    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"No pending checkpoint for run {run_id}. It may have already been resolved.",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Orchestrator returned {resp.status_code}: {resp.text[:200]}",
        )

    # Emit a human_responded event so the SSE stream reflects the decision.
    if run is not None:
        await _emit(
            run,
            "human_responded",
            {
                "approved": body.approved,
                "has_edits": body.sub_tasks is not None,
                "feedback": body.feedback,
            },
        )

    bound_log.info("gateway.approve_checkpoint.ok")
    return {
        "status": "resumed",
        "run_id": run_id,
        "approved": body.approved,
        "message": (
            "Pipeline resumed. Sub-tasks will be dispatched to agents."
            if body.approved
            else "Rejected. Orchestrator will re-decompose the query."
        ),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Gateway liveness check.

    Also reports orchestrator reachability so ops can diagnose connectivity
    issues without looking at logs.
    """
    orch_ok: bool = False
    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            r = await http.get(f"{ORCHESTRATOR_URL.rstrip('/')}/health")
            orch_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "agentmesh-gateway",
        "orchestrator_reachable": orch_ok,
        "orchestrator_url": ORCHESTRATOR_URL,
        "active_runs": len(await store.all_run_ids()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
