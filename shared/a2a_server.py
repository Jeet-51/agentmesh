"""
Reusable A2A server-side building blocks.

Each agent's a2a_server.py does three things:
  1. Subclass TaskHandler and implement handle_task()
  2. Call create_a2a_app(handler, title) to get a FastAPI app
  3. Uvicorn runs that app

The router handles HTTP plumbing, task-state bookkeeping (in-memory),
and structured logging of every inbound hop.  Agents only write domain logic.

Example (agents/research/a2a_server.py):

    from shared.a2a_server import TaskHandler, create_a2a_app
    from shared.models import AgentMessage, TaskCard

    class ResearchHandler(TaskHandler):
        async def handle_task(self, task_card: TaskCard, trace_id: str) -> AgentMessage:
            findings = await run_crew(task_card)
            return AgentMessage(
                trace_id=trace_id,
                protocol_used="a2a",
                sender_framework=Framework.CREWAI,
                message=findings.model_dump(mode="json"),
            )

    app = create_a2a_app(ResearchHandler(), title="Research Agent")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from shared.a2a_types import A2AError, A2AErrorBody, A2ARequest, A2AResult, TaskState
from shared.models import AgentMessage, TaskCard, _new_uuid

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# TaskHandler protocol — agents implement this
# ---------------------------------------------------------------------------


class TaskHandler(ABC):
    """
    Abstract base for agent-specific task logic.

    Subclass this in each agent's a2a_server.py and implement handle_task.
    The router calls handle_task after unpacking and validating the wire format.
    """

    @abstractmethod
    async def handle_task(self, task_card: TaskCard, trace_id: str) -> AgentMessage:
        """
        Execute the task described by task_card and return a result envelope.

        Implementations should:
        - Run the agent's domain logic (crew, graph, ADK agent, etc.)
        - Wrap the result in AgentMessage with the correct sender_framework
        - Not catch broad exceptions — let them propagate to the router
        """
        ...

    async def on_cancel(self, task_id: str) -> None:
        """Called when a tasks/cancel request arrives. Override if cancellable."""
        log.info("a2a.task.cancel_requested", task_id=task_id)


# ---------------------------------------------------------------------------
# In-memory task store (sufficient for portfolio; swap for Redis in prod)
# ---------------------------------------------------------------------------


class _TaskStore:
    """Thread-safe in-memory store for in-flight task states."""

    def __init__(self) -> None:
        self._store: dict[str, TaskState] = {}
        self._lock = asyncio.Lock()

    async def set(self, task_id: str, state: TaskState) -> None:
        async with self._lock:
            self._store[task_id] = state

    async def get(self, task_id: str) -> TaskState | None:
        async with self._lock:
            return self._store.get(task_id)

    async def delete(self, task_id: str) -> None:
        async with self._lock:
            self._store.pop(task_id, None)


# ---------------------------------------------------------------------------
# Router / app factory
# ---------------------------------------------------------------------------


def create_a2a_app(handler: TaskHandler, title: str = "A2A Agent") -> FastAPI:
    """
    Build a FastAPI application with A2A and health endpoints wired up.

    The returned app is passed directly to uvicorn.  Each agent calls this
    once at module level.
    """
    app = FastAPI(title=title, version="1.0.0")
    store = _TaskStore()

    # ------------------------------------------------------------------
    # POST /a2a  — main A2A dispatch endpoint
    # ------------------------------------------------------------------

    @app.post("/a2a")
    async def a2a_endpoint(request: Request) -> JSONResponse:
        raw: dict[str, Any] = await request.json()

        # Validate the JSON-RPC envelope.
        try:
            a2a_req = A2ARequest.model_validate(raw)
        except Exception as exc:
            return _error_response(id=raw.get("id", ""), code=-32600, message=f"Invalid request: {exc}")

        req_id = a2a_req.id
        method = a2a_req.method

        log.info("a2a.request.received", method=method, request_id=req_id, title=title)

        # ---- message/send -----------------------------------------------
        if method == "message/send":
            return await _handle_send(a2a_req, req_id, handler, store, title)

        # ---- tasks/get --------------------------------------------------
        if method == "tasks/get":
            task_id = a2a_req.params.get("task_id", "")
            return await _handle_get(task_id, req_id, store)

        # ---- tasks/cancel -----------------------------------------------
        if method == "tasks/cancel":
            task_id = a2a_req.params.get("task_id", "")
            return await _handle_cancel(task_id, req_id, handler, store)

        return _error_response(req_id, code=-32601, message=f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": title}

    return app


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------


async def _handle_send(
    a2a_req: A2ARequest,
    req_id: str,
    handler: TaskHandler,
    store: _TaskStore,
    title: str,
) -> JSONResponse:
    """Unpack the AgentMessage, run the handler, and return a result envelope."""
    message_data = a2a_req.params.get("message")
    if not message_data:
        return _error_response(req_id, code=-32600, message="params.message is required for message/send")

    try:
        envelope = AgentMessage.model_validate(message_data)
    except Exception as exc:
        return _error_response(req_id, code=-32600, message=f"Invalid AgentMessage: {exc}")

    try:
        task_card = TaskCard.model_validate(envelope.message)
    except Exception as exc:
        return _error_response(req_id, code=-32600, message=f"Invalid TaskCard in message: {exc}")

    trace_id = envelope.trace_id
    task_id = task_card.task_id

    # Mark as in-progress immediately (visible to polls before handler returns).
    await store.set(task_id, TaskState(task_id=task_id, status="in_progress"))

    log.info(
        "a2a.task.start",
        task_id=task_id,
        trace_id=trace_id,
        sender=envelope.sender_framework.value,
        service=title,
    )

    try:
        result: AgentMessage = await handler.handle_task(task_card, trace_id)
    except Exception as exc:
        log.error("a2a.task.failed", task_id=task_id, trace_id=trace_id, error=str(exc))
        await store.set(task_id, TaskState(task_id=task_id, status="failed", error=str(exc)))
        return _error_response(req_id, code=-32000, message=f"Agent error: {exc}")

    await store.set(task_id, TaskState(task_id=task_id, status="completed", result=result))

    log.info(
        "a2a.task.complete",
        task_id=task_id,
        trace_id=trace_id,
        service=title,
    )

    response = A2AResult(id=req_id, result=result)
    return JSONResponse(content=response.model_dump(mode="json"))


async def _handle_get(task_id: str, req_id: str, store: _TaskStore) -> JSONResponse:
    """Return current TaskState for a task_id."""
    if not task_id:
        return _error_response(req_id, code=-32600, message="params.task_id is required for tasks/get")

    state = await store.get(task_id)
    if state is None:
        return _error_response(req_id, code=-32000, message=f"Unknown task_id: {task_id}")

    # Wrap TaskState inside an AgentMessage so the client always gets AgentMessage back.
    wrapper = AgentMessage(
        trace_id=_new_uuid(),
        protocol_used="a2a",
        sender_framework="gateway",  # type: ignore[arg-type]  — sentinel for internal poll
        message={"task_state": state.model_dump(mode="json")},
    )
    response = A2AResult(id=req_id, result=wrapper)
    return JSONResponse(content=response.model_dump(mode="json"))


async def _handle_cancel(
    task_id: str,
    req_id: str,
    handler: TaskHandler,
    store: _TaskStore,
) -> JSONResponse:
    """Mark a task cancelled and notify the handler."""
    if not task_id:
        return _error_response(req_id, code=-32600, message="params.task_id is required for tasks/cancel")

    await handler.on_cancel(task_id)
    await store.set(task_id, TaskState(task_id=task_id, status="cancelled"))

    wrapper = AgentMessage(
        trace_id=_new_uuid(),
        protocol_used="a2a",
        sender_framework="gateway",  # type: ignore[arg-type]
        message={"cancelled": True, "task_id": task_id},
    )
    response = A2AResult(id=req_id, result=wrapper)
    return JSONResponse(content=response.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _error_response(id: str, code: int, message: str) -> JSONResponse:
    err = A2AError(id=id, error=A2AErrorBody(code=code, message=message))
    log.warning("a2a.error_response", code=code, message=message)
    return JSONResponse(content=err.model_dump(mode="json"), status_code=200)
    # A2A / JSON-RPC errors return HTTP 200 with an error body per spec.
