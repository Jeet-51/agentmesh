"""
A2A protocol wire types.

Follows the A2A v1.0 spec (JSON-RPC 2.0 envelope) for the HTTP transport layer.
Domain payloads live inside AgentMessage.message — the wire types here are
protocol-level containers only.

A2A methods used in AgentMesh:
  message/send  — dispatch a task card to an agent
  tasks/get     — poll an in-flight task by task_id
  tasks/cancel  — cancel a running task (used on retry path)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.models import AgentMessage, _new_uuid


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 envelope
# ---------------------------------------------------------------------------


class A2ARequest(BaseModel):
    """Outbound A2A request following JSON-RPC 2.0 structure."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str = Field(default_factory=_new_uuid, description="Request ID for correlating responses.")
    method: Literal["message/send", "tasks/get", "tasks/cancel"] = Field(...)
    params: dict[str, Any] = Field(default_factory=dict)


class A2AResult(BaseModel):
    """Successful A2A response envelope."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str = Field(..., description="Echoed from the request.")
    result: AgentMessage = Field(..., description="The agent's response wrapped in AgentMessage.")


class A2AError(BaseModel):
    """A2A error response envelope."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str = Field(...)
    error: A2AErrorBody = Field(...)


class A2AErrorBody(BaseModel):
    """JSON-RPC 2.0 error body."""

    code: int = Field(..., description="Error code: -32600 invalid request, -32000 agent error.")
    message: str = Field(...)
    data: dict[str, Any] | None = Field(default=None)


# Rebuild after forward ref on A2AError → A2AErrorBody
A2AError.model_rebuild()


# ---------------------------------------------------------------------------
# Task status poll response
# ---------------------------------------------------------------------------


class TaskState(BaseModel):
    """Returned by tasks/get — describes the current state of a running task."""

    task_id: str
    status: Literal["pending", "in_progress", "completed", "failed", "cancelled"]
    result: AgentMessage | None = Field(default=None, description="Populated when status == completed.")
    error: str | None = Field(default=None, description="Populated when status == failed.")
