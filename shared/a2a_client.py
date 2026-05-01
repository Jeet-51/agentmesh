"""
Reusable async A2A client.

The orchestrator imports A2AClient to call the research and synthesis agents.
All inter-agent HTTP logic lives here so it is never duplicated per-agent.

Usage:
    async with A2AClient() as client:
        response = await client.send_task(
            agent_url=RESEARCH_AGENT_URL,
            task_card=card,
            trace_id=state.trace_id,
        )
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from shared.a2a_types import A2AError, A2ARequest, A2AResult, TaskState
from shared.models import AgentMessage, Framework, TaskCard, _new_uuid, _utcnow

log = structlog.get_logger(__name__)

# How long to wait for an agent to respond to a send (seconds).
DEFAULT_SEND_TIMEOUT = 120.0
# Interval between polls when waiting for a long-running task (seconds).
DEFAULT_POLL_INTERVAL = 2.0
# Maximum total time to wait when polling (seconds).
DEFAULT_POLL_TIMEOUT = 300.0


class A2AClientError(Exception):
    """Raised when an A2A call fails unrecoverably."""

    def __init__(self, message: str, agent_url: str, task_id: str | None = None) -> None:
        super().__init__(message)
        self.agent_url = agent_url
        self.task_id = task_id


class A2AClient:
    """
    Async HTTP client for A2A agent communication.

    Handles the JSON-RPC 2.0 envelope, structured logging of every hop,
    trace-ID propagation, and async task polling.  One instance can be
    reused across all calls within a request lifecycle.

    Context manager usage ensures the underlying httpx.AsyncClient is
    properly closed even if a call raises.
    """

    def __init__(
        self,
        sender_framework: Framework = Framework.LANGGRAPH,
        send_timeout: float = DEFAULT_SEND_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        poll_timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> None:
        self._sender_framework = sender_framework
        self._send_timeout = send_timeout
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> A2AClient:
        self._http = httpx.AsyncClient(timeout=self._send_timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_task(
        self,
        agent_url: str,
        task_card: TaskCard,
        trace_id: str,
    ) -> AgentMessage:
        """
        Send a task card to an agent and return its AgentMessage response.

        For agents that process synchronously the response arrives immediately.
        For async agents, call send_task_and_wait instead.

        Raises A2AClientError on HTTP or protocol-level failure.
        """
        self._ensure_open()

        envelope = self._wrap_task_card(task_card, trace_id)
        request = A2ARequest(
            method="message/send",
            params={"message": envelope.model_dump(mode="json")},
        )

        log.info(
            "a2a.send_task",
            agent_url=agent_url,
            task_id=task_card.task_id,
            context_id=task_card.context_id,
            trace_id=trace_id,
            sender=self._sender_framework.value,
            receiver=task_card.receiver_framework.value,
        )

        raw = await self._post(agent_url, request)
        response = self._parse_result(raw, agent_url, task_card.task_id)

        log.info(
            "a2a.send_task.ok",
            agent_url=agent_url,
            task_id=task_card.task_id,
            trace_id=trace_id,
            response_protocol=response.protocol_used,
        )
        return response

    async def get_task(self, agent_url: str, task_id: str, trace_id: str) -> TaskState:
        """
        Poll an in-flight task by ID.

        Returns the current TaskState; caller decides whether to keep polling.
        """
        self._ensure_open()

        request = A2ARequest(
            method="tasks/get",
            params={"task_id": task_id, "trace_id": trace_id},
        )

        log.debug("a2a.get_task", agent_url=agent_url, task_id=task_id, trace_id=trace_id)

        raw = await self._post(agent_url, request)

        if "error" in raw:
            err = A2AError.model_validate(raw)
            raise A2AClientError(
                f"tasks/get failed: {err.error.message}",
                agent_url=agent_url,
                task_id=task_id,
            )

        result = A2AResult.model_validate(raw)
        task_state_data = result.result.message.get("task_state")
        if task_state_data is None:
            raise A2AClientError(
                "tasks/get response missing task_state in message payload",
                agent_url=agent_url,
                task_id=task_id,
            )
        return TaskState.model_validate(task_state_data)

    async def send_task_and_wait(
        self,
        agent_url: str,
        task_card: TaskCard,
        trace_id: str,
    ) -> AgentMessage:
        """
        Send a task and poll until it completes or the poll timeout expires.

        Useful for long-running agents (CrewAI crew can take 30-60 s).
        Raises A2AClientError if the task fails or the poll timeout is hit.
        """
        initial = await self.send_task(agent_url, task_card, trace_id)

        # If the agent replied synchronously with a completed result, return it.
        if initial.message.get("status") != "in_progress":
            return initial

        task_id = task_card.task_id
        elapsed = 0.0

        while elapsed < self._poll_timeout:
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

            state = await self.get_task(agent_url, task_id, trace_id)
            log.debug(
                "a2a.poll",
                task_id=task_id,
                status=state.status,
                elapsed_s=round(elapsed, 1),
            )

            if state.status == "completed":
                if state.result is None:
                    raise A2AClientError(
                        "Task completed but result is None",
                        agent_url=agent_url,
                        task_id=task_id,
                    )
                return state.result

            if state.status == "failed":
                raise A2AClientError(
                    f"Task failed: {state.error}",
                    agent_url=agent_url,
                    task_id=task_id,
                )

        raise A2AClientError(
            f"Poll timeout after {self._poll_timeout}s",
            agent_url=agent_url,
            task_id=task_id,
        )

    async def cancel_task(self, agent_url: str, task_id: str, trace_id: str) -> None:
        """Send a cancellation signal to a running task (best-effort)."""
        self._ensure_open()

        request = A2ARequest(
            method="tasks/cancel",
            params={"task_id": task_id, "trace_id": trace_id},
        )
        try:
            await self._post(agent_url, request)
            log.info("a2a.cancel_task.sent", agent_url=agent_url, task_id=task_id)
        except Exception as exc:
            log.warning("a2a.cancel_task.failed", agent_url=agent_url, task_id=task_id, error=str(exc))

    async def health_check(self, agent_url: str) -> bool:
        """Return True if the agent's /health endpoint responds 200."""
        self._ensure_open()
        try:
            resp = await self._http.get(f"{agent_url.rstrip('/')}/health", timeout=5.0)  # type: ignore[union-attr]
            return resp.status_code == 200
        except Exception as exc:
            log.warning("a2a.health_check.failed", agent_url=agent_url, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._http is None:
            raise RuntimeError("A2AClient must be used as an async context manager.")

    def _wrap_task_card(self, task_card: TaskCard, trace_id: str) -> AgentMessage:
        """Wrap a TaskCard inside an AgentMessage envelope for transport."""
        return AgentMessage(
            trace_id=trace_id,
            protocol_used="a2a",
            sender_framework=self._sender_framework,
            message=task_card.model_dump(mode="json"),
        )

    async def _post(self, agent_url: str, request: A2ARequest) -> dict:
        """Execute an HTTP POST to the agent's /a2a endpoint."""
        url = f"{agent_url.rstrip('/')}/a2a"
        try:
            resp = await self._http.post(  # type: ignore[union-attr]
                url,
                json=request.model_dump(mode="json"),
                headers={
                    "Content-Type": "application/json",
                    "X-A2A-Version": "1.0",
                },
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            log.error(
                "a2a.http_error",
                url=url,
                status=exc.response.status_code,
                body=exc.response.text[:500],
            )
            raise A2AClientError(
                f"HTTP {exc.response.status_code} from {agent_url}",
                agent_url=agent_url,
            ) from exc
        except httpx.RequestError as exc:
            log.error("a2a.request_error", url=url, error=str(exc))
            raise A2AClientError(
                f"Connection error to {agent_url}: {exc}",
                agent_url=agent_url,
            ) from exc

    def _parse_result(self, raw: dict, agent_url: str, task_id: str) -> AgentMessage:
        """Parse raw JSON-RPC response into AgentMessage or raise on error."""
        if "error" in raw:
            err = A2AError.model_validate(raw)
            raise A2AClientError(
                f"A2A error {err.error.code}: {err.error.message}",
                agent_url=agent_url,
                task_id=task_id,
            )
        result = A2AResult.model_validate(raw)
        return result.result
