"""
Research agent A2A server.

Subclasses TaskHandler from shared/a2a_server.py.  The orchestrator sends
a TaskCard whose payload is a serialised SubTask; this server unpacks it,
runs ResearchCrew, and returns ResearchFindings wrapped in AgentMessage.

CrewAI's kickoff() is synchronous and CPU/IO-bound (LLM calls + web search).
We run it in a thread via asyncio.to_thread() so the FastAPI event loop
remains free to handle health checks and A2A polls during a long crew run.
"""

from __future__ import annotations

import asyncio
import logging
import os

import structlog
import uvicorn

from agents.research.crew import ResearchCrew
from shared.a2a_server import TaskHandler, create_a2a_app
from shared.models import AgentMessage, Framework, SubTask, TaskCard

# ---------------------------------------------------------------------------
# Logging setup
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
# A2A TaskHandler
# ---------------------------------------------------------------------------


class ResearchHandler(TaskHandler):
    """
    A2A handler for the research agent service.

    Receives a TaskCard from the orchestrator, validates the SubTask
    payload, runs ResearchCrew in a thread pool, and returns
    ResearchFindings as an AgentMessage.

    Error propagation: any exception raised here is caught by the
    create_a2a_app router, which stores the failure in TaskStore and
    returns a JSON-RPC error response to the orchestrator.
    """

    async def handle_task(self, task_card: TaskCard, trace_id: str) -> AgentMessage:
        """Run the three-agent research crew and return findings."""
        bound_log = log.bind(
            trace_id=trace_id,
            task_id=task_card.task_id,
            context_id=task_card.context_id,
            sender=task_card.sender_framework.value,
            retry_count=task_card.retry_count,
        )
        bound_log.info("research.handle_task.start")

        # Unpack and validate the SubTask from the TaskCard payload.
        try:
            sub_task = SubTask.model_validate(task_card.payload)
        except Exception as exc:
            bound_log.error("research.handle_task.bad_payload", error=str(exc))
            raise ValueError(f"TaskCard payload is not a valid SubTask: {exc}") from exc

        bound_log.info(
            "research.handle_task.crew_starting",
            sub_task_id=sub_task.sub_task_id,
            topic=sub_task.topic,
        )

        # Run synchronous crew.kickoff() in a thread to avoid blocking the loop.
        crew = ResearchCrew(sub_task=sub_task, trace_id=trace_id)

        try:
            findings = await asyncio.to_thread(crew.run)
        except Exception as exc:
            bound_log.error(
                "research.handle_task.crew_error",
                sub_task_id=sub_task.sub_task_id,
                error=str(exc),
            )
            raise

        bound_log.info(
            "research.handle_task.complete",
            sub_task_id=sub_task.sub_task_id,
            confidence=findings.confidence_score,
            fact_check_passed=findings.fact_check_passed,
            source_count=len(findings.sources),
        )

        return AgentMessage(
            trace_id=trace_id,
            protocol_used="a2a",
            sender_framework=Framework.CREWAI,
            message=findings.model_dump(mode="json"),
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = create_a2a_app(ResearchHandler(), title="Research Agent")

if __name__ == "__main__":
    uvicorn.run(
        "a2a_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8002)),
        reload=False,
    )
