"""
Synthesis agent A2A server.

Subclasses TaskHandler from shared/a2a_server.py.  The orchestrator sends
a TaskCard whose payload is:
  {
    "run_id":   "<orchestration run id>",
    "query":    "<original user query>",
    "findings": [<ResearchFindings>, ...]
  }

This server initialises SynthesisAgent once at startup (FastAPI lifespan)
and reuses the same ADK runner + MCP connections across all requests.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI

from agents.synthesis.agent import SynthesisAgent
from shared.a2a_server import TaskHandler, create_a2a_app
from shared.models import AgentMessage, Framework, ResearchFindings, TaskCard

# ---------------------------------------------------------------------------
# Structured logging setup
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
# Module-level agent singleton — initialised in lifespan
# ---------------------------------------------------------------------------

_agent: SynthesisAgent | None = None


# ---------------------------------------------------------------------------
# A2A TaskHandler
# ---------------------------------------------------------------------------


class SynthesisHandler(TaskHandler):
    """
    A2A handler for the synthesis agent service.

    Receives a TaskCard from the orchestrator containing the original query
    and all ResearchFindings, runs the ADK synthesis agent, and returns a
    SynthesisReport wrapped in AgentMessage.
    """

    async def handle_task(self, task_card: TaskCard, trace_id: str) -> AgentMessage:
        """Enrich findings with MCP tools and return a SynthesisReport."""
        bound_log = log.bind(
            trace_id=trace_id,
            task_id=task_card.task_id,
            context_id=task_card.context_id,
            sender=task_card.sender_framework.value,
            retry_count=task_card.retry_count,
        )
        bound_log.info("synthesis.handle_task.start")

        if _agent is None:
            raise RuntimeError("SynthesisAgent is not initialised. Check the lifespan startup.")

        # Unpack payload.
        payload = task_card.payload
        run_id: str = payload.get("run_id", task_card.context_id)
        query: str = payload.get("query", "")
        raw_findings: list[dict[str, Any]] = payload.get("findings", [])

        if not query:
            raise ValueError("TaskCard payload missing required field: 'query'")

        # Validate each findings dict into a typed ResearchFindings model.
        findings: list[ResearchFindings] = []
        for i, raw in enumerate(raw_findings):
            try:
                findings.append(ResearchFindings.model_validate(raw))
            except Exception as exc:
                bound_log.warning(
                    "synthesis.handle_task.findings_parse_error",
                    index=i,
                    error=str(exc),
                )
                # Skip malformed findings rather than failing the whole run.

        if not findings:
            raise ValueError(
                "TaskCard payload contained no valid ResearchFindings. "
                "Cannot synthesise an empty dataset."
            )

        bound_log.info(
            "synthesis.handle_task.running",
            run_id=run_id,
            finding_count=len(findings),
        )

        report = await _agent.synthesize(
            query=query,
            findings=findings,
            run_id=run_id,
            trace_id=trace_id,
        )

        bound_log.info(
            "synthesis.handle_task.complete",
            report_id=report.report_id,
            citations=len(report.citations),
            mcp_tools_called=report.mcp_tools_called,
            overall_confidence=report.confidence_scores.get("overall"),
        )

        return AgentMessage(
            trace_id=trace_id,
            protocol_used="a2a",
            sender_framework=Framework.GOOGLE_ADK,
            message=report.model_dump(mode="json"),
        )


# ---------------------------------------------------------------------------
# FastAPI app with lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the SynthesisAgent (and all MCP connections) at startup.
    Clean up connections gracefully on shutdown.
    """
    global _agent
    log.info("synthesis.lifespan.startup")

    _agent = SynthesisAgent()
    try:
        await _agent.initialize()
        log.info("synthesis.lifespan.ready")
    except Exception as exc:
        # Agent initialisation failure is non-fatal — it will use fallback path.
        log.error("synthesis.lifespan.init_error", error=str(exc))

    yield

    log.info("synthesis.lifespan.shutdown")
    if _agent is not None:
        await _agent.close()


def create_app() -> FastAPI:
    """Build the synthesis FastAPI app with lifespan and A2A routes."""
    base_app = create_a2a_app(SynthesisHandler(), title="Synthesis Agent")
    base_app.router.lifespan_context = lifespan
    return base_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "a2a_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8003)),
        reload=False,
    )
