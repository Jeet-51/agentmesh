"""
LangGraph node implementations for the AgentMesh orchestrator.

Each function is an async LangGraph node — it receives the current
OrchestratorState and returns a dict of field updates.  LangGraph merges
the returned dict into the state before calling the next node and checkpoints
the full state after every node, which is what makes the human-in-the-loop
pause resumable across HTTP requests.

Node execution order:
  decompose_query  →  human_checkpoint  →  dispatch_tasks  →  merge_results
                  ↑_____reject__________|
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import structlog
from google import genai
from google.genai import types as genai_types
from langgraph.types import interrupt

from shared.a2a_client import A2AClient, A2AClientError
from shared.models import (
    Framework,
    OrchestratorState,
    PartialResult,
    ReportStatus,
    ResearchFindings,
    SubTask,
    SubTaskStatus,
    SynthesisReport,
    TaskCard,
)

log = structlog.get_logger(__name__)

RESEARCH_AGENT_URL = os.environ.get("RESEARCH_AGENT_URL", "http://research:8002")
SYNTHESIS_AGENT_URL = os.environ.get("SYNTHESIS_AGENT_URL", "http://synthesis:8003")

# Module-level singleton — configured once, reused across calls.
_gemini_client: genai.Client | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _gemini() -> genai.Client:
    """Return the cached google-genai Client, initialising on first call."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _gemini_client


# ---------------------------------------------------------------------------
# Prompt template for decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a research orchestrator. Decompose the user query into 2-3 focused,
non-overlapping sub-tasks that can each be researched independently.

User query: {query}
{feedback_block}
Return ONLY a JSON object with this exact structure — no markdown fences:
{{
  "sub_tasks": [
    {{
      "topic": "<concise topic, max 100 chars>",
      "instructions": "<specific research instructions: what to find, which sources to check, which claims to verify>"
    }}
  ]
}}

Rules:
- 2 sub-tasks for simple queries, 3 for multi-faceted ones.
- Each topic must be independently researchable with no overlap.
- Instructions must be concrete and actionable, not vague.
"""


# ---------------------------------------------------------------------------
# Node 1 — decompose_query
# ---------------------------------------------------------------------------


async def decompose_query(state: OrchestratorState) -> dict[str, Any]:
    """
    Call Gemini Flash to split the user query into 2-3 SubTasks.

    If the human_checkpoint previously rejected a decomposition, the node
    incorporates that context so Gemini produces a revised breakdown.
    """
    bound_log = log.bind(trace_id=state.trace_id, run_id=state.run_id, node="decompose_query")
    bound_log.info("node.start", query_preview=state.query[:120])

    # Include rejection context if this is a re-run.
    feedback_block = ""
    if state.sub_tasks and not state.human_approved:
        previous_topics = ", ".join(t.topic for t in state.sub_tasks)
        feedback_block = (
            f"\nA previous decomposition was rejected by the user. "
            f"Previous topics were: {previous_topics}. "
            "Produce a meaningfully different breakdown."
        )

    prompt = _DECOMPOSE_PROMPT.format(query=state.query, feedback_block=feedback_block)

    try:
        response = await _gemini().aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
    except Exception as exc:
        bound_log.error("node.decompose.gemini_error", error=str(exc))
        raise

    try:
        raw: dict[str, Any] = json.loads(response.text)
        raw_tasks: list[dict[str, str]] = raw["sub_tasks"]
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("Gemini returned an empty sub_tasks list.")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        bound_log.error(
            "node.decompose.parse_error",
            raw_response=response.text[:500],
            error=str(exc),
        )
        raise

    sub_tasks = [
        SubTask(
            topic=item["topic"],
            instructions=item["instructions"],
            parent_query=state.query,
        )
        for item in raw_tasks[:3]  # hard-cap at 3
    ]

    bound_log.info(
        "node.decompose.done",
        sub_task_count=len(sub_tasks),
        topics=[t.topic for t in sub_tasks],
    )

    return {
        "sub_tasks": sub_tasks,
        "human_approved": False,
        "status": ReportStatus.AWAITING_HUMAN,
        "updated_at": _utcnow(),
    }


# ---------------------------------------------------------------------------
# Node 2 — human_checkpoint
# ---------------------------------------------------------------------------


async def human_checkpoint(state: OrchestratorState) -> dict[str, Any]:
    """
    Real async pause for human review of the decomposed sub-tasks.

    Uses LangGraph's interrupt() which suspends graph execution and
    checkpoints state at this exact point.  The graph runner (a2a_server.py)
    surfaces the interrupt payload to the gateway, which forwards it to the
    React frontend.  Execution resumes only when the gateway calls:

        graph.invoke(Command(resume=payload), config=checkpoint_config)

    Expected resume payload shapes:
      {"approved": True}
        — approve the sub-tasks as-is
      {"approved": True, "sub_tasks": [{...}, ...]}
        — approve with user-edited sub-tasks
      {"approved": False, "feedback": "..."}
        — reject; orchestrator routes back to decompose_query
    """
    bound_log = log.bind(trace_id=state.trace_id, run_id=state.run_id, node="human_checkpoint")
    bound_log.info("node.start", sub_task_count=len(state.sub_tasks))

    # Graph suspends here. The interrupt value is delivered to the caller
    # as a GraphInterrupt exception; it re-enters here when resumed.
    resume: dict[str, Any] = interrupt(
        {
            "run_id": state.run_id,
            "trace_id": state.trace_id,
            "sub_tasks": [t.model_dump(mode="json") for t in state.sub_tasks],
            "message": "Review the decomposed sub-tasks. Approve, edit, or reject.",
        }
    )

    approved: bool = bool(resume.get("approved", False))

    if not approved:
        feedback = resume.get("feedback", "no feedback provided")
        bound_log.info("node.checkpoint.rejected", feedback=feedback)
        # Return human_approved=False; the conditional edge routes back to decompose_query.
        return {
            "human_approved": False,
            "status": ReportStatus.PENDING,
            "updated_at": _utcnow(),
        }

    # Approved — check for edited sub-tasks.
    if "sub_tasks" in resume:
        try:
            edited = [SubTask.model_validate(t) for t in resume["sub_tasks"]]
        except Exception as exc:
            bound_log.warning("node.checkpoint.edit_parse_error", error=str(exc))
            edited = state.sub_tasks  # fall back to original on bad input
        bound_log.info("node.checkpoint.approved_with_edits", edited_count=len(edited))
        return {
            "sub_tasks": edited,
            "human_approved": True,
            "status": ReportStatus.RUNNING,
            "updated_at": _utcnow(),
        }

    bound_log.info("node.checkpoint.approved_as_is")
    return {
        "human_approved": True,
        "status": ReportStatus.RUNNING,
        "updated_at": _utcnow(),
    }


# ---------------------------------------------------------------------------
# Node 3 — dispatch_tasks
# ---------------------------------------------------------------------------


async def dispatch_tasks(state: OrchestratorState) -> dict[str, Any]:
    """
    Dispatch each SubTask to the research agent, then pass all findings
    to the synthesis agent.  Both calls use send_task_and_wait() because
    CrewAI and ADK runs are long-lived (30-90 s).

    Retry policy: each agent gets one retry (attempt 0 + attempt 1).
    SubTask.retry_count tracks per-task retries.  Any agent that exhausts
    retries is recorded in PartialResult so the user still gets output.
    """
    bound_log = log.bind(trace_id=state.trace_id, run_id=state.run_id, node="dispatch_tasks")
    bound_log.info("node.start", sub_task_count=len(state.sub_tasks))

    all_findings: list[ResearchFindings] = []
    failed_agents: list[Framework] = []
    failure_reasons: dict[str, str] = {}
    updated_sub_tasks: list[SubTask] = list(state.sub_tasks)

    async with A2AClient(sender_framework=Framework.LANGGRAPH) as client:

        # ----------------------------------------------------------------
        # Phase 1 — research agent (one TaskCard per SubTask)
        # ----------------------------------------------------------------
        for idx, sub_task in enumerate(updated_sub_tasks):
            bound_log.info(
                "node.dispatch.research.begin",
                sub_task_id=sub_task.sub_task_id,
                topic=sub_task.topic,
            )
            findings, final_sub_task = await _call_research(
                client, sub_task, state, bound_log
            )
            updated_sub_tasks[idx] = final_sub_task

            if findings is not None:
                all_findings.append(findings)
            else:
                if Framework.CREWAI not in failed_agents:
                    failed_agents.append(Framework.CREWAI)
                failure_reasons[Framework.CREWAI.value] = (
                    f"Research failed for sub_task {sub_task.sub_task_id} after retry"
                )

        # If every research task failed, return a PartialResult immediately.
        if not all_findings:
            bound_log.error(
                "node.dispatch.all_research_failed",
                failed=[f.value for f in failed_agents],
            )
            partial = PartialResult(
                run_id=state.run_id,
                completed_agents=[],
                failed_agents=failed_agents,
                partial_report="All research tasks failed. No findings available.",
                failure_reasons=failure_reasons,
            )
            return {
                "sub_tasks": updated_sub_tasks,
                "findings": [],
                "partial_result": partial,
                "status": ReportStatus.PARTIAL,
                "updated_at": _utcnow(),
            }

        # ----------------------------------------------------------------
        # Phase 2 — synthesis agent (all findings in one TaskCard)
        # ----------------------------------------------------------------
        bound_log.info("node.dispatch.synthesis.begin", finding_count=len(all_findings))
        synthesis_report = await _call_synthesis(client, all_findings, state, bound_log)

        if synthesis_report is not None:
            bound_log.info(
                "node.dispatch.synthesis.ok",
                report_id=synthesis_report.report_id,
                citations=len(synthesis_report.citations),
            )
            return {
                "sub_tasks": updated_sub_tasks,
                "findings": all_findings,
                "final_report": synthesis_report,
                "status": ReportStatus.COMPLETED,
                "updated_at": _utcnow(),
            }

        # Synthesis failed after retry — build partial from research-only findings.
        failed_agents.append(Framework.GOOGLE_ADK)
        failure_reasons[Framework.GOOGLE_ADK.value] = "Synthesis agent failed after retry"
        partial_narrative = "\n\n---\n\n".join(
            f"**{f.sub_task_id}**\n{f.findings}" for f in all_findings
        )
        partial = PartialResult(
            run_id=state.run_id,
            completed_agents=[Framework.CREWAI],
            failed_agents=failed_agents,
            partial_report=f"Synthesis failed. Research findings:\n\n{partial_narrative}",
            available_findings=all_findings,
            failure_reasons=failure_reasons,
        )
        return {
            "sub_tasks": updated_sub_tasks,
            "findings": all_findings,
            "partial_result": partial,
            "status": ReportStatus.PARTIAL,
            "updated_at": _utcnow(),
        }


async def _call_research(
    client: A2AClient,
    sub_task: SubTask,
    state: OrchestratorState,
    bound_log: Any,
) -> tuple[ResearchFindings | None, SubTask]:
    """
    Send one SubTask to the research agent, retrying once on failure.

    Returns (findings, updated_sub_task).  findings is None if both
    attempts fail.
    """
    for attempt in range(2):
        updated = sub_task.model_copy(
            update={
                "status": SubTaskStatus.RETRYING if attempt == 1 else SubTaskStatus.DISPATCHED,
                "retry_count": attempt,
                "assigned_to": Framework.CREWAI,
            }
        )
        try:
            card = TaskCard(
                sender_framework=Framework.LANGGRAPH,
                receiver_framework=Framework.CREWAI,
                context_id=state.run_id,
                payload=updated.model_dump(mode="json"),
                retry_count=attempt,
            )
            bound_log.info(
                "node.dispatch.research.send",
                sub_task_id=sub_task.sub_task_id,
                attempt=attempt,
                task_id=card.task_id,
            )
            response = await client.send_task_and_wait(
                agent_url=RESEARCH_AGENT_URL,
                task_card=card,
                trace_id=state.trace_id,
            )
            findings = ResearchFindings.model_validate(response.message)
            completed = updated.model_copy(
                update={"status": SubTaskStatus.COMPLETED, "completed_at": _utcnow()}
            )
            bound_log.info(
                "node.dispatch.research.ok",
                sub_task_id=sub_task.sub_task_id,
                confidence=findings.confidence_score,
                sources=len(findings.sources),
            )
            return findings, completed

        except (A2AClientError, Exception) as exc:
            bound_log.warning(
                "node.dispatch.research.attempt_failed",
                sub_task_id=sub_task.sub_task_id,
                attempt=attempt,
                error=str(exc),
            )

    failed = sub_task.model_copy(
        update={"status": SubTaskStatus.FAILED, "retry_count": 1, "assigned_to": Framework.CREWAI}
    )
    bound_log.error("node.dispatch.research.exhausted", sub_task_id=sub_task.sub_task_id)
    return None, failed


async def _call_synthesis(
    client: A2AClient,
    findings: list[ResearchFindings],
    state: OrchestratorState,
    bound_log: Any,
) -> SynthesisReport | None:
    """
    Send all research findings to the synthesis agent, retrying once on failure.

    Returns SynthesisReport or None if both attempts fail.
    """
    payload = {
        "run_id": state.run_id,
        "query": state.query,
        "findings": [f.model_dump(mode="json") for f in findings],
    }

    for attempt in range(2):
        try:
            card = TaskCard(
                sender_framework=Framework.LANGGRAPH,
                receiver_framework=Framework.GOOGLE_ADK,
                context_id=state.run_id,
                payload=payload,
                retry_count=attempt,
            )
            bound_log.info(
                "node.dispatch.synthesis.send",
                finding_count=len(findings),
                attempt=attempt,
                task_id=card.task_id,
            )
            response = await client.send_task_and_wait(
                agent_url=SYNTHESIS_AGENT_URL,
                task_card=card,
                trace_id=state.trace_id,
            )
            return SynthesisReport.model_validate(response.message)

        except (A2AClientError, Exception) as exc:
            bound_log.warning(
                "node.dispatch.synthesis.attempt_failed",
                attempt=attempt,
                error=str(exc),
            )

    bound_log.error("node.dispatch.synthesis.exhausted")
    return None


# ---------------------------------------------------------------------------
# Node 4 — merge_results
# ---------------------------------------------------------------------------


async def merge_results(state: OrchestratorState) -> dict[str, Any]:
    """
    Finalise the orchestration run.

    dispatch_tasks sets either final_report (full success) or partial_result
    (degraded path).  This node logs the outcome, confirms the terminal
    status, and is where LangSmith trace data gets finalised.
    """
    bound_log = log.bind(trace_id=state.trace_id, run_id=state.run_id, node="merge_results")

    if state.final_report is not None:
        bound_log.info(
            "node.merge.complete",
            report_id=state.final_report.report_id,
            citation_count=len(state.final_report.citations),
            mcp_tools_called=state.final_report.mcp_tools_called,
            confidence_scores=state.final_report.confidence_scores,
        )
        return {"status": ReportStatus.COMPLETED, "updated_at": _utcnow()}

    if state.partial_result is not None:
        bound_log.warning(
            "node.merge.partial",
            completed=[f.value for f in state.partial_result.completed_agents],
            failed=[f.value for f in state.partial_result.failed_agents],
            finding_count=len(state.partial_result.available_findings),
            failure_reasons=state.partial_result.failure_reasons,
        )
        return {"status": ReportStatus.PARTIAL, "updated_at": _utcnow()}

    # Should never reach here — dispatch_tasks always sets one of the above.
    bound_log.error("node.merge.inconsistent_state — neither final_report nor partial_result is set")
    return {"status": ReportStatus.FAILED, "updated_at": _utcnow()}
