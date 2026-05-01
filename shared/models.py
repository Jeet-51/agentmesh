"""
Shared Pydantic v2 models for AgentMesh.

All inter-agent data crosses service boundaries through these contracts.
Every model that travels over A2A or MCP is wrapped in AgentMessage so
protocol, sender framework, and trace ID are always visible in logs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Framework(str, Enum):
    """Identifies which AI framework produced or will consume a message."""

    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    GOOGLE_ADK = "google_adk"
    GATEWAY = "gateway"


class Protocol(str, Enum):
    """Transport protocol used for this inter-agent hop."""

    A2A = "a2a"
    MCP = "mcp"


class SubTaskStatus(str, Enum):
    """Lifecycle state of a decomposed sub-task."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ReportStatus(str, Enum):
    """Overall status of the orchestration run."""

    PENDING = "pending"
    AWAITING_HUMAN = "awaiting_human"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Core envelope models
# ---------------------------------------------------------------------------


class TaskCard(BaseModel):
    """
    A2A task envelope — every cross-agent delegation is wrapped in this.

    Carries routing metadata (sender/receiver framework, context continuity)
    alongside the opaque payload so the A2A layer can route without
    understanding domain content.
    """

    task_id: str = Field(default_factory=_new_uuid, description="Unique ID for this task instance.")
    context_id: str = Field(default_factory=_new_uuid, description="Thread ID shared across related tasks (A2A contextId).")
    sender_framework: Framework = Field(..., description="Framework that created this task card.")
    receiver_framework: Framework = Field(..., description="Framework expected to execute this task.")
    payload: dict[str, Any] = Field(..., description="Domain payload — must be JSON-serialisable.")
    timestamp: datetime = Field(default_factory=_utcnow, description="UTC creation time.")
    retry_count: int = Field(default=0, ge=0, le=2, description="Number of times this task has been retried.")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class AgentMessage(BaseModel):
    """
    Universal inter-agent wrapper.

    Every payload that crosses a service boundary — whether via A2A or MCP —
    is stamped with a trace_id, the protocol used, and the sending framework.
    This makes cross-framework hops immediately visible in structured logs.
    """

    trace_id: str = Field(default_factory=_new_uuid, description="End-to-end trace ID propagated through all hops.")
    protocol_used: Literal["a2a", "mcp"] = Field(..., description="Transport protocol for this hop.")
    sender_framework: Framework = Field(..., description="Framework that emitted this message.")
    timestamp: datetime = Field(default_factory=_utcnow)
    message: dict[str, Any] = Field(..., description="Wrapped domain payload.")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ---------------------------------------------------------------------------
# Orchestrator domain models
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """
    A single unit of decomposed research work.

    The orchestrator creates one SubTask per decomposed topic slice and
    dispatches each to the research agent via A2A.
    """

    sub_task_id: str = Field(default_factory=_new_uuid)
    topic: str = Field(..., min_length=1, max_length=512, description="Focused research topic for this slice.")
    instructions: str = Field(..., min_length=1, description="Specific instructions for the research crew.")
    status: SubTaskStatus = Field(default=SubTaskStatus.PENDING)
    parent_query: str = Field(..., description="Original user query this sub-task was derived from.")
    assigned_to: Framework | None = Field(default=None, description="Framework this sub-task was dispatched to.")
    retry_count: int = Field(default=0, ge=0, le=2, description="Per-sub-task retry counter; max 1 retry before partial result.")
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = Field(default=None)

    @field_validator("topic")
    @classmethod
    def topic_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be blank")
        return v.strip()


class OrchestratorState(BaseModel):
    """
    LangGraph StateGraph state object.

    Passed between nodes at each graph step. Mutable fields are updated
    in-place by each node; LangGraph checkpoints the full state after
    every node so the human-in-the-loop pause is resumable.
    """

    run_id: str = Field(default_factory=_new_uuid, description="Unique ID for this orchestration run.")
    query: str = Field(..., min_length=1, description="Raw user query.")
    sub_tasks: list[SubTask] = Field(default_factory=list)
    findings: list[ResearchFindings] = Field(default_factory=list)
    final_report: SynthesisReport | None = Field(default=None)
    partial_result: PartialResult | None = Field(default=None)
    human_approved: bool = Field(default=False, description="Set to True after human confirms sub-tasks at checkpoint.")
    status: ReportStatus = Field(default=ReportStatus.PENDING)
    trace_id: str = Field(default_factory=_new_uuid, description="Propagated to all child AgentMessage envelopes.")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ---------------------------------------------------------------------------
# Research agent domain models
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A single cited source returned by the research crew."""

    url: str | None = Field(default=None)
    title: str = Field(..., min_length=1)
    snippet: str = Field(default="", description="Relevant excerpt from the source.")
    retrieved_at: datetime = Field(default_factory=_utcnow)


class ResearchFindings(BaseModel):
    """
    Structured output from the CrewAI research crew.

    Returned via A2A to the orchestrator after the 3-member crew
    (Searcher → Fact-checker → Summarizer) completes its run.
    """

    sub_task_id: str = Field(..., description="Links back to the SubTask that triggered this run.")
    findings: str = Field(..., min_length=1, description="Synthesised research narrative from the Summarizer agent.")
    sources: list[Source] = Field(default_factory=list, description="All sources cited by the crew.")
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Crew's self-assessed confidence (0–1)."
    )
    fact_check_passed: bool = Field(..., description="Whether the Fact-checker agent validated the claims.")
    raw_search_results: list[str] = Field(default_factory=list, description="Raw snippets from the Searcher agent.")
    completed_at: datetime = Field(default_factory=_utcnow)

    @field_validator("confidence_score")
    @classmethod
    def score_precision(cls, v: float) -> float:
        return round(v, 4)


# ---------------------------------------------------------------------------
# Synthesis agent domain models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single citation in the synthesis report."""

    citation_id: str = Field(default_factory=_new_uuid)
    source_url: str | None = Field(default=None)
    source_title: str = Field(..., min_length=1)
    claim: str = Field(..., description="The specific claim this citation supports.")
    tool_used: Literal["yfinance", "wikipedia", "edgar", "crewai_research"] = Field(
        ..., description="Which MCP tool or agent produced this citation."
    )


class SynthesisReport(BaseModel):
    """
    Final structured output from the Google ADK synthesis agent.

    Produced after the ADK agent enriches the research findings with
    live data from MCP tools (yfinance, Wikipedia, SEC EDGAR).
    """

    report_id: str = Field(default_factory=_new_uuid)
    run_id: str = Field(..., description="Links back to the OrchestratorState run.")
    narrative: str = Field(..., min_length=1, description="Full prose report with inline citation markers.")
    citations: list[Citation] = Field(default_factory=list)
    confidence_scores: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict,
        description="Per-section confidence scores keyed by section name.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations derived from the research.",
    )
    mcp_tools_called: list[str] = Field(
        default_factory=list,
        description="Names of MCP tools invoked during synthesis.",
    )
    completed_at: datetime = Field(default_factory=_utcnow)

    @field_validator("confidence_scores")
    @classmethod
    def validate_scores(cls, v: dict[str, float]) -> dict[str, float]:
        return {k: round(score, 4) for k, score in v.items()}

    @model_validator(mode="after")
    def at_least_one_citation(self) -> SynthesisReport:
        if not self.citations:
            raise ValueError("SynthesisReport must include at least one citation.")
        return self


# ---------------------------------------------------------------------------
# Partial result (retry / failure path)
# ---------------------------------------------------------------------------


class PartialResult(BaseModel):
    """
    Returned by the orchestrator when one or more agents fail after retry.

    Ensures the user always gets something useful even under partial failure.
    """

    run_id: str = Field(..., description="Links back to the OrchestratorState run.")
    completed_agents: list[Framework] = Field(
        default_factory=list,
        description="Frameworks that returned results successfully.",
    )
    failed_agents: list[Framework] = Field(
        default_factory=list,
        description="Frameworks that failed after one retry.",
    )
    partial_report: str = Field(
        ...,
        description="Best-effort narrative assembled from whatever completed.",
    )
    available_findings: list[ResearchFindings] = Field(default_factory=list)
    failure_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Per-framework error messages keyed by Framework value.",
    )
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def must_have_failures(self) -> PartialResult:
        if not self.failed_agents:
            raise ValueError("PartialResult requires at least one failed agent.")
        return self


# ---------------------------------------------------------------------------
# Forward references (OrchestratorState references ResearchFindings and
# SynthesisReport which are defined after it in naive ordering)
# ---------------------------------------------------------------------------

OrchestratorState.model_rebuild()
