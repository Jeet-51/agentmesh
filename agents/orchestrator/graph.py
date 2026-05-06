"""
LangGraph StateGraph definition for the AgentMesh orchestrator.

Graph topology:
                      ┌──────────────────────────────────┐
                      │                                  │
  START → decompose_query → human_checkpoint → dispatch_tasks → merge_results → END
                               │
                         (approved=False)
                               │
                      ← ← ← ← ┘   (re-decompose with feedback)

The human_checkpoint node uses LangGraph interrupt() to pause the graph.
The conditional edge after human_checkpoint checks state.human_approved:
  True  → dispatch_tasks
  False → decompose_query   (user rejected; re-run with feedback incorporated)
"""

from __future__ import annotations

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from shared.models import OrchestratorState, ReportStatus

from nodes import (
    decompose_query,
    dispatch_tasks,
    human_checkpoint,
    merge_results,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_after_checkpoint(state: OrchestratorState) -> str:
    """
    Conditional edge fired after human_checkpoint returns.

    Returns the name of the next node to execute.
    """
    if state.human_approved:
        log.debug("graph.route.approved", run_id=state.run_id)
        return "dispatch_tasks"

    log.debug("graph.route.rejected", run_id=state.run_id)
    return "decompose_query"


def route_after_dispatch(state: OrchestratorState) -> str:
    """
    Conditional edge fired after dispatch_tasks returns.

    Always goes to merge_results; the merge node handles both success
    and partial-result paths without branching.
    """
    return "merge_results"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Construct and compile the orchestrator StateGraph.

    Uses MemorySaver as the checkpointer so interrupt() state survives
    across the HTTP request boundary (the graph runner suspends, the
    checkpoint HTTP endpoint resumes it in a new request).

    Returns a compiled graph ready for .ainvoke() / .astream().
    """
    builder = StateGraph(OrchestratorState)

    # Register nodes.
    builder.add_node("decompose_query", decompose_query)
    builder.add_node("human_checkpoint", human_checkpoint)
    builder.add_node("dispatch_tasks", dispatch_tasks)
    builder.add_node("merge_results", merge_results)

    # Entry edge.
    builder.add_edge(START, "decompose_query")

    # decompose_query always proceeds to human_checkpoint.
    builder.add_edge("decompose_query", "human_checkpoint")

    # human_checkpoint: branch on approval.
    builder.add_conditional_edges(
        "human_checkpoint",
        route_after_checkpoint,
        {
            "dispatch_tasks": "dispatch_tasks",
            "decompose_query": "decompose_query",
        },
    )

    # dispatch_tasks always goes to merge_results.
    builder.add_edge("dispatch_tasks", "merge_results")

    # merge_results is the terminal node.
    builder.add_edge("merge_results", END)

    checkpointer = MemorySaver()
    compiled = builder.compile(
        checkpointer=checkpointer,
        # Declare the interrupt point so LangGraph knows to checkpoint here.
        interrupt_before=["human_checkpoint"],
    )

    log.info("graph.compiled", nodes=["decompose_query", "human_checkpoint", "dispatch_tasks", "merge_results"])
    return compiled


# Module-level graph instance — imported by a2a_server.py.
graph = build_graph()
