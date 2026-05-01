"""
Google ADK synthesis agent for AgentMesh.

Receives ResearchFindings from the orchestrator, enriches them with live
data from three MCP tool servers (yfinance, Wikipedia, SEC EDGAR), and
produces a final SynthesisReport with citations and confidence scores.

Architecture
------------
SynthesisAgent wraps an ADK LlmAgent.  At startup it creates MCPToolset
connections to all three tool servers.  Each tool becomes an ADK tool that
Gemini Flash can call during the synthesis run.

Tool call tracking
------------------
Every time the agent calls an MCP tool, the runner emits a FunctionCall
event.  We collect those events to build Citation objects, ensuring the
at_least_one_citation validator on SynthesisReport is always satisfied.

Graceful degradation
--------------------
If an MCP tool server is unreachable at startup, the agent continues with
the remaining tools.  If ALL three fail, the agent still synthesises a
report using only the ResearchFindings input — no empty responses.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any

import structlog
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from google.genai import types as genai_types

from shared.models import (
    Citation,
    Framework,
    ResearchFindings,
    SynthesisReport,
    _new_uuid,
)

log = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# MCP tool server URLs
# ---------------------------------------------------------------------------

_YFINANCE_URL = os.environ.get("YFINANCE_TOOL_URL", "http://yfinance-tool:8011/sse")
_WIKIPEDIA_URL = os.environ.get("WIKIPEDIA_TOOL_URL", "http://wikipedia-tool:8012/sse")
_EDGAR_URL = os.environ.get("EDGAR_TOOL_URL", "http://edgar-tool:8013/sse")

# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are an expert research synthesis agent with access to live financial data,
encyclopedic knowledge, and SEC regulatory filings.

You have received structured research findings from a multi-agent research team.
Your task is to:
1. Read all provided findings carefully.
2. Use your MCP tools to enrich, verify, and extend the findings with live data.
3. Produce a comprehensive, well-cited intelligence report.

CITATION RULES (strictly enforced):
- Every factual claim must end with a tool attribution in brackets:
  [yfinance], [wikipedia], [edgar], or [research] (for claims from the provided findings).
- Aim to call each available tool at least once so the report is grounded in live data.
- If a tool returns an error, note it briefly and continue with available sources.

OUTPUT FORMAT:
After your analysis and tool calls, output a JSON block (and nothing after it)
wrapped in ```json ... ``` with this exact structure:

```json
{
  "narrative": "<Full report, 400-800 words, prose with inline [tool] citations>",
  "confidence_scores": {
    "overall": <float 0.0-1.0>,
    "<section>": <float 0.0-1.0>
  },
  "recommended_actions": [
    "<actionable recommendation 1>",
    "<actionable recommendation 2>"
  ],
  "citations": [
    {
      "source_title": "<descriptive title for this source>",
      "source_url": "<url or empty string>",
      "claim": "<specific claim this citation supports>",
      "tool_used": "<yfinance|wikipedia|edgar|crewai_research>"
    }
  ]
}
```

confidence_score guidance:
  0.9-1.0  Strong multi-source consensus, all claims verified with live data.
  0.7-0.9  Good evidence, minor gaps, most claims tool-verified.
  0.5-0.7  Moderate confidence, some claims rely on research findings only.
  0.0-0.5  Weak or conflicting evidence.

MINIMUM: at least one citation from a live tool (not crewai_research alone).
"""

# ---------------------------------------------------------------------------
# SynthesisAgent
# ---------------------------------------------------------------------------


class SynthesisAgent:
    """
    Manages an ADK LlmAgent with MCPToolset connections to all three tool servers.

    Lifecycle:
      agent = SynthesisAgent()
      await agent.initialize()          # call once at startup
      report = await agent.synthesize(...)
      await agent.close()               # call on shutdown
    """

    def __init__(self) -> None:
        self._runner: Runner | None = None
        self._session_service: InMemorySessionService | None = None
        self._exit_stack = AsyncExitStack()
        self._connected_tools: list[str] = []   # names of tool servers that connected OK
        self._all_tools: list[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Connect to all three MCP tool servers and build the ADK runner.

        Servers that fail to connect are skipped — the agent continues with
        whatever subset is available.  If all three fail, the agent falls
        back to the research-findings-only path in synthesize().
        """
        log.info("synthesis_agent.initialize.start")

        await self._exit_stack.__aenter__()

        # Try each tool server independently.
        for name, url in [
            ("yfinance", _YFINANCE_URL),
            ("wikipedia", _WIKIPEDIA_URL),
            ("edgar", _EDGAR_URL),
        ]:
            try:
                toolset = MCPToolset(
                    connection_params=SseServerParams(url=url),
                )
                tools = await self._exit_stack.enter_async_context(toolset)
                self._all_tools.extend(tools)
                self._connected_tools.append(name)
                log.info("synthesis_agent.mcp.connected", tool_server=name, url=url)
            except Exception as exc:
                log.warning(
                    "synthesis_agent.mcp.connection_failed",
                    tool_server=name,
                    url=url,
                    error=str(exc),
                )

        if not self._connected_tools:
            log.warning(
                "synthesis_agent.mcp.all_failed",
                message="All MCP tool servers unreachable. Will synthesise from research findings only.",
            )

        # Build the ADK agent with whatever tools connected.
        adk_agent = LlmAgent(
            model="gemini-2.0-flash",
            name="synthesis_agent",
            description="Synthesises research findings into a cited intelligence report.",
            instruction=_SYNTHESIS_PROMPT,
            tools=self._all_tools,
        )

        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=adk_agent,
            app_name="agentmesh-synthesis",
            session_service=self._session_service,
        )

        log.info(
            "synthesis_agent.initialize.done",
            connected_tools=self._connected_tools,
            tool_count=len(self._all_tools),
        )

    async def close(self) -> None:
        """Clean up all MCP connections."""
        await self._exit_stack.__aexit__(None, None, None)
        log.info("synthesis_agent.closed")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        query: str,
        findings: list[ResearchFindings],
        run_id: str,
        trace_id: str,
    ) -> SynthesisReport:
        """
        Run the ADK agent over the provided findings and return a SynthesisReport.

        Tracks every MCP tool call emitted by the runner to build citations.
        Falls back to a research-only report if no tools are connected or
        the agent fails to produce valid JSON.
        """
        bound_log = log.bind(trace_id=trace_id, run_id=run_id)
        bound_log.info(
            "synthesis_agent.synthesize.start",
            finding_count=len(findings),
            connected_tools=self._connected_tools,
        )

        if self._runner is None or self._session_service is None:
            raise RuntimeError("SynthesisAgent.initialize() must be called before synthesize().")

        # Build the user prompt — paste all findings inline.
        user_prompt = self._build_prompt(query, findings)

        # Create a fresh session for this run (each run is isolated).
        session = await self._session_service.create_session(
            app_name="agentmesh-synthesis",
            user_id=run_id,
        )

        tool_calls_made: list[dict[str, Any]] = []
        final_text: str = ""

        try:
            async for event in self._runner.run_async(
                user_id=run_id,
                session_id=session.id,
                new_message=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_prompt)],
                ),
            ):
                # Collect tool calls for citation building.
                fn_calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
                for fn_call in fn_calls:
                    tool_calls_made.append(
                        {
                            "tool_name": fn_call.name,
                            "args": dict(fn_call.args) if fn_call.args else {},
                        }
                    )
                    bound_log.debug(
                        "synthesis_agent.tool_call",
                        tool=fn_call.name,
                        args=fn_call.args,
                    )

                # Capture the final text response.
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_text = "".join(
                            p.text for p in event.content.parts if hasattr(p, "text") and p.text
                        )

        except Exception as exc:
            bound_log.error("synthesis_agent.runner.error", error=str(exc))
            # Fall back to a report built from research findings alone.
            return self._fallback_report(query, findings, run_id, str(exc))

        bound_log.info(
            "synthesis_agent.run.complete",
            tool_calls=len(tool_calls_made),
            response_chars=len(final_text),
        )

        # Parse the JSON block from the final text response.
        report = self._parse_response(
            text=final_text,
            findings=findings,
            run_id=run_id,
            tool_calls=tool_calls_made,
            bound_log=bound_log,
        )

        bound_log.info(
            "synthesis_agent.synthesize.done",
            report_id=report.report_id,
            citations=len(report.citations),
            mcp_tools_called=report.mcp_tools_called,
        )
        return report

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, findings: list[ResearchFindings]) -> str:
        """Render the user message with the original query and all findings inlined."""
        sections = [f"## Original Research Query\n{query}\n"]

        for i, f in enumerate(findings, 1):
            sources_text = "\n".join(
                f"  - [{s.title}]({s.url or 'n/a'}): {s.snippet}"
                for s in f.sources
            )
            sections.append(
                f"## Research Finding {i} (sub_task_id: {f.sub_task_id})\n"
                f"**Fact check passed:** {f.fact_check_passed} | "
                f"**Confidence:** {f.confidence_score}\n\n"
                f"{f.findings}\n\n"
                f"**Sources:**\n{sources_text or '  (none provided)'}\n"
            )

        sections.append(
            "\n---\nNow enrich these findings with your MCP tools and produce "
            "the final JSON report as specified in your instructions."
        )

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        text: str,
        findings: list[ResearchFindings],
        run_id: str,
        tool_calls: list[dict[str, Any]],
        bound_log: Any,
    ) -> SynthesisReport:
        """
        Extract the JSON block from the agent's final response and build SynthesisReport.

        Three-level fallback:
          1. Parse ```json ... ``` block.
          2. Regex-find first {...} block.
          3. Return a report built entirely from the raw text + tool_calls.
        """
        # Level 1: extract fenced JSON block.
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        raw_json: dict[str, Any] | None = None

        if json_match:
            try:
                raw_json = json.loads(json_match.group(1))
                bound_log.debug("synthesis.parse.via_fenced_json")
            except json.JSONDecodeError as exc:
                bound_log.warning("synthesis.parse.fenced_json_invalid", error=str(exc))

        # Level 2: find first {...} block in raw text.
        if raw_json is None:
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    raw_json = json.loads(obj_match.group())
                    bound_log.debug("synthesis.parse.via_regex_json")
                except json.JSONDecodeError as exc:
                    bound_log.warning("synthesis.parse.regex_json_invalid", error=str(exc))

        # Level 3: build from raw text + tool_calls.
        if raw_json is None:
            bound_log.warning("synthesis.parse.falling_back_to_raw_text")
            return self._build_report_from_text(text, findings, run_id, tool_calls)

        return self._build_report_from_json(raw_json, findings, run_id, tool_calls)

    def _build_report_from_json(
        self,
        raw: dict[str, Any],
        findings: list[ResearchFindings],
        run_id: str,
        tool_calls: list[dict[str, Any]],
    ) -> SynthesisReport:
        """Construct SynthesisReport from a successfully parsed JSON dict."""
        # Parse citations from JSON — supplement with any tool calls not already cited.
        citations: list[Citation] = []
        for c in raw.get("citations", []):
            tool_used = c.get("tool_used", "crewai_research")
            if tool_used not in ("yfinance", "wikipedia", "edgar", "crewai_research"):
                tool_used = "crewai_research"
            citations.append(
                Citation(
                    source_title=c.get("source_title", "Unknown"),
                    source_url=c.get("source_url") or None,
                    claim=c.get("claim", ""),
                    tool_used=tool_used,  # type: ignore[arg-type]
                )
            )

        # Guarantee at_least_one_citation — inject tool-call citations if JSON had none.
        citations = self._ensure_citations(citations, tool_calls, findings)

        confidence_scores: dict[str, float] = {}
        for k, v in raw.get("confidence_scores", {}).items():
            try:
                confidence_scores[k] = float(v)
            except (TypeError, ValueError):
                pass
        if not confidence_scores:
            confidence_scores["overall"] = 0.7

        return SynthesisReport(
            run_id=run_id,
            narrative=raw.get("narrative", "Report generation produced no narrative."),
            citations=citations,
            confidence_scores=confidence_scores,
            recommended_actions=raw.get("recommended_actions", []),
            mcp_tools_called=list({c["tool_name"] for c in tool_calls}),
        )

    def _build_report_from_text(
        self,
        text: str,
        findings: list[ResearchFindings],
        run_id: str,
        tool_calls: list[dict[str, Any]],
    ) -> SynthesisReport:
        """Last-resort: use raw agent text as the narrative, build citations from tool calls."""
        citations = self._ensure_citations([], tool_calls, findings)
        narrative = text.strip() or "The synthesis agent returned an empty response."

        return SynthesisReport(
            run_id=run_id,
            narrative=narrative,
            citations=citations,
            confidence_scores={"overall": 0.5},
            recommended_actions=["Review raw agent output — structured parsing failed."],
            mcp_tools_called=list({c["tool_name"] for c in tool_calls}),
        )

    def _ensure_citations(
        self,
        existing: list[Citation],
        tool_calls: list[dict[str, Any]],
        findings: list[ResearchFindings],
    ) -> list[Citation]:
        """
        Guarantee at least one citation exists.

        Priority:
          1. Keep valid citations already in `existing`.
          2. Inject one citation per unique MCP tool call recorded by the runner.
          3. If still empty, inject one citation per ResearchFinding source.
          4. Absolute fallback: a single [crewai_research] citation.
        """
        citations = list(existing)

        # Supplement from runner tool calls.
        seen_tools: set[str] = {c.tool_used for c in citations}
        _tool_map = {"get_stock_price": "yfinance", "get_financials": "yfinance",
                     "search": "wikipedia", "get_summary": "wikipedia",
                     "search_filings": "edgar", "get_10k": "edgar"}
        for tc in tool_calls:
            tool_name = tc["tool_name"]
            tool_used = _tool_map.get(tool_name, "crewai_research")
            if tool_used not in seen_tools:
                ticker = tc.get("args", {}).get("ticker", "") or tc.get("args", {}).get("query", "")
                citations.append(Citation(
                    source_title=f"{tool_name}({ticker})" if ticker else tool_name,
                    source_url=None,
                    claim=f"Live data retrieved via {tool_name}",
                    tool_used=tool_used,  # type: ignore[arg-type]
                ))
                seen_tools.add(tool_used)

        # Supplement from research finding sources.
        if not citations:
            for f in findings:
                for s in f.sources[:2]:
                    citations.append(Citation(
                        source_title=s.title,
                        source_url=s.url,
                        claim=f"Research finding for sub-task {f.sub_task_id}",
                        tool_used="crewai_research",
                    ))
                    if citations:
                        break

        # Absolute fallback.
        if not citations:
            citations.append(Citation(
                source_title="Research findings (fallback)",
                source_url=None,
                claim="Synthesised from provided research findings",
                tool_used="crewai_research",
            ))

        return citations

    # ------------------------------------------------------------------
    # Fallback report (all MCP tools failed or runner crashed)
    # ------------------------------------------------------------------

    def _fallback_report(
        self,
        query: str,
        findings: list[ResearchFindings],
        run_id: str,
        error: str,
    ) -> SynthesisReport:
        """
        Build a best-effort report from research findings alone.

        Called when the ADK runner raises an unhandled exception or when
        all MCP tool servers were unreachable at initialization.
        """
        log.warning("synthesis_agent.fallback_report", run_id=run_id, error=error)

        combined = "\n\n".join(
            f"**{f.sub_task_id}** (confidence {f.confidence_score}):\n{f.findings}"
            for f in findings
        )
        narrative = (
            f"## Synthesis Report (degraded — MCP tools unavailable)\n\n"
            f"**Original query:** {query}\n\n"
            f"{combined}\n\n"
            f"*Note: Live data enrichment was unavailable ({error}). "
            "This report is based on research findings only.*"
        )

        citations = self._ensure_citations([], [], findings)

        return SynthesisReport(
            run_id=run_id,
            narrative=narrative,
            citations=citations,
            confidence_scores={"overall": 0.5, "degraded": 1.0},
            recommended_actions=[
                "Re-run with MCP tool servers online for live data enrichment.",
                "Review research findings directly for primary source access.",
            ],
            mcp_tools_called=[],
        )
