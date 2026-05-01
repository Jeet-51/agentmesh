"""
CrewAI research crew for AgentMesh.

Three-agent sequential crew:
  Searcher → Fact-checker → Summarizer

All agents use Gemini Flash via LiteLLM ("gemini/gemini-2.0-flash").
Search tool: Tavily if TAVILY_API_KEY is set; DuckDuckGo as automatic fallback.

The public interface is ResearchCrew.run() which returns a typed
ResearchFindings Pydantic model.  Because CrewAI's kickoff() is
synchronous, the A2A server wraps this in asyncio.to_thread().
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import structlog
from crewai import LLM, Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from shared.models import ResearchFindings, Source, SubTask

log = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Intermediate Pydantic model — produced by the Summarizer task
#
# Does NOT include sub_task_id (injected by the caller from the TaskCard).
# Kept intentionally flat so the LLM can reliably produce it as JSON.
# ---------------------------------------------------------------------------


class _CrewSummary(BaseModel):
    """Structured output the Summarizer agent is instructed to produce."""

    findings: str = Field(..., description="200-500 word narrative summarising key findings.")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    fact_check_passed: bool = Field(
        ..., description="True only if core claims are supported by 2+ independent sources."
    )
    sources: list[dict[str, str]] = Field(
        default_factory=list,
        description="[{title, url, snippet}, ...] for every source cited.",
    )


# ---------------------------------------------------------------------------
# LLM + search tool setup
# ---------------------------------------------------------------------------


def _build_llm() -> LLM:
    """
    Return a CrewAI LLM backed by Gemini Flash via LiteLLM.

    LiteLLM resolves GEMINI_API_KEY or GOOGLE_API_KEY; we set both so
    whichever convention the installed version expects will work.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY must be set for the research crew LLM.")

    os.environ.setdefault("GEMINI_API_KEY", api_key)

    return LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.1,
        max_tokens=4096,
    )


def _build_search_tool() -> Any:
    """
    Return the best available search tool.

    Priority:
      1. TavilySearchResults if TAVILY_API_KEY is set.
      2. DuckDuckGoSearchRun as a zero-config fallback.

    Both are LangChain tools; CrewAI >= 0.51 accepts them natively.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            log.info("research.tool.selected", tool="tavily")
            return TavilySearchResults(max_results=6)
        except ImportError:
            log.warning(
                "research.tool.tavily_import_failed",
                reason="langchain-community not installed; falling back to DuckDuckGo",
            )

    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        log.info("research.tool.selected", tool="duckduckgo")
        return DuckDuckGoSearchRun()
    except ImportError as exc:
        raise ImportError(
            "No search tool available. Install langchain-community for DuckDuckGo "
            "or set TAVILY_API_KEY for Tavily."
        ) from exc


# ---------------------------------------------------------------------------
# ResearchCrew
# ---------------------------------------------------------------------------


class ResearchCrew:
    """
    Encapsulates the three-agent CrewAI research crew.

    Sequential execution: Searcher → Fact-checker → Summarizer.
    Each downstream agent reads previous agents' outputs via task context.

    Usage:
        crew = ResearchCrew(sub_task=sub_task, trace_id=trace_id)
        findings = crew.run()   # synchronous; wrap with asyncio.to_thread()
    """

    def __init__(self, sub_task: SubTask, trace_id: str) -> None:
        self.sub_task = sub_task
        self.trace_id = trace_id
        self._llm = _build_llm()
        self._search_tool = _build_search_tool()
        self._bound_log = log.bind(
            trace_id=trace_id,
            sub_task_id=sub_task.sub_task_id,
            topic=sub_task.topic,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> ResearchFindings:
        """Execute the crew and return a validated ResearchFindings model."""
        self._bound_log.info("crew.run.start")

        crew = self._build_crew()

        try:
            result = crew.kickoff(
                inputs={
                    "topic": self.sub_task.topic,
                    "instructions": self.sub_task.instructions,
                }
            )
        except Exception as exc:
            self._bound_log.error("crew.kickoff.error", error=str(exc))
            raise

        findings = self._parse_output(result)

        self._bound_log.info(
            "crew.run.complete",
            confidence=findings.confidence_score,
            fact_check_passed=findings.fact_check_passed,
            source_count=len(findings.sources),
        )
        return findings

    # ------------------------------------------------------------------
    # Crew / agent / task construction
    # ------------------------------------------------------------------

    def _build_crew(self) -> Crew:
        """Construct the Crew with all agents and tasks wired up."""

        # ---- Agents --------------------------------------------------

        searcher = Agent(
            role="Senior Research Specialist",
            goal=(
                "Find comprehensive, current information about the topic using web search. "
                "Prioritise primary sources, recent publications, and authoritative outlets. "
                "Record every source URL and title."
            ),
            backstory=(
                "You are an investigative researcher who always backs claims with sources. "
                "You run multiple searches with varied query terms to ensure full coverage "
                "and never present unsupported assertions as fact."
            ),
            tools=[self._search_tool],
            llm=self._llm,
            verbose=False,
            max_iter=5,
        )

        fact_checker = Agent(
            role="Fact Verification Specialist",
            goal=(
                "Critically evaluate every major claim from the Searcher. "
                "Identify which claims are backed by multiple independent sources "
                "and which are uncertain or single-sourced. "
                "Set fact_check_passed=true only when core claims are solid."
            ),
            backstory=(
                "You are a rigorous fact-checker trained to separate verified fact "
                "from speculation. You cross-reference claims and always flag uncertainty "
                "rather than fill gaps with assumptions."
            ),
            tools=[],
            llm=self._llm,
            verbose=False,
            max_iter=2,
        )

        summarizer = Agent(
            role="Research Synthesizer",
            goal=(
                "Distil the search results and fact-check report into a structured, "
                "accurate JSON summary. Assign a confidence score that honestly reflects "
                "the quality and consistency of the evidence."
            ),
            backstory=(
                "You transform raw research into clear, well-cited summaries. "
                "You are scrupulous about accuracy and always represent uncertainty honestly."
            ),
            tools=[],
            llm=self._llm,
            verbose=False,
            max_iter=2,
        )

        # ---- Tasks ---------------------------------------------------

        search_task = Task(
            description=(
                "Research the topic below thoroughly using web search.\n\n"
                "Topic: {topic}\n"
                "Instructions: {instructions}\n\n"
                "Requirements:\n"
                "- Run at least 2 searches with different query terms.\n"
                "- For every fact, record: [Source Title](URL): key excerpt.\n"
                "- Aim for 5-8 distinct, credible sources.\n"
                "- Include publication dates where visible."
            ),
            expected_output=(
                "A structured list of findings with each item formatted as:\n"
                "[SOURCE TITLE](URL): key fact or excerpt.\n"
                "Minimum 5 sources. Include raw search snippets where available."
            ),
            agent=searcher,
        )

        fact_check_task = Task(
            description=(
                "Review the Searcher's results and verify the core claims.\n\n"
                "Topic: {topic}\n\n"
                "For each major claim:\n"
                "  1. Check whether it appears in 2+ independent sources.\n"
                "  2. Note any contradictions between sources.\n"
                "  3. Flag single-source or unverifiable claims.\n\n"
                "Conclude with:\n"
                "  FACT_CHECK_PASSED: true   (core claims supported by 2+ sources, no major contradictions)\n"
                "  FACT_CHECK_PASSED: false  (significant unverified or contradicted claims)\n"
            ),
            expected_output=(
                "A fact-check report with three sections:\n"
                "VERIFIED CLAIMS: bullet list of well-supported facts.\n"
                "UNVERIFIED CLAIMS: bullet list of weak or single-source claims.\n"
                "FACT_CHECK_PASSED: true or false with a 2-sentence justification."
            ),
            context=[search_task],
            agent=fact_checker,
        )

        summarize_task = Task(
            description=(
                "Synthesise the search results and fact-check report into a final structured summary.\n\n"
                "Topic: {topic}\n\n"
                "Return ONLY a JSON object — no markdown fences, no preamble:\n"
                "{{\n"
                '  "findings": "<200-500 word narrative of key findings, written for a non-expert>",\n'
                '  "confidence_score": <float 0.0–1.0>,\n'
                '  "fact_check_passed": <true|false>,\n'
                '  "sources": [\n'
                '    {{"title": "<source title>", "url": "<url or empty string>", "snippet": "<key excerpt, max 200 chars>"}}\n'
                "  ]\n"
                "}}\n\n"
                "confidence_score bands:\n"
                "  0.9–1.0  Strong consensus, multiple high-quality sources, fact-check passed.\n"
                "  0.7–0.9  Good evidence, minor gaps or 1-2 unverified claims.\n"
                "  0.5–0.7  Mixed evidence, notable unverified claims.\n"
                "  0.0–0.5  Weak, contradictory, or very thin evidence.\n\n"
                "Include 3-6 sources. Prefer sources the fact-checker verified."
            ),
            expected_output=(
                "A single valid JSON object with keys: findings (str), "
                "confidence_score (float), fact_check_passed (bool), "
                "sources (list of {title, url, snippet})."
            ),
            context=[search_task, fact_check_task],
            agent=summarizer,
            output_pydantic=_CrewSummary,
        )

        return Crew(
            agents=[searcher, fact_checker, summarizer],
            tasks=[search_task, fact_check_task, summarize_task],
            process=Process.sequential,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, crew_result: Any) -> ResearchFindings:
        """
        Convert CrewAI's CrewOutput into a typed ResearchFindings model.

        Strategy (most to least reliable):
          1. crew_result.pydantic  — populated when output_pydantic works.
          2. crew_result.json_dict — populated when output_json works.
          3. _extract_json_summary — regex + JSON parse from raw text.
        """
        summary: _CrewSummary | None = None

        # Path 1: output_pydantic produced a validated model.
        pydantic_out = getattr(crew_result, "pydantic", None)
        if isinstance(pydantic_out, _CrewSummary):
            summary = pydantic_out
            self._bound_log.debug("crew.parse.via_pydantic")

        # Path 2: output_json produced a dict.
        if summary is None:
            json_dict = getattr(crew_result, "json_dict", None)
            if isinstance(json_dict, dict):
                try:
                    summary = _CrewSummary.model_validate(json_dict)
                    self._bound_log.debug("crew.parse.via_json_dict")
                except Exception as exc:
                    self._bound_log.warning("crew.parse.json_dict_invalid", error=str(exc))

        # Path 3: extract JSON from the raw string output.
        if summary is None:
            raw_text = getattr(crew_result, "raw", str(crew_result))
            summary = self._extract_json_summary(str(raw_text))
            self._bound_log.debug("crew.parse.via_regex_extraction")

        # Convert sources dicts → Source models.
        sources = [
            Source(
                url=s.get("url") or None,
                title=s.get("title", "Unknown"),
                snippet=s.get("snippet", ""),
            )
            for s in summary.sources
        ]

        # Collect raw task outputs for trace visibility (first 3 only).
        tasks_output = getattr(crew_result, "tasks_output", []) or []
        raw_search_results = [str(t.raw) for t in tasks_output if getattr(t, "raw", None)][:3]

        return ResearchFindings(
            sub_task_id=self.sub_task.sub_task_id,
            findings=summary.findings,
            sources=sources,
            confidence_score=summary.confidence_score,
            fact_check_passed=summary.fact_check_passed,
            raw_search_results=raw_search_results,
        )

    def _extract_json_summary(self, text: str) -> _CrewSummary:
        """
        Pull the first JSON object out of a raw string and parse it as _CrewSummary.

        Handles model outputs wrapped in markdown code fences.
        """
        # Strip markdown fences.
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()

        # Find the outermost {...} block.
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            self._bound_log.error(
                "crew.parse.no_json_block",
                raw_preview=text[:400],
            )
            raise ValueError(
                "Crew output contained no JSON block. "
                f"Raw preview: {text[:200]!r}"
            )

        try:
            data = json.loads(match.group())
            return _CrewSummary.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            self._bound_log.error(
                "crew.parse.json_parse_error",
                error=str(exc),
                raw=match.group()[:400],
            )
            raise ValueError(f"Could not parse crew output as _CrewSummary: {exc}") from exc
