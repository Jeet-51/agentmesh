"""
CrewAI research crew for AgentMesh.

Three-agent sequential crew:
  Searcher → Fact-checker → Summarizer

All agents use Gemini Flash Lite via LiteLLM ("gemini/gemini-2.5-flash").
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
from crewai.tools import tool as crewai_tool
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
        model="gemini/gemini-2.5-flash",
        temperature=0.1,
        max_tokens=4096,
    )


def _build_search_tool() -> Any:
    """
    Return the best available search tool as a CrewAI-native BaseTool.

    CrewAI 1.x requires tools to be instances of crewai.tools.BaseTool.
    LangChain tools (TavilySearchResults, DuckDuckGoSearchRun) are no longer
    accepted directly. We use the @crewai_tool decorator to wrap the underlying
    libraries so CrewAI gets a compatible tool object.

    Priority:
      1. Tavily if TAVILY_API_KEY is set (higher quality results).
      2. DuckDuckGo as a zero-config fallback.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from tavily import TavilyClient as _TavilyClient
            _tc = _TavilyClient(api_key=tavily_key)

            @crewai_tool("Tavily Web Search")
            def tavily_search(query: str) -> str:
                """Search the web for current, accurate information. Input is a search query string."""
                resp = _tc.search(query, max_results=6)
                parts = []
                for r in resp.get("results", []):
                    parts.append(
                        f"[{r.get('title', 'No title')}]({r.get('url', '')}): "
                        f"{r.get('content', '')[:400]}"
                    )
                return "\n\n".join(parts) or "No results found."

            log.info("research.tool.selected", tool="tavily")
            return tavily_search
        except Exception as exc:
            log.warning("research.tool.tavily_failed", reason=str(exc))

    # DuckDuckGo fallback — duckduckgo-search 6+ uses DDGS context manager.
    try:
        from duckduckgo_search import DDGS as _DDGS

        @crewai_tool("DuckDuckGo Web Search")
        def ddg_search(query: str) -> str:
            """Search the web for information. Input is a search query string."""
            with _DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=6))
            parts = []
            for r in results:
                parts.append(
                    f"[{r.get('title', 'No title')}]({r.get('href', '')}): "
                    f"{r.get('body', '')[:400]}"
                )
            return "\n\n".join(parts) or "No results found."

        log.info("research.tool.selected", tool="duckduckgo")
        return ddg_search
    except ImportError as exc:
        raise ImportError(
            "No search tool available. Install duckduckgo-search or set TAVILY_API_KEY."
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
                "Review claims from the Searcher and assess their credibility. "
                "Mark fact_check_passed=True if claims come from real search results "
                "and are plausible given the sources — even if only one source covers them. "
                "Only mark fact_check_passed=False when sources directly contradict each other "
                "or when a claim has zero source support."
            ),
            backstory=(
                "You are a pragmatic fact-checker who distinguishes between 'unverified' "
                "and 'false'. A claim sourced from a credible outlet is considered verified "
                "even if only one source covers it. You flag contradictions and fabrications, "
                "not merely gaps in cross-referencing."
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
                "- Run at least 2-3 searches with different query terms.\n"
                "- For EVERY source found, you MUST record it in EXACTLY this format:\n"
                "  SOURCE: [Title](https://full-url-here.com) | SNIPPET: key fact or excerpt\n"
                "- The URL inside the parentheses MUST be the complete https:// URL as returned "
                "by the search tool — copy it character for character.\n"
                "- Aim for 5-8 distinct, credible sources with real URLs.\n"
                "- Never omit or shorten URLs — partial URLs are useless.\n"
                "- Include publication dates where visible."
            ),
            expected_output=(
                "A numbered list of research findings. Each item MUST follow this format:\n"
                "N. SOURCE: [Title](https://exact-url.com) | SNIPPET: key fact (max 200 chars)\n\n"
                "Rules:\n"
                "- Every entry must have a complete https:// URL in the parentheses.\n"
                "- Minimum 5 sources. No placeholder URLs.\n"
                "- Copy URLs verbatim from search tool output."
            ),
            agent=searcher,
        )

        fact_check_task = Task(
            description=(
                "Review the Searcher's results and assess credibility of claims.\n\n"
                "Topic: {topic}\n\n"
                "CRITICAL: You MUST preserve every SOURCE line from the Searcher's output "
                "verbatim — including the full URLs — in your report. Do not shorten or omit URLs.\n\n"
                "For each major claim:\n"
                "  1. Is it sourced from a real search result with a URL? If yes, it is credible.\n"
                "  2. Does any other source directly CONTRADICT it? If yes, flag it.\n"
                "  3. Is it a completely unsourced assertion? If yes, flag it.\n\n"
                "IMPORTANT: Do NOT mark claims as unverified simply because only one source covers them.\n"
                "Single-source claims from credible outlets are acceptable.\n\n"
                "Conclude with:\n"
                "  FACT_CHECK_PASSED: true   (claims are sourced and no direct contradictions)\n"
                "  FACT_CHECK_PASSED: false  (claims are directly contradicted or entirely unsourced)\n"
            ),
            expected_output=(
                "A fact-check report with these sections:\n"
                "SOURCES REVIEWED: Repeat ALL source lines from the Searcher with their full URLs.\n"
                "CREDIBLE CLAIMS: bullet list of sourced, plausible facts.\n"
                "QUESTIONABLE CLAIMS: bullet list of directly contradicted or unsourced claims.\n"
                "FACT_CHECK_PASSED: true or false with a 1-sentence justification."
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
                '    {{"title": "<outlet or page name>", "url": "<EXACT URL copied from the search result — this is MANDATORY, never leave blank or use a placeholder; copy the full https:// URL verbatim from the Searcher output>", "snippet": "<key excerpt, max 200 chars>"}}\n'
                "  ]\n"
                "}}\n\n"
                "URL RULES — strictly enforced:\n"
                "  - Every source MUST have a complete URL starting with https://\n"
                "  - Copy URLs EXACTLY as they appeared in the Searcher's markdown links: [Title](URL)\n"
                "  - NEVER use empty string, null, 'N/A', 'URL', or any placeholder\n"
                "  - NEVER invent or fabricate URLs\n"
                "  - If a source has no URL: use the outlet's root domain e.g. https://reuters.com\n"
                "  - Tavily results always include URLs — look for them in the Searcher's output\n\n"
                "confidence_score bands — base this on EVIDENCE QUALITY, not fact-check result alone:\n"
                "  0.85–1.0  Multiple high-quality sources, specific figures, strong consensus.\n"
                "  0.70–0.85 Good evidence from 3+ credible outlets.\n"
                "  0.55–0.70 Findings from 1-2 credible sources, plausible.\n"
                "  0.0–0.55  Contradictory or very thin evidence.\n\n"
                "IMPORTANT: fact_check_passed=false should reduce confidence by AT MOST 0.15.\n"
                "Include 3-6 sources. Every source MUST have a real title and a real https:// URL."
            ),
            expected_output=(
                "A single valid JSON object with keys: findings (str), "
                "confidence_score (float), fact_check_passed (bool), "
                "sources (list of {title, url, snippet}). "
                "Every source must have a non-empty title and a complete https:// URL "
                "copied verbatim from the search results."
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

        # Collect raw task outputs for trace visibility and URL fallback (first 3 tasks).
        tasks_output = getattr(crew_result, "tasks_output", []) or []
        raw_search_results = [str(t.raw) for t in tasks_output if getattr(t, "raw", None)][:3]

        # Convert sources dicts → Source models (with URL validation).
        sources = []
        for s in summary.sources:
            raw_title   = s.get("title", "").strip()
            raw_url     = (s.get("url") or "").strip()
            raw_snippet = s.get("snippet", "").strip()

            # Skip completely empty entries
            if not raw_title and not raw_url and not raw_snippet:
                continue

            title = raw_title if raw_title else "Web Source"
            url   = self._validate_url(raw_url)
            sources.append(Source(url=url, title=title, snippet=raw_snippet))

        # ── Fallback URL extraction ──────────────────────────────────────────
        # If fewer than 2 sources carry a valid URL the LLM dropped them; recover
        # them from the raw Searcher / Fact-checker output using regex.
        valid_url_count = sum(1 for s in sources if s.url)
        if valid_url_count < 2 and raw_search_results:
            self._bound_log.info(
                "crew.parse.url_fallback",
                valid_before=valid_url_count,
                raw_tasks=len(raw_search_results),
            )
            extracted = self._extract_urls_from_raw(raw_search_results)
            existing_urls = {s.url for s in sources if s.url}
            existing_titles = {s.title.lower() for s in sources}

            for item in extracted:
                url = item["url"]
                title = item["title"]
                if url in existing_urls:
                    # Back-fill the URL into the matching URL-less source if titles align
                    for src in sources:
                        if src.url is None and src.title.lower() in title.lower():
                            src.url = url
                            existing_urls.add(url)
                            break
                    continue
                # Add as a new source
                sources.append(Source(url=url, title=title, snippet=""))
                existing_urls.add(url)

            new_valid = sum(1 for s in sources if s.url)
            self._bound_log.info(
                "crew.parse.url_fallback.done",
                valid_after=new_valid,
                added=new_valid - valid_url_count,
            )

        # Confidence floor: fact_check_passed=False should cost at most 0.15.
        confidence = summary.confidence_score
        if not summary.fact_check_passed and confidence < 0.55:
            confidence = 0.55  # floor — poor fact-check alone can't drop below this

        return ResearchFindings(
            sub_task_id=self.sub_task.sub_task_id,
            findings=summary.findings,
            sources=sources,
            confidence_score=confidence,
            fact_check_passed=summary.fact_check_passed,
            raw_search_results=raw_search_results,
        )

    @staticmethod
    def _validate_url(raw: str) -> str | None:
        """
        Return a cleaned, usable URL string or None.

        Rejects placeholders, empty strings, and non-HTTP values.
        Prepends https:// to bare domains if needed.
        """
        if not raw:
            return None
        url = raw.strip().strip('"\'')
        _invalid = {"", "null", "none", "n/a", "url", "na", "#", "undefined", "http", "https"}
        if url.lower() in _invalid:
            return None
        if url.startswith("https://") or url.startswith("http://"):
            # Must have at least a domain part
            return url if len(url) > 10 else None
        # Bare domain — prepend https://
        if "." in url and "/" in url or ("." in url and not url.startswith("/")):
            return f"https://{url}"
        return None

    @staticmethod
    def _extract_urls_from_raw(raw_texts: list[str]) -> list[dict[str, str]]:
        """
        Fallback: pull [Title](URL) markdown links from raw searcher/fact-checker output.

        Returns a deduplicated list of {title, url} dicts with validated URLs.
        """
        combined = "\n".join(raw_texts)
        # Match markdown-style links: [Title](https://...)
        link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)\s]{8,})\)")
        # Also match bare URLs adjacent to SOURCE: lines
        bare_pattern = re.compile(r"SOURCE:\s*\[([^\]]+)\]\((https?://[^\)\s]{8,})\)")

        seen_urls: set[str] = set()
        results: list[dict[str, str]] = []

        for pattern in (bare_pattern, link_pattern):
            for m in pattern.finditer(combined):
                title, url = m.group(1).strip(), m.group(2).strip()
                if url not in seen_urls and len(url) > 10:
                    seen_urls.add(url)
                    results.append({"title": title, "url": url})

        return results

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
