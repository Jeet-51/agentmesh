// ─── Sub-task ────────────────────────────────────────────────────────────────

export type SubTaskStatus =
  | 'pending'
  | 'dispatched'
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'retrying'

export interface SubTask {
  sub_task_id: string
  topic: string
  instructions: string
  status: SubTaskStatus
  assigned_to?: string | null
  retry_count?: number
}

// ─── SSE events ──────────────────────────────────────────────────────────────

export type SSEEventType =
  | 'started'
  | 'decomposed'
  | 'awaiting_human'
  | 'human_responded'
  | 'dispatched'
  | 'research_done'
  | 'synthesis_done'
  | 'merged'
  | 'partial'
  | 'failed'
  | 'ping'

export interface SSEEvent {
  event_type: SSEEventType
  run_id: string
  trace_id: string
  timestamp: string
  data: Record<string, unknown>
}

// ─── Research ────────────────────────────────────────────────────────────────

export interface Source {
  url: string | null
  title: string
  snippet: string
}

export interface ResearchFindings {
  sub_task_id: string
  findings: string
  sources: Source[]
  confidence_score: number
  fact_check_passed: boolean
}

// ─── Synthesis ───────────────────────────────────────────────────────────────

export type ToolUsed = 'yfinance' | 'wikipedia' | 'edgar' | 'crewai_research' | 'newsapi'

export interface Citation {
  citation_id?: string
  source_url: string | null
  source_title: string
  claim: string
  tool_used: ToolUsed
}

export interface SynthesisReport {
  report_id: string
  run_id: string
  narrative: string
  citations: Citation[]
  confidence_scores: Record<string, number>
  recommended_actions: string[]
  mcp_tools_called: string[]
}

export interface PartialResult {
  run_id: string
  completed_agents: string[]
  failed_agents: string[]
  partial_report: string
  failure_reasons: Record<string, string>
}

// ─── History ─────────────────────────────────────────────────────────────────

export interface HistoryItem {
  id: string
  query: string
  timestamp: string
  confidence: number
  report: SynthesisReport
  status: 'completed' | 'partial'
}

// ─── App state ───────────────────────────────────────────────────────────────

export type AppView =
  | 'idle'
  | 'loading'
  | 'awaiting_human'
  | 'running'
  | 'completed'
  | 'partial'
  | 'failed'

export interface RunState {
  runId: string
  traceId: string
  query: string
  status: AppView
  subTasks: SubTask[]
  events: SSEEvent[]
  report: SynthesisReport | null
  partialResult: PartialResult | null
  error: string | null
}
