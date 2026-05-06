import type { SSEEvent, SSEEventType, SubTask, SynthesisReport, PartialResult } from './types'

const BASE = (import.meta.env.VITE_GATEWAY_URL as string | undefined) ?? 'http://localhost:8000'

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText })) as { detail?: string }
    throw new Error(body.detail ?? `HTTP ${resp.status}`)
  }
  return resp.json() as Promise<T>
}

// ─── Run endpoints ───────────────────────────────────────────────────────────

export interface StartRunResponse {
  run_id: string
  trace_id: string
  status: string
  message: string
}

export function submitQuery(query: string): Promise<StartRunResponse> {
  return request<StartRunResponse>('/run', {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
}

export interface RunStatusResponse {
  run_id: string
  trace_id: string
  status: string
  query: string
  sub_tasks: SubTask[] | null
  findings_count: number
  final_report: SynthesisReport | null
  partial_result: PartialResult | null
}

export function getRunStatus(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/run/${runId}`)
}

// ─── Checkpoint endpoints ─────────────────────────────────────────────────────

export function approveCheckpoint(
  runId: string,
  approved: true,
  subTasks: SubTask[],
): Promise<void>
export function approveCheckpoint(
  runId: string,
  approved: false,
  subTasks: null,
  feedback: string,
): Promise<void>
export function approveCheckpoint(
  runId: string,
  approved: boolean,
  subTasks: SubTask[] | null,
  feedback?: string,
): Promise<void> {
  return request<void>(`/checkpoint/${runId}/approve`, {
    method: 'POST',
    body: JSON.stringify({
      approved,
      sub_tasks: subTasks ?? undefined,
      feedback: feedback ?? undefined,
    }),
  })
}

// ─── SSE stream ──────────────────────────────────────────────────────────────

const SSE_EVENT_TYPES: SSEEventType[] = [
  'started', 'decomposed', 'awaiting_human', 'human_responded',
  'dispatched', 'research_done', 'synthesis_done',
  'merged', 'partial', 'failed', 'ping',
]

export function connectSSE(
  runId: string,
  onEvent: (e: SSEEvent) => void,
  onError?: (e: Event) => void,
): EventSource {
  const es = new EventSource(`${BASE}/run/${runId}/stream`)

  const handler = (raw: MessageEvent) => {
    console.log('[SSE]', raw.type, raw.data)
    try {
      const parsed = JSON.parse(raw.data as string) as SSEEvent
      // Use the named SSE event type as the canonical event_type so the
      // switch in App.tsx always receives the correct value even if the
      // gateway JSON payload omits or mis-spells the field.
      const evt: SSEEvent = {
        ...parsed,
        event_type: (parsed.event_type || raw.type) as SSEEventType,
      }
      onEvent(evt)
    } catch {
      // silently drop malformed frames
    }
  }

  // Register a listener for every named event type the gateway emits.
  for (const type of SSE_EVENT_TYPES) {
    es.addEventListener(type, handler)
  }

  if (onError) {
    es.onerror = onError
  }

  return es
}
