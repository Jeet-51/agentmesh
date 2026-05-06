import { useState, useEffect, useCallback, useRef } from 'react'
import type { AppView, RunState, SSEEvent, SubTask, HistoryItem } from './types'
import { submitQuery, approveCheckpoint, getRunStatus, connectSSE } from './api'
import QueryInput from './components/QueryInput'
import HumanCheckpoint from './components/HumanCheckpoint'
import AgentTrace from './components/AgentTrace'
import ReportView from './components/ReportView'
import QueryHistory from './components/QueryHistory'
import NeuralBackground from './components/NeuralBackground'
import EvalDashboard from './components/EvalDashboard'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function initialState(): RunState {
  return {
    runId: '',
    traceId: '',
    query: '',
    status: 'idle',
    subTasks: [],
    events: [],
    report: null,
    partialResult: null,
    error: null,
  }
}

function loadHistory(): HistoryItem[] {
  try {
    return JSON.parse(localStorage.getItem('agentmesh_history') || '[]')
  } catch {
    return []
  }
}

function avgConfidence(scores: Record<string, number>): number {
  const vals = Object.values(scores)
  if (!vals.length) return 0
  return vals.reduce((a, b) => a + b, 0) / vals.length
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [run, setRun] = useState<RunState>(initialState)
  const [checkpointLoading, setCheckpointLoading] = useState(false)
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (localStorage.getItem('agentmesh_theme') as 'dark' | 'light') || 'dark'
  )
  const [history, setHistory] = useState<HistoryItem[]>(loadHistory)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [evalOpen, setEvalOpen] = useState(false)
  const sseRef = useRef<EventSource | null>(null)
  const savedRunIdRef = useRef<string>('')

  // Persist theme
  useEffect(() => {
    localStorage.setItem('agentmesh_theme', theme)
  }, [theme])

  // Close SSE on unmount
  useEffect(() => {
    return () => { sseRef.current?.close() }
  }, [])

  // Save completed runs to history
  useEffect(() => {
    if (
      (run.status !== 'completed' && run.status !== 'partial') ||
      !run.report ||
      !run.runId ||
      run.runId === savedRunIdRef.current
    ) return

    savedRunIdRef.current = run.runId

    const newItem: HistoryItem = {
      id: run.runId,
      query: run.query,
      timestamp: new Date().toISOString(),
      confidence: avgConfidence(run.report.confidence_scores),
      report: run.report,
      status: run.status as 'completed' | 'partial',
    }

    setHistory(prev => {
      const updated = [newItem, ...prev.filter(h => h.id !== newItem.id)].slice(0, 20)
      localStorage.setItem('agentmesh_history', JSON.stringify(updated))
      return updated
    })
  }, [run.status, run.report, run.runId, run.query])

  // ─── SSE handler ────────────────────────────────────────────────────────────

  const handleSSEEvent = useCallback((evt: SSEEvent) => {
    if (evt.event_type === 'ping') return

    setRun(prev => {
      const next: RunState = { ...prev, events: [...prev.events, evt] }

      switch (evt.event_type) {
        case 'decomposed':
          if (Array.isArray(evt.data.sub_tasks)) {
            next.subTasks = evt.data.sub_tasks as SubTask[]
          }
          break
        case 'awaiting_human':
          next.status = 'awaiting_human'
          if (Array.isArray(evt.data.sub_tasks) && next.subTasks.length === 0) {
            next.subTasks = evt.data.sub_tasks as SubTask[]
          }
          break
        case 'dispatched':
          next.status = 'running'
          break
        case 'merged': {
          next.status = 'completed'
          const fr = evt.data.final_report
          if (fr && typeof fr === 'object') next.report = fr as RunState['report']
          break
        }
        case 'partial': {
          next.status = 'partial'
          const pr = evt.data.partial_result
          if (pr && typeof pr === 'object') next.partialResult = pr as RunState['partialResult']
          break
        }
        case 'failed':
          next.status = 'failed'
          if (typeof evt.data.reason === 'string') next.error = evt.data.reason
          break
      }

      return next
    })
  }, [])

  // Poll once if report wasn't in the merged event
  useEffect(() => {
    if (run.status !== 'completed' && run.status !== 'partial') return
    if (run.report !== null) return

    let cancelled = false
    getRunStatus(run.runId)
      .then(data => {
        if (cancelled) return
        setRun(prev => ({
          ...prev,
          report: data.final_report ?? prev.report,
          partialResult: data.partial_result ?? prev.partialResult,
        }))
      })
      .catch(() => {})

    return () => { cancelled = true }
  }, [run.status, run.runId, run.report])

  // ─── Handlers ───────────────────────────────────────────────────────────────

  async function handleSubmit(query: string) {
    setRun({ ...initialState(), query, status: 'loading' })
    savedRunIdRef.current = ''
    try {
      const { run_id, trace_id } = await submitQuery(query)
      setRun(prev => ({ ...prev, runId: run_id, traceId: trace_id }))
      sseRef.current?.close()
      sseRef.current = connectSSE(run_id, handleSSEEvent, () => {
        setRun(prev =>
          prev.status === 'completed' || prev.status === 'partial' || prev.status === 'failed'
            ? prev
            : { ...prev, error: 'SSE connection lost' }
        )
      })
    } catch (err) {
      setRun(prev => ({
        ...prev,
        status: 'failed',
        error: err instanceof Error ? err.message : 'Unknown error',
      }))
    }
  }

  async function handleApprove(edited: SubTask[]) {
    setCheckpointLoading(true)
    try {
      await approveCheckpoint(run.runId, true, edited)
      setRun(prev => ({ ...prev, status: 'running', subTasks: edited }))
    } catch (err) {
      setRun(prev => ({ ...prev, error: err instanceof Error ? err.message : 'Approval failed' }))
    } finally {
      setCheckpointLoading(false)
    }
  }

  async function handleReject(feedback: string) {
    setCheckpointLoading(true)
    try {
      await approveCheckpoint(run.runId, false, null, feedback)
      setRun(prev => ({ ...prev, status: 'loading', subTasks: [] }))
    } catch (err) {
      setRun(prev => ({ ...prev, error: err instanceof Error ? err.message : 'Rejection failed' }))
    } finally {
      setCheckpointLoading(false)
    }
  }

  function handleReset() {
    sseRef.current?.close()
    sseRef.current = null
    setRun(initialState())
  }

  function handleHistorySelect(item: HistoryItem) {
    sseRef.current?.close()
    sseRef.current = null
    setRun({
      ...initialState(),
      runId: item.id,
      query: item.query,
      status: item.status,
      report: item.report,
    })
  }

  function handleHistoryDelete(id: string) {
    setHistory(prev => {
      const updated = prev.filter(h => h.id !== id)
      localStorage.setItem('agentmesh_history', JSON.stringify(updated))
      return updated
    })
  }

  function handleHistoryClear() {
    setHistory([])
    localStorage.removeItem('agentmesh_history')
  }

  // ─── View routing ────────────────────────────────────────────────────────────

  const view: AppView = run.status

  return (
    <div className={`app-root theme-${theme}`}>

      {/* Neural mesh background — draws its own bg fill, sits below all UI */}
      <NeuralBackground theme={theme} />

      {/* Fixed overlay — History (left) */}
      <div className="fixed top-4 left-4 z-30 no-print">
        <button
          onClick={() => setHistoryOpen(true)}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition-all"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-strong)',
            color: 'var(--text-secondary)',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-hover)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'var(--bg-surface)')}
        >
          <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          History
          {history.length > 0 && (
            <span style={{
              background: 'var(--accent)',
              color: '#fff',
              fontSize: '9px',
              fontWeight: 700,
              padding: '1px 6px',
              borderRadius: '99px',
            }}>
              {history.length}
            </span>
          )}
        </button>
      </div>

      {/* Fixed overlay — Eval + Theme toggle (right) */}
      <div className="fixed top-4 right-4 z-30 no-print" style={{ display: 'flex', gap: 8 }}>
        {/* Eval button */}
        <button
          onClick={() => setEvalOpen(true)}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition-all"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-strong)',
            color: 'var(--text-secondary)',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-hover)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'var(--bg-surface)')}
          title="Open Eval Dashboard"
        >
          <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <polyline strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          Evals
        </button>

        {/* Theme toggle */}
        <button
          onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
          className="w-9 h-9 flex items-center justify-center rounded-xl transition-all"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-strong)',
            color: 'var(--text-secondary)',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-hover)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'var(--bg-surface)')}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? (
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd"
                d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
                clipRule="evenodd" />
            </svg>
          ) : (
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
              <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
            </svg>
          )}
        </button>
      </div>

      {/* Main content — sits above the canvas */}
      <div style={{ position: 'relative', zIndex: 1 }}>
        {(view === 'idle' || view === 'loading') && (
          <QueryInput
            onSubmit={handleSubmit}
            loading={view === 'loading'}
            error={run.error}
          />
        )}

        {view === 'awaiting_human' && (
          <HumanCheckpoint
            runId={run.runId}
            subTasks={run.subTasks}
            onApprove={handleApprove}
            onReject={handleReject}
            loading={checkpointLoading}
          />
        )}

        {view === 'running' && (
          <AgentTrace events={run.events} query={run.query} />
        )}

        {(view === 'completed' || view === 'partial' || view === 'failed') && (
          <ReportView
            query={run.query}
            report={run.report}
            partialResult={run.partialResult}
            status={view as 'completed' | 'partial' | 'failed'}
            onReset={handleReset}
          />
        )}
      </div>

      {/* History sidebar */}
      <QueryHistory
        history={history}
        onSelect={handleHistorySelect}
        onDelete={handleHistoryDelete}
        onClearAll={handleHistoryClear}
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
      />

      {/* Eval Dashboard overlay */}
      {evalOpen && <EvalDashboard onClose={() => setEvalOpen(false)} />}
    </div>
  )
}
