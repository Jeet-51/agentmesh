import { useEffect, useRef, useState } from 'react'
import type { SSEEvent, SSEEventType } from '../types'

interface Props {
  events: SSEEvent[]
  query: string
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

const PIPELINE = [
  { label: 'Received' },
  { label: 'Decomposed' },
  { label: 'Researching' },
  { label: 'Synthesising' },
  { label: 'Complete' },
]

function getPipelineStage(events: SSEEvent[]): number {
  const types = new Set(events.map(e => e.event_type))
  if (types.has('merged') || types.has('partial') || types.has('failed')) return 5
  if (types.has('synthesis_done')) return 4
  if (types.has('research_done'))  return 3
  if (types.has('dispatched') || types.has('human_responded')) return 2
  if (types.has('decomposed') || types.has('awaiting_human'))  return 1
  return 0
}

// ─── Agent metadata ───────────────────────────────────────────────────────────

const AGENT_MESSAGES: Record<number, string[]> = {
  0: ['Initialising orchestrator…', 'Planning research tasks…', 'Breaking down your query…'],
  1: ['Decomposing sub-tasks…', 'Analysing query scope…', 'Identifying research topics…'],
  2: ['Dispatching to CrewAI agents…', 'Spinning up research crew…', 'Allocating sub-tasks…'],
  3: ['Searching the web…', 'Fact-checking findings…', 'Summarising research…', 'Building research report…'],
  4: ['Pulling live market data…', 'Cross-referencing sources…', 'Analysing findings…', 'Writing intelligence report…'],
}
const AGENT_ICONS: Record<number, string> = { 0: '🔮', 1: '🔮', 2: '🔮', 3: '🔬', 4: '⚗️' }
const AGENT_NAMES: Record<number, string> = {
  0: 'Orchestrator', 1: 'Orchestrator', 2: 'Orchestrator',
  3: 'Research Agent', 4: 'Synthesis Agent',
}
const STAGE_DURATION: Record<number, number> = { 0: 5, 1: 5, 2: 5, 3: 150, 4: 60 }

// ─── Event display metadata ───────────────────────────────────────────────────

interface EvStyle { dot: string; badge: string; label: string }

const EV_STYLES: Partial<Record<SSEEventType, EvStyle>> = {
  started:         { dot: '#a99fff', badge: '#312a6e', label: 'Orchestrator' },
  decomposed:      { dot: '#a99fff', badge: '#312a6e', label: 'Orchestrator' },
  awaiting_human:  { dot: '#f5a623', badge: '#3a2a1a', label: 'Checkpoint'   },
  human_responded: { dot: '#22d3a0', badge: '#1a2a1a', label: 'Human'        },
  dispatched:      { dot: '#a99fff', badge: '#312a6e', label: 'Orchestrator' },
  research_done:   { dot: '#22d3a0', badge: '#1a3a2a', label: 'Research'     },
  synthesis_done:  { dot: '#d3a0ff', badge: '#2a1a3a', label: 'Synthesis'    },
  merged:          { dot: '#22d3a0', badge: '#1a3a2a', label: 'Complete'     },
  partial:         { dot: '#f5a623', badge: '#3a2a1a', label: 'Partial'      },
  failed:          { dot: '#f25c5c', badge: '#3a1a1a', label: 'Error'        },
  ping:            { dot: '#55557a', badge: '#111122', label: 'Keepalive'    },
}

const EV_LABELS: Partial<Record<SSEEventType, string>> = {
  started:         'Run accepted — orchestrator initialising',
  decomposed:      'Query decomposed into sub-tasks',
  awaiting_human:  'Waiting for human approval',
  human_responded: 'Decision received — dispatching agents',
  dispatched:      'Sub-tasks sent to research agents',
  research_done:   'Research findings received',
  synthesis_done:  'Synthesis report generated',
  merged:          'Final report assembled',
  partial:         'Partial result — some agents failed',
  failed:          'Run failed',
  ping:            'Keepalive',
}

function formatTime(iso: string) {
  try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) }
  catch { return '' }
}

// ─── Thinking dots ────────────────────────────────────────────────────────────

function ThinkingDots() {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, marginLeft: 4 }}>
      {[0, 1, 2].map(i => (
        <span key={i} className="animate-thinking-dot"
          style={{
            display: 'inline-block', width: 4, height: 4, borderRadius: '50%',
            background: 'var(--accent)', animationDelay: `${i * 0.22}s`,
          }} />
      ))}
    </span>
  )
}

// ─── Pipeline bar ─────────────────────────────────────────────────────────────

function PipelineBar({ stage }: { stage: number }) {
  const total = PIPELINE.length
  const pct   = Math.min(100, (stage / total) * 100)

  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
      borderRadius: 16, padding: '20px 24px', marginBottom: 16,
    }}>
      <div style={{ position: 'relative', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        {/* Track */}
        <div style={{
          position: 'absolute', top: 15, left: 16, right: 16, height: 1,
          background: 'var(--border-strong)',
        }} />
        {/* Filled */}
        <div style={{
          position: 'absolute', top: 15, left: 16, height: 1,
          width: `calc(${pct}% - 32px)`,
          background: 'var(--accent)',
          transition: 'width 0.6s ease',
        }} />

        {PIPELINE.map((s, i) => {
          const done   = i < stage
          const active = i === stage && stage < total
          return (
            <div key={s.label} style={{
              flex: '1 1 0', maxWidth: `${100 / total}%`,
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
              position: 'relative', zIndex: 1,
            }}>
              <div
                className={active ? 'stage-active' : ''}
                style={{
                  width: 30, height: 30, borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: done ? 13 : 11, fontWeight: 600,
                  border: done ? 'none' : `${active ? 2 : 1}px solid ${active ? 'var(--accent)' : 'var(--border-strong)'}`,
                  background: done ? 'var(--accent)' : 'var(--bg-base)',
                  color: done ? '#fff' : active ? 'var(--accent)' : 'var(--text-tertiary)',
                  transition: 'all 0.4s ease',
                }}>
                {done ? '✓' : i + 1}
              </div>
              <span style={{
                fontSize: 10, fontWeight: 500, textAlign: 'center', whiteSpace: 'nowrap',
                color: done ? 'var(--green)' : active ? 'var(--accent)' : 'var(--text-tertiary)',
              }}>
                {s.label}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Active agent card ────────────────────────────────────────────────────────

function AgentCard({ stage, stageStart }: { stage: number; stageStart: number }) {
  const [msgIdx, setMsgIdx] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const msgs     = AGENT_MESSAGES[stage] ?? AGENT_MESSAGES[0]
  const duration = STAGE_DURATION[stage] ?? 60

  useEffect(() => {
    const id = setInterval(() => setMsgIdx(i => (i + 1) % msgs.length), 3000)
    return () => clearInterval(id)
  }, [msgs.length])

  useEffect(() => {
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - stageStart) / 1000)), 1000)
    return () => clearInterval(id)
  }, [stageStart])

  const remaining = Math.max(0, duration - elapsed)
  const barPct    = Math.min(94, (elapsed / duration) * 100)

  return (
    <div className="animate-slide-up" style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-strong)',
      borderLeft: '3px solid var(--accent)',
      borderRadius: 16, padding: 20, marginBottom: 16,
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <span style={{ fontSize: 20 }}>{AGENT_ICONS[stage]}</span>
        <div style={{ flex: 1 }}>
          <p style={{ margin: 0, fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
            {AGENT_NAMES[stage]}
          </p>
          <p style={{ margin: 0, fontSize: 12, color: 'var(--text-tertiary)' }}>
            {remaining > 0 ? `~${remaining}s remaining` : 'almost done…'}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%', background: 'var(--green)',
            animation: 'pulseLive 1.5s ease-in-out infinite',
            display: 'inline-block',
          }} />
          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--green)' }}>Live</span>
        </div>
      </div>

      {/* Progress bar */}
      <div style={{
        height: 3, borderRadius: 99, background: 'var(--bg-elevated)',
        overflow: 'hidden', marginBottom: 12,
      }}>
        <div style={{
          height: '100%', borderRadius: 99,
          width: `${barPct}%`, background: 'var(--accent)',
          transition: 'width 1s ease',
        }} />
      </div>

      {/* Message */}
      <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)' }}>
        {msgs[msgIdx % msgs.length]}
        <ThinkingDots />
      </p>

      {stage === 3 && (
        <p style={{ margin: '8px 0 0', fontSize: 12, color: 'var(--text-tertiary)' }}>
          Agents run sequentially — each sub-task takes ~45s
        </p>
      )}
    </div>
  )
}

// ─── Skeleton ─────────────────────────────────────────────────────────────────

function Skeleton() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {[80, 55, 65].map((w, i) => (
        <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <div className="skeleton-pulse" style={{ width: 8, height: 8, borderRadius: '50%', marginTop: 4, flexShrink: 0 }} />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div className="skeleton-pulse" style={{ height: 11, width: '35%', borderRadius: 6 }} />
            <div className="skeleton-pulse" style={{ height: 11, width: `${w}%`, borderRadius: 6 }} />
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function AgentTrace({ events, query }: Props) {
  const bottomRef     = useRef<HTMLDivElement>(null)
  const stageStartRef = useRef<Record<number, number>>({})
  const [, forceRender] = useState(0)

  const visible    = events.filter(e => e.event_type !== 'ping')
  const stage      = getPipelineStage(events)
  const isFinished = visible.some(e =>
    e.event_type === 'merged' || e.event_type === 'partial' || e.event_type === 'failed'
  )

  useEffect(() => {
    if (stageStartRef.current[stage] === undefined) {
      stageStartRef.current[stage] = Date.now()
      forceRender(n => n + 1)
    }
  }, [stage])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  const stageStart = stageStartRef.current[stage] ?? Date.now()

  return (
    <div className="animate-fade-in" style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column',
      alignItems: 'center', padding: '48px 16px', background: 'var(--bg-base)',
    }}>
      <div style={{ width: '100%', maxWidth: 680 }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            {!isFinished && (
              <span style={{
                width: 8, height: 8, borderRadius: '50%', background: 'var(--green)',
                display: 'inline-block', animation: 'pulseLive 1.5s ease-in-out infinite',
              }} />
            )}
            <span style={{
              fontSize: 11, fontWeight: 500, letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: isFinished ? 'var(--text-tertiary)' : 'var(--green)',
            }}>
              {isFinished ? 'Processing complete' : 'AGENTS RUNNING'}
            </span>
          </div>
          <h2 style={{
            margin: '0 0 6px', fontSize: 22, fontWeight: 600,
            color: 'var(--text-primary)', lineHeight: 1.35,
          }}>
            {query}
          </h2>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)' }}>
            Estimated: ~2–4 min · LangGraph → CrewAI → Google ADK
          </p>
        </div>

        {/* ── Pipeline bar ── */}
        <PipelineBar stage={stage} />

        {/* ── Agent thinking card ── */}
        {!isFinished && stage < PIPELINE.length && (
          <AgentCard stage={stage} stageStart={stageStart} />
        )}

        {/* ── Event log ── */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
          borderRadius: 16, padding: '20px 24px',
        }}>
          <p style={{
            margin: '0 0 16px', fontSize: 11, fontWeight: 500,
            textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-tertiary)',
          }}>
            Event Log
          </p>

          <div style={{ position: 'relative' }}>
            {visible.length > 0 && (
              <div style={{
                position: 'absolute', left: 6, top: 4, bottom: 4, width: 1,
                background: 'var(--border)',
              }} />
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {visible.length === 0 ? (
                <Skeleton />
              ) : (
                visible.map((evt, idx) => {
                  const style   = EV_STYLES[evt.event_type] ?? { dot: '#55557a', badge: '#111122', label: 'Event' }
                  const isLast  = idx === visible.length - 1
                  const label   = EV_LABELS[evt.event_type] ?? evt.event_type

                  return (
                    <div key={`${evt.event_type}-${idx}`} className="animate-slide-left"
                      style={{
                        display: 'flex', gap: 14, alignItems: 'flex-start',
                        animationDelay: `${Math.min(idx * 0.04, 0.3)}s`,
                      }}>
                      {/* Dot */}
                      <div style={{ position: 'relative', flexShrink: 0, marginTop: 3, zIndex: 1 }}>
                        <div style={{
                          width: 13, height: 13, borderRadius: '50%',
                          background: style.dot,
                          border: `2px solid var(--bg-surface)`,
                        }} />
                        {isLast && !isFinished && (
                          <div style={{
                            position: 'absolute', inset: -3, borderRadius: '50%',
                            background: style.dot, opacity: 0.3,
                            animation: 'pulseLive 1.5s ease-in-out infinite',
                          }} />
                        )}
                      </div>

                      {/* Content */}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3, flexWrap: 'wrap' }}>
                          <span style={{
                            fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 99,
                            background: style.badge, color: style.dot,
                          }}>
                            {style.label}
                          </span>
                          <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                            {formatTime(evt.timestamp)}
                          </span>
                        </div>
                        <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                          {label}
                        </p>
                        {evt.event_type === 'research_done' && typeof evt.data.findings_count === 'number' && (
                          <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--text-tertiary)' }}>
                            {evt.data.findings_count} finding(s) received
                          </p>
                        )}
                        {evt.event_type === 'synthesis_done' && typeof evt.data.citation_count === 'number' && (
                          <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--text-tertiary)' }}>
                            {evt.data.citation_count} citation(s) produced
                          </p>
                        )}
                        {evt.event_type === 'failed' && typeof evt.data.reason === 'string' && (
                          <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--red)' }}>
                            {evt.data.reason}
                          </p>
                        )}
                      </div>
                    </div>
                  )
                })
              )}
              <div ref={bottomRef} />
            </div>
          </div>
        </div>
      </div>

      <style>{`@keyframes pulseLive { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.85)} }`}</style>
    </div>
  )
}
