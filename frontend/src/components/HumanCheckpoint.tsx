import { useState } from 'react'
import type { SubTask } from '../types'

interface Props {
  runId: string
  subTasks: SubTask[]
  onApprove: (edited: SubTask[]) => void
  onReject: (feedback: string) => void
  loading: boolean
}

const TASK_ICONS = ['🔍', '📊', '📰', '💡', '⚙️', '🧪']

// ─── Spinner ──────────────────────────────────────────────────────────────────

const spinnerStyle: React.CSSProperties = {
  width: 15, height: 15,
  border: '2px solid rgba(255,255,255,0.25)',
  borderTopColor: '#fff',
  borderRadius: '50%',
  display: 'inline-block',
  animation: 'spin 0.7s linear infinite',
}

// ─── TaskCard ─────────────────────────────────────────────────────────────────

interface TaskCardProps {
  task: SubTask
  index: number
  loading: boolean
  onChange: (field: 'topic' | 'instructions', value: string) => void
}

function TaskCard({ task, index, loading, onChange }: TaskCardProps) {
  const [collapsed, setCollapsed] = useState(false)
  const [topicFocused, setTopicFocused]   = useState(false)
  const [instrFocused, setInstrFocused]   = useState(false)
  const icon = TASK_ICONS[index % TASK_ICONS.length]

  return (
    <div
      className="animate-slide-up"
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-strong)',
        borderRadius: 16,
        overflow: 'hidden',
        animationDelay: `${index * 0.07}s`,
      }}
    >
      {/* Card header */}
      <button
        type="button"
        onClick={() => setCollapsed(c => !c)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 12,
          padding: '14px 20px', background: 'none', border: 'none',
          cursor: 'pointer', textAlign: 'left',
        }}
      >
        <span style={{ fontSize: 16, flexShrink: 0 }}>{icon}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Sub-task {index + 1}
          </span>
          {collapsed && (
            <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', margin: '2px 0 0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {task.topic}
            </p>
          )}
        </div>
        <span style={{
          fontSize: 10, fontWeight: 500, padding: '2px 8px', borderRadius: 20,
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          color: 'var(--text-tertiary)', flexShrink: 0,
        }}>
          {task.status}
        </span>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
          style={{ color: 'var(--text-tertiary)', flexShrink: 0, transition: 'transform 0.15s ease', transform: collapsed ? 'none' : 'rotate(180deg)' }}>
          <path d="M2 5l5 5 5-5" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {/* Collapsible body */}
      {!collapsed && (
        <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border)' }}>
          <div style={{ paddingTop: 16 }}>

            {/* Topic */}
            <label style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em', display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span>Topic</span>
              <span style={{ color: task.topic.length > 200 ? 'var(--red)' : 'var(--text-tertiary)' }}>
                {task.topic.length}/200
              </span>
            </label>
            <input
              value={task.topic}
              onChange={e => onChange('topic', e.target.value)}
              disabled={loading}
              style={{
                width: '100%', padding: '10px 14px', fontSize: 13,
                fontFamily: 'inherit', color: 'var(--text-primary)',
                background: 'var(--bg-elevated)', borderRadius: 10,
                border: `1px solid ${topicFocused ? 'var(--accent)' : 'var(--border-strong)'}`,
                outline: 'none', transition: 'border-color 0.15s ease',
                marginBottom: 16,
              }}
              onFocus={() => setTopicFocused(true)}
              onBlur={() => setTopicFocused(false)}
            />

            {/* Instructions */}
            <label style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em', display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span>Research instructions</span>
              <span style={{ color: task.instructions.length > 500 ? 'var(--red)' : 'var(--text-tertiary)' }}>
                {task.instructions.length}/500
              </span>
            </label>
            <p style={{ fontSize: 11, color: 'var(--text-tertiary)', marginBottom: 8, marginTop: 0 }}>
              ✎ Edit to customise the research focus for this sub-task
            </p>
            <textarea
              value={task.instructions}
              onChange={e => onChange('instructions', e.target.value)}
              disabled={loading}
              style={{
                width: '100%', padding: '10px 14px', fontSize: 13,
                fontFamily: 'inherit', color: 'var(--text-primary)',
                background: 'var(--bg-elevated)', borderRadius: 10,
                border: `1px solid ${instrFocused ? 'var(--accent)' : 'var(--border-strong)'}`,
                outline: 'none', resize: 'vertical', minHeight: 100, maxHeight: 340,
                transition: 'border-color 0.15s ease', lineHeight: 1.6,
              }}
              onFocus={() => setInstrFocused(true)}
              onBlur={() => setInstrFocused(false)}
            />
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function HumanCheckpoint({ subTasks, onApprove, onReject, loading }: Props) {
  const [edited, setEdited]           = useState<SubTask[]>(() => subTasks.map(t => ({ ...t })))
  const [showReject, setShowReject]   = useState(false)
  const [feedback, setFeedback]       = useState('')
  const [approvHover, setApprovHover] = useState(false)
  const [rejectHover, setRejectHover] = useState(false)
  const [fbFocused, setFbFocused]     = useState(false)

  function updateField(idx: number, field: 'topic' | 'instructions', value: string) {
    setEdited(prev => prev.map((t, i) => i === idx ? { ...t, [field]: value } : t))
  }

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-12 animate-fade-in"
      style={{ background: 'var(--bg-base)' }}
    >
      <div style={{ width: '100%', maxWidth: 640 }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            {/* Amber pulsing dot */}
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: 'var(--amber)',
              boxShadow: '0 0 10px rgba(245,166,35,0.6)',
              display: 'inline-block',
              animation: 'pulseLive 1.6s ease-in-out infinite',
            }} />
            <span style={{
              fontSize: 11, fontWeight: 500, color: 'var(--amber)',
              textTransform: 'uppercase', letterSpacing: '0.1em',
            }}>
              Human checkpoint
            </span>
          </div>

          <h2 style={{ fontSize: 24, fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 10px', letterSpacing: '-0.3px' }}>
            Review research plan before proceeding
          </h2>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.7 }}>
            Your query was split into{' '}
            <strong style={{ color: 'var(--text-primary)' }}>{edited.length} sub-task{edited.length !== 1 ? 's' : ''}</strong>.
            {' '}Edit any topic or instructions, then approve to send to research agents.
          </p>
        </div>

        {/* ── Info box ── */}
        <div style={{
          borderRadius: 12, padding: '12px 16px', marginBottom: 24,
          background: 'rgba(124,106,247,0.08)', border: '1px solid rgba(124,106,247,0.2)',
        }}>
          <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', margin: '0 0 4px', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            What happens next?
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.65 }}>
            Each sub-task is sent to a <strong style={{ color: 'var(--text-primary)' }}>CrewAI research crew</strong> (Searcher → Fact-checker → Summarizer).
            Results are synthesised by a <strong style={{ color: 'var(--text-primary)' }}>Google ADK agent</strong> using live financial data.
            Estimated time: <strong style={{ color: 'var(--text-primary)' }}>2–4 min</strong>.
          </p>
        </div>

        {/* ── Sub-task cards ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 20 }}>
          {edited.map((task, idx) => (
            <TaskCard
              key={task.sub_task_id}
              task={task}
              index={idx}
              loading={loading}
              onChange={(field, value) => updateField(idx, field, value)}
            />
          ))}
        </div>

        {/* ── Rejection feedback ── */}
        {showReject && (
          <div className="animate-slide-up" style={{ marginBottom: 16 }}>
            <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', marginBottom: 8 }}>
              What should the AI change?
            </p>
            <textarea
              value={feedback}
              onChange={e => setFeedback(e.target.value)}
              onFocus={() => setFbFocused(true)}
              onBlur={() => setFbFocused(false)}
              placeholder="Optional: describe changes for the next decomposition…"
              rows={2}
              style={{
                width: '100%', padding: '10px 14px', fontSize: 13, fontFamily: 'inherit',
                color: 'var(--text-primary)', background: 'var(--bg-surface)',
                borderRadius: 10, resize: 'none', outline: 'none',
                border: `1px solid ${fbFocused ? 'var(--red)' : 'rgba(242,92,92,0.35)'}`,
                transition: 'border-color 0.15s ease', lineHeight: 1.6,
              }}
            />
          </div>
        )}

        {/* ── Action buttons ── */}
        <div style={{ display: 'flex', gap: 12 }}>
          {/* Approve */}
          <button
            onClick={() => onApprove(edited)}
            disabled={loading}
            onMouseEnter={() => setApprovHover(true)}
            onMouseLeave={() => setApprovHover(false)}
            style={{
              flex: 1, height: 48, borderRadius: 12, border: 'none',
              background: approvHover ? 'var(--accent-hover)' : 'var(--accent)',
              color: '#fff', fontSize: 15, fontWeight: 500, fontFamily: 'inherit',
              cursor: loading ? 'default' : 'pointer',
              transition: 'all 0.15s ease',
              transform: approvHover && !loading ? 'translateY(-1px)' : 'none',
              opacity: loading ? 0.7 : 1,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            }}
          >
            {loading ? (
              <><span style={spinnerStyle} />Dispatching…</>
            ) : (
              <>
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7"/>
                </svg>
                Approve &amp; Run
              </>
            )}
          </button>

          {/* Reject */}
          <button
            onClick={() => showReject ? onReject(feedback) : setShowReject(true)}
            disabled={loading}
            onMouseEnter={() => setRejectHover(true)}
            onMouseLeave={() => setRejectHover(false)}
            style={{
              flex: 1, height: 48, borderRadius: 12,
              background: 'transparent',
              border: `1px solid ${rejectHover ? 'var(--border-strong)' : 'var(--border-strong)'}`,
              color: rejectHover ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontSize: 15, fontWeight: 500, fontFamily: 'inherit',
              cursor: loading ? 'default' : 'pointer',
              transition: 'all 0.15s ease',
              opacity: loading ? 0.5 : 1,
            }}
          >
            {showReject ? 'Confirm Reject' : 'Reject & Regenerate'}
          </button>
        </div>

        <p style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-tertiary)', marginTop: 12 }}>
          Expand each card to edit topic or instructions before approving
        </p>

      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
