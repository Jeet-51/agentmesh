import { useState, useEffect, useRef, useCallback } from 'react'

interface Props {
  onSubmit: (query: string) => void
  loading: boolean
  error: string | null
}

// ─── Example queries ──────────────────────────────────────────────────────────

const CATEGORIES = [
  {
    label: 'Finance',
    queries: [
      "What is Nvidia's financial performance and AI strategy in 2025?",
      "Compare Apple vs Microsoft revenue growth over the last 3 years",
      "Analyze Tesla's profitability challenges and competitive position",
    ],
  },
  {
    label: 'Markets',
    queries: [
      "What are the key risks in the semiconductor industry right now?",
      "How is the Federal Reserve's policy affecting tech valuations?",
      "What is the investment thesis for Amazon Web Services in 2025?",
    ],
  },
  {
    label: 'Technology',
    queries: [
      "Explain the competitive landscape in large language models",
      "What is the state of quantum computing commercialization?",
      "How are AI agents transforming enterprise software?",
    ],
  },
]

const ALL_EXAMPLES = CATEGORIES.flatMap(c => c.queries)

const FRAMEWORK_BADGES = [
  { label: 'LangGraph',  color: '#a78bfa' },
  { label: 'CrewAI',     color: '#34d399' },
  { label: 'Google ADK', color: '#60a5fa' },
  { label: 'MCP',        color: '#fbbf24' },
  { label: 'A2A',        color: '#f472b6' },
]

// ─── Typewriter placeholder hook ─────────────────────────────────────────────

function useTypewriterPlaceholder(examples: string[], holdMs = 3800) {
  const [display, setDisplay] = useState('')
  const [exIdx, setExIdx]     = useState(0)
  const [phase, setPhase]     = useState<'type' | 'pause' | 'erase'>('type')
  const charRef = useRef(0)

  useEffect(() => {
    const target = examples[exIdx]
    if (phase === 'type') {
      if (charRef.current < target.length) {
        const t = setTimeout(() => {
          charRef.current += 1
          setDisplay(target.slice(0, charRef.current))
        }, 26)
        return () => clearTimeout(t)
      } else {
        const t = setTimeout(() => setPhase('erase'), holdMs)
        return () => clearTimeout(t)
      }
    }
    if (phase === 'erase') {
      if (charRef.current > 0) {
        const t = setTimeout(() => {
          charRef.current -= 1
          setDisplay(target.slice(0, charRef.current))
        }, 12)
        return () => clearTimeout(t)
      } else {
        setExIdx(i => (i + 1) % examples.length)
        setPhase('type')
      }
    }
  }, [display, phase, exIdx, examples, holdMs])

  return display
}

// ─── Spinner style ────────────────────────────────────────────────────────────

const spinnerStyle: React.CSSProperties = {
  width: 16,
  height: 16,
  border: '2px solid rgba(255,255,255,0.25)',
  borderTopColor: '#fff',
  borderRadius: '50%',
  display: 'inline-block',
  animation: 'spin 0.7s linear infinite',
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function QueryInput({ onSubmit, loading, error }: Props) {
  const [query, setQuery]       = useState('')
  const [focused, setFocused]   = useState(false)
  const [btnHover, setBtnHover] = useState(false)
  const [openCat, setOpenCat]   = useState<number | null>(null)
  const textareaRef             = useRef<HTMLTextAreaElement>(null)
  const placeholder             = useTypewriterPlaceholder(ALL_EXAMPLES)

  // Auto-grow textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.max(80, el.scrollHeight)}px`
  }, [query])

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (query.trim() && !loading) onSubmit(query.trim())
    }
  }, [query, loading, onSubmit])

  const canSubmit = !loading && query.trim().length > 0

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-16 animate-fade-in"
      style={{ background: 'var(--bg-base)' }}
    >
      <div className="w-full" style={{ maxWidth: 680 }}>

        {/* ── Logo + Title ── */}
        <div className="flex flex-col items-center mb-10">
          <div
            style={{
              width: 56, height: 56, borderRadius: 16,
              background: 'linear-gradient(135deg, #7c6af7 0%, #22d3a0 100%)',
              boxShadow: '0 0 40px #7c6af730',
              fontSize: 22, fontWeight: 700, color: '#fff',
              letterSpacing: '-0.5px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              marginBottom: 16,
            }}
          >
            AM
          </div>

          <h1 style={{ fontSize: 28, fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 6px', letterSpacing: '-0.5px' }}>
            AgentMesh
          </h1>
          <p style={{ fontSize: 15, color: 'var(--text-secondary)', margin: 0 }}>
            Multi-agent research — grounded in live data
          </p>

          {/* Framework badges */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 16, justifyContent: 'center' }}>
            {FRAMEWORK_BADGES.map(b => (
              <span key={b.label} style={{
                fontSize: 11, fontWeight: 500, padding: '3px 10px', borderRadius: 20,
                border: `1px solid ${b.color}66`, color: b.color, background: 'transparent',
              }}>
                {b.label}
              </span>
            ))}
          </div>
        </div>

        {/* ── Textarea ── */}
        <div style={{
          borderRadius: 16,
          border: `1px solid ${focused ? 'var(--accent)' : 'var(--border-strong)'}`,
          background: 'var(--bg-surface)',
          transition: 'border-color 0.15s ease',
          marginBottom: 12,
        }}>
          <textarea
            ref={textareaRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            disabled={loading}
            placeholder={focused ? '' : (placeholder || 'Ask anything about a company, market, or technology…')}
            style={{
              width: '100%', minHeight: 80,
              padding: '18px 20px',
              fontSize: 16, fontFamily: 'inherit', fontWeight: 400,
              color: 'var(--text-primary)', background: 'transparent',
              border: 'none', outline: 'none', resize: 'none',
              lineHeight: 1.6, display: 'block', borderRadius: 16,
            }}
          />
        </div>

        {/* ── Submit button ── */}
        <button
          onClick={() => canSubmit && onSubmit(query.trim())}
          disabled={!canSubmit}
          onMouseEnter={() => setBtnHover(true)}
          onMouseLeave={() => setBtnHover(false)}
          style={{
            width: '100%', height: 48, borderRadius: 12, border: 'none',
            background: canSubmit ? (btnHover ? 'var(--accent-hover)' : 'var(--accent)') : 'var(--bg-elevated)',
            color: canSubmit ? '#fff' : 'var(--text-tertiary)',
            fontSize: 15, fontWeight: 500, fontFamily: 'inherit',
            cursor: canSubmit ? 'pointer' : 'default',
            transition: 'all 0.15s ease',
            transform: canSubmit && btnHover ? 'translateY(-1px)' : 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          }}
        >
          {loading
            ? <><span style={spinnerStyle} />Starting research…</>
            : 'Research this →'
          }
        </button>

        {/* ── Hint ── */}
        <p style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-tertiary)', marginTop: 12 }}>
          Press Enter to submit · Shift+Enter for newline · ~2–4 min
        </p>

        {/* ── Error ── */}
        {error && (
          <div className="animate-slide-up" style={{
            marginTop: 16, padding: '12px 16px', borderRadius: 12,
            background: 'rgba(242,92,92,0.08)', border: '1px solid rgba(242,92,92,0.25)',
            color: 'var(--red)', fontSize: 13,
          }}>
            {error}
          </div>
        )}

        {/* ── Example categories ── */}
        <div style={{ marginTop: 32 }}>
          {CATEGORIES.map((cat, ci) => (
            <div key={cat.label} style={{ marginBottom: 16 }}>
              <button
                onClick={() => setOpenCat(openCat === ci ? null : ci)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 6,
                  marginBottom: 8, padding: 0,
                }}
              >
                <span style={{
                  fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)',
                  textTransform: 'uppercase', letterSpacing: '0.08em',
                }}>
                  {cat.label}
                </span>
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"
                  style={{ color: 'var(--text-tertiary)', transition: 'transform 0.15s ease',
                    transform: openCat === ci ? 'rotate(180deg)' : 'none' }}>
                  <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth={1.5}
                    strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>

              {openCat !== ci && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {cat.queries.map(q => (
                    <button key={q}
                      onClick={() => { setQuery(q); setTimeout(() => textareaRef.current?.focus(), 0) }}
                      disabled={loading}
                      onMouseEnter={e => {
                        e.currentTarget.style.borderColor = 'var(--border-strong)'
                        e.currentTarget.style.color = 'var(--text-primary)'
                        e.currentTarget.style.background = 'var(--bg-hover)'
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.borderColor = 'var(--border)'
                        e.currentTarget.style.color = 'var(--text-secondary)'
                        e.currentTarget.style.background = 'var(--bg-surface)'
                      }}
                      style={{
                        background: 'var(--bg-surface)', border: '1px solid var(--border)',
                        borderRadius: 8, padding: '8px 14px', fontSize: 13,
                        fontFamily: 'inherit', color: 'var(--text-secondary)',
                        cursor: loading ? 'default' : 'pointer',
                        textAlign: 'left', transition: 'all 0.15s ease', lineHeight: 1.5,
                      }}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

      </div>

      {/* Spin keyframe via style tag */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
