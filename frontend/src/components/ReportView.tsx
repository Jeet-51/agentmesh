import { useState, useMemo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import type { SynthesisReport, Citation, ToolUsed, PartialResult } from '../types'

interface Props {
  query: string
  report: SynthesisReport | null
  partialResult: PartialResult | null
  status: 'completed' | 'partial' | 'failed'
  onReset: () => void
}

// ─── Tool styles ──────────────────────────────────────────────────────────────

const TOOL_META: Record<ToolUsed, { label: string; bg: string; color: string }> = {
  yfinance:        { label: 'yfinance',   bg: 'var(--yfinance-badge)',          color: '#7eb8f5' },
  wikipedia:       { label: 'Wikipedia',  bg: 'var(--edgar-badge)',              color: '#7edc9a' },
  edgar:           { label: 'SEC EDGAR',  bg: 'var(--edgar-badge)',              color: '#7edc9a' },
  crewai_research: { label: 'Research',   bg: 'var(--research-badge)',           color: '#a99fff' },
  newsapi:         { label: 'NewsAPI',    bg: 'rgba(251,146,60,0.15)',           color: '#fb923c' },
}

const TOOL_GROUP_ORDER: ToolUsed[] = ['crewai_research', 'yfinance', 'edgar', 'wikipedia', 'newsapi']

// ─── Helpers ──────────────────────────────────────────────────────────────────

function preprocessMarkdown(text: string): string {
  let out = text
    .replace(/\[research\]/gi,  '<span class="badge-research">research</span>')
    .replace(/\[yfinance\]/gi,  '<span class="badge-yfinance">yfinance</span>')
    .replace(/\[edgar\]/gi,     '<span class="badge-edgar">SEC EDGAR</span>')
    .replace(/\[wikipedia\]/gi, '<span class="badge-wikipedia">wikipedia</span>')

  out = out.replace(
    /(?<!\*)\b(\$[\d,]+(?:\.\d+)?(?:\s*(?:trillion|billion|million|T|B|M|K))?|[\d,.]+%|\d+(?:\.\d+)?x)\b(?!\*)/g,
    '**$1**'
  )
  return out
}

function extractTakeaways(narrative: string): string[] {
  const lines = narrative.split('\n')
  const bullets: string[] = []
  for (const line of lines) {
    const t = line.trim()
    if (/^[-*•]\s+.{15,}/.test(t)) {
      bullets.push(t.replace(/^[-*•]\s+/, '').replace(/\*\*/g, ''))
      if (bullets.length >= 5) break
    }
  }
  if (bullets.length >= 2) return bullets

  const takes: string[] = []
  let buf = ''
  for (const line of lines) {
    if (line.startsWith('## ')) {
      if (buf) {
        const s = buf.trim().split(/(?<=[.!?])\s+/)[0]
        if (s && s.length > 25) takes.push(s.replace(/\*\*/g, ''))
      }
      buf = ''
    } else if (buf === '' && line.trim() && !line.startsWith('#')) {
      buf = line.trim()
    }
  }
  if (buf) {
    const s = buf.trim().split(/(?<=[.!?])\s+/)[0]
    if (s && s.length > 25) takes.push(s.replace(/\*\*/g, ''))
  }
  return takes.slice(0, 5)
}

function overallConfidence(scores: Record<string, number>): number {
  const vals = Object.values(scores)
  if (!vals.length) return 0
  return vals.reduce((a, b) => a + b, 0) / vals.length
}

function confidenceColor(pct: number): string {
  if (pct >= 80) return 'var(--green)'
  if (pct >= 60) return 'var(--amber)'
  return 'var(--red)'
}

// ─── CopyButton ───────────────────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const [hover, setHover]   = useState(false)
  return (
    <button
      onClick={async () => {
        try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000) }
        catch { /* ignore */ }
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '8px 14px', borderRadius: 10, fontSize: 13, fontWeight: 500,
        fontFamily: 'inherit', cursor: 'pointer', transition: 'all 0.15s ease',
        background: hover ? 'var(--bg-hover)' : 'var(--bg-elevated)',
        border: '1px solid var(--border-strong)',
        color: copied ? 'var(--green)' : 'var(--text-secondary)',
      }}
    >
      {copied ? (
        <>
          <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7"/>
          </svg>
          Copied!
        </>
      ) : (
        <>
          <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
          </svg>
          Copy report
        </>
      )}
    </button>
  )
}

// ─── CitationGroup ────────────────────────────────────────────────────────────

function CitationGroup({ tool, citations }: { tool: ToolUsed; citations: Citation[] }) {
  const meta = TOOL_META[tool]
  return (
    <div className="print-section">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{
          fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 20,
          background: meta.bg, color: meta.color,
        }}>
          {meta.label}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
          {citations.length} source{citations.length !== 1 ? 's' : ''}
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {citations.map((c, i) => (
          <div
            key={c.citation_id ?? i}
            style={{
              borderRadius: 10, padding: '12px 14px',
              background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            }}
          >
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '0 0 6px', lineHeight: 1.65 }}>
              {c.claim}
            </p>
            {c.source_url ? (
              <a
                href={c.source_url}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  fontSize: 12, color: 'var(--accent)', textDecoration: 'none',
                  transition: 'opacity 0.15s ease',
                }}
                onMouseEnter={e => (e.currentTarget.style.opacity = '0.75')}
                onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
              >
                <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                </svg>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 360 }}>
                  {c.source_title || c.source_url}
                </span>
              </a>
            ) : (
              <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                {c.source_title || 'Web research'}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ReportView({ query, report, partialResult, status, onReset }: Props) {
  const [resetHover, setResetHover] = useState(false)
  const isPartial = status === 'partial'

  const processedNarrative = useMemo(() =>
    report ? preprocessMarkdown(report.narrative) : '', [report])

  const takeaways = useMemo(() =>
    report ? extractTakeaways(report.narrative) : [], [report])

  const overall = useMemo(() =>
    report ? overallConfidence(report.confidence_scores) : 0, [report])

  const overallPct = Math.round(overall * 100)

  const groupedCitations = useMemo(() => {
    if (!report) return []
    const groups = new Map<ToolUsed, Citation[]>()
    for (const c of report.citations) {
      if (!groups.has(c.tool_used)) groups.set(c.tool_used, [])
      groups.get(c.tool_used)!.push(c)
    }
    return TOOL_GROUP_ORDER
      .filter(t => groups.has(t))
      .map(t => ({ tool: t, citations: groups.get(t)! }))
  }, [report])

  const timestamp = useMemo(() => new Date().toLocaleString([], {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
  }), [])

  return (
    <div
      className="min-h-screen px-4 py-10 pb-16 flex flex-col items-center animate-fade-in print-area"
      style={{ background: 'var(--bg-base)' }}
    >
      <div style={{ width: '100%', maxWidth: 760, display: 'flex', flexDirection: 'column', gap: 16 }}>

        {/* ── Warning: Partial ── */}
        {isPartial && (
          <div className="no-print" style={{
            borderRadius: 12, padding: '12px 16px',
            background: 'rgba(245,166,35,0.08)', border: '1px solid rgba(245,166,35,0.25)',
          }}>
            <p style={{ color: 'var(--amber)', fontSize: 14, fontWeight: 500, margin: '0 0 2px' }}>⚠ Partial result</p>
            <p style={{ fontSize: 12, color: 'var(--amber)', opacity: 0.75, margin: 0 }}>
              {partialResult
                ? `Failed: ${partialResult.failed_agents.join(', ')}. Report built from available data.`
                : 'One or more agents failed. Report may be incomplete.'}
            </p>
          </div>
        )}

        {/* ── Warning: Failed ── */}
        {status === 'failed' && !report && (
          <div style={{
            borderRadius: 12, padding: '12px 16px',
            background: 'rgba(242,92,92,0.08)', border: '1px solid rgba(242,92,92,0.25)',
          }}>
            <p style={{ color: 'var(--red)', fontSize: 14, fontWeight: 500, margin: '0 0 2px' }}>Run failed</p>
            <p style={{ fontSize: 12, color: 'var(--red)', opacity: 0.75, margin: 0 }}>
              {partialResult?.partial_report ?? 'The agents encountered an unrecoverable error.'}
            </p>
            <button
              onClick={onReset}
              style={{
                marginTop: 12, padding: '8px 20px', borderRadius: 10, fontSize: 13,
                fontWeight: 500, fontFamily: 'inherit', cursor: 'pointer',
                background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)',
                color: 'var(--text-secondary)',
              }}
            >
              ← New Query
            </button>
          </div>
        )}

        {report && (
          <>
            {/* ── SECTION 1: Report header ── */}
            <div className="print-section" style={{
              borderRadius: 20, padding: '24px 28px',
              background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
              boxShadow: 'var(--shadow)',
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, marginBottom: 16 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <span style={{
                      width: 7, height: 7, borderRadius: '50%', background: 'var(--green)',
                      boxShadow: '0 0 8px rgba(34,211,160,0.6)', display: 'inline-block',
                      flexShrink: 0,
                    }} />
                    <span style={{
                      fontSize: 11, fontWeight: 500, color: 'var(--green)',
                      textTransform: 'uppercase', letterSpacing: '0.1em',
                    }}>
                      {isPartial ? 'Partial report' : 'Report ready'}
                    </span>
                  </div>
                  <h2 style={{
                    fontSize: 22, fontWeight: 600, color: 'var(--text-primary)',
                    margin: 0, letterSpacing: '-0.3px', lineHeight: 1.35,
                  }}>
                    {query}
                  </h2>
                </div>
              </div>

              {/* Meta row */}
              <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>{timestamp}</span>
                <span style={{
                  fontSize: 11, padding: '2px 8px', borderRadius: 20,
                  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  color: 'var(--text-tertiary)',
                }}>
                  Powered by AgentMesh
                </span>
                {['LangGraph', 'CrewAI', 'ADK'].map(f => (
                  <span key={f} style={{
                    fontSize: 10, fontWeight: 500, padding: '2px 7px', borderRadius: 20,
                    border: '1px solid rgba(124,106,247,0.3)', color: 'var(--accent)',
                    background: 'transparent',
                  }}>
                    {f}
                  </span>
                ))}
              </div>
            </div>

            {/* ── TOP: New Query button ── */}
            <div className="no-print" style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={onReset}
                onMouseEnter={e => {
                  e.currentTarget.style.background = 'var(--accent)'
                  e.currentTarget.style.color = '#fff'
                  e.currentTarget.style.borderColor = 'var(--accent)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'var(--bg-elevated)'
                  e.currentTarget.style.color = 'var(--text-secondary)'
                  e.currentTarget.style.borderColor = 'var(--border-strong)'
                }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '8px 18px', borderRadius: 10, fontSize: 13, fontWeight: 500,
                  fontFamily: 'inherit', cursor: 'pointer', transition: 'all 0.15s ease',
                  background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)',
                  color: 'var(--text-secondary)',
                }}
              >
                ← New Query
              </button>
            </div>

            {/* ── SECTION 2: Confidence ── */}
            <div className="print-section" style={{
              borderRadius: 20, padding: '24px 28px',
              background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <h3 style={{
                  fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)',
                  textTransform: 'uppercase', letterSpacing: '0.09em', margin: 0,
                }}>
                  Research Confidence
                </h3>
                {/* Large percentage */}
                <span style={{
                  fontSize: 36, fontWeight: 700, color: confidenceColor(overallPct),
                  lineHeight: 1, fontVariantNumeric: 'tabular-nums',
                }}>
                  {overallPct}%
                </span>
              </div>

              {/* Overall bar */}
              <div style={{
                height: 6, borderRadius: 99, overflow: 'hidden',
                background: 'var(--bg-elevated)', marginBottom: 16,
              }}>
                <div style={{
                  height: '100%', borderRadius: 99,
                  width: `${overallPct}%`,
                  background: `linear-gradient(90deg, var(--red) 0%, var(--amber) 50%, var(--green) 100%)`,
                  transition: 'width 0.8s ease-out',
                }} />
              </div>

              {/* Per-task bars */}
              {Object.keys(report.confidence_scores).length > 1 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {Object.entries(report.confidence_scores).map(([key, val]) => {
                    const p = Math.round(val * 100)
                    return (
                      <div key={key}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: 'var(--text-tertiary)', textTransform: 'capitalize' }}>{key}</span>
                          <span style={{ fontSize: 12, fontWeight: 600, fontVariantNumeric: 'tabular-nums', color: confidenceColor(p) }}>{p}%</span>
                        </div>
                        <div style={{ height: 4, borderRadius: 99, overflow: 'hidden', background: 'var(--bg-elevated)' }}>
                          <div style={{
                            height: '100%', borderRadius: 99, width: `${p}%`,
                            background: confidenceColor(p), transition: 'width 0.7s ease-out',
                          }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* ── SECTION 3: Two-column row ── */}
            <div className="print-section" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              {/* Key Takeaways */}
              <div style={{
                borderRadius: 20, padding: '20px 24px',
                background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
              }}>
                <h3 style={{
                  fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)',
                  textTransform: 'uppercase', letterSpacing: '0.09em',
                  margin: '0 0 14px', display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <span>⚡</span> Key Takeaways
                </h3>
                {takeaways.length > 0 ? (
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {takeaways.map((t, i) => (
                      <li key={i} style={{ display: 'flex', gap: 10, fontSize: 13, lineHeight: 1.55 }}>
                        <span style={{
                          width: 5, height: 5, borderRadius: '50%', background: 'var(--accent)',
                          flexShrink: 0, marginTop: 6,
                        }} />
                        <span style={{ color: 'var(--text-secondary)' }}>{t}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p style={{ fontSize: 13, color: 'var(--text-tertiary)', margin: 0 }}>
                    See full analysis below.
                  </p>
                )}
              </div>

              {/* Recommended Actions */}
              <div style={{
                borderRadius: 20, padding: '20px 24px',
                background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
              }}>
                <h3 style={{
                  fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)',
                  textTransform: 'uppercase', letterSpacing: '0.09em',
                  margin: '0 0 14px', display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <span>✓</span> Recommended Actions
                </h3>
                {report.recommended_actions.length > 0 ? (
                  <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {report.recommended_actions.map((action, i) => (
                      <li key={i} style={{ display: 'flex', gap: 10, fontSize: 13, lineHeight: 1.55 }}>
                        <span style={{
                          fontSize: 11, fontWeight: 700, color: 'var(--accent)',
                          flexShrink: 0, fontVariantNumeric: 'tabular-nums', marginTop: 2,
                        }}>
                          {i + 1}.
                        </span>
                        <span style={{ color: 'var(--text-secondary)' }}>{action}</span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p style={{ fontSize: 13, color: 'var(--text-tertiary)', margin: 0 }}>
                    No specific actions recommended.
                  </p>
                )}
              </div>
            </div>

            {/* ── SECTION 4: Full Analysis ── */}
            <div className="print-section" style={{
              borderRadius: 20, padding: '24px 28px',
              background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
            }}>
              <h3 style={{
                fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)',
                textTransform: 'uppercase', letterSpacing: '0.09em', margin: '0 0 20px',
              }}>
                Full Analysis
              </h3>
              <div className="markdown-body">
                <Markdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    h2: ({ children }) => (
                      <h2 style={{
                        color: 'var(--text-primary)', fontSize: '1.05rem', fontWeight: 600,
                        borderBottom: '1px solid var(--border)', paddingBottom: '0.35rem',
                        marginTop: '1.75rem', marginBottom: '0.75rem',
                      }}>{children}</h2>
                    ),
                    h3: ({ children }) => (
                      <h3 style={{
                        color: 'var(--text-secondary)', fontSize: '0.95rem', fontWeight: 600,
                        marginTop: '1.25rem', marginBottom: '0.5rem',
                      }}>{children}</h3>
                    ),
                    p: ({ children }) => (
                      <p style={{
                        color: 'var(--text-secondary)', fontSize: '15px',
                        lineHeight: '1.85', marginBottom: '1rem',
                      }}>{children}</p>
                    ),
                    ul: ({ children }) => (
                      <ul style={{
                        paddingLeft: '1.5rem', marginBottom: '1rem',
                        color: 'var(--text-secondary)', listStyleType: 'disc',
                      }}>{children}</ul>
                    ),
                    ol: ({ children }) => (
                      <ol style={{
                        paddingLeft: '1.5rem', marginBottom: '1rem',
                        color: 'var(--text-secondary)', listStyleType: 'decimal',
                      }}>{children}</ol>
                    ),
                    li: ({ children }) => (
                      <li style={{
                        marginBottom: '0.375rem', lineHeight: '1.75',
                        fontSize: '15px', color: 'var(--text-secondary)',
                      }}>{children}</li>
                    ),
                    strong: ({ children }) => (
                      <strong style={{ fontWeight: 650, color: 'var(--text-primary)' }}>
                        {children}
                      </strong>
                    ),
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: 'var(--accent)', textDecoration: 'underline', textUnderlineOffset: '3px' }}
                      >
                        {children}
                      </a>
                    ),
                  }}
                >
                  {processedNarrative}
                </Markdown>
              </div>
            </div>

            {/* ── SECTION 5: Sources & Citations ── */}
            {groupedCitations.length > 0 && (
              <div className="print-section" style={{
                borderRadius: 20, padding: '24px 28px',
                background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
              }}>
                <h3 style={{
                  fontSize: 11, fontWeight: 500, color: 'var(--text-tertiary)',
                  textTransform: 'uppercase', letterSpacing: '0.09em',
                  margin: '0 0 16px', display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
                  </svg>
                  Sources &amp; Citations
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  {groupedCitations.map(({ tool, citations }) => (
                    <CitationGroup key={tool} tool={tool} citations={citations} />
                  ))}
                </div>
              </div>
            )}

            {/* ── SECTION 6: Action row ── */}
            <div className="no-print" style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <CopyButton text={report.narrative} />

              <button
                onClick={() => window.print()}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-hover)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'var(--bg-elevated)')}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '8px 14px', borderRadius: 10, fontSize: 13, fontWeight: 500,
                  fontFamily: 'inherit', cursor: 'pointer', transition: 'all 0.15s ease',
                  background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)',
                  color: 'var(--text-secondary)',
                }}
              >
                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
                Download PDF
              </button>

              <button
                onClick={onReset}
                onMouseEnter={() => setResetHover(true)}
                onMouseLeave={() => setResetHover(false)}
                style={{
                  marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6,
                  padding: '8px 18px', borderRadius: 10, fontSize: 13, fontWeight: 500,
                  fontFamily: 'inherit', cursor: 'pointer', transition: 'all 0.15s ease',
                  background: resetHover ? 'var(--accent)' : 'var(--bg-elevated)',
                  border: '1px solid var(--border-strong)',
                  color: resetHover ? '#fff' : 'var(--text-secondary)',
                }}
              >
                ← New Query
              </button>
            </div>

            {/* Footer */}
            <p className="no-print" style={{
              textAlign: 'center', fontSize: 12, color: 'var(--text-tertiary)',
              paddingBottom: 8, margin: 0,
            }}>
              Powered by <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>AgentMesh</span>
              {' '}· LangGraph · CrewAI · Google ADK · MCP · A2A
            </p>
          </>
        )}

        {/* Reset if no report (edge case) */}
        {!report && status !== 'failed' && (
          <button
            onClick={onReset}
            style={{
              width: '100%', padding: '12px', borderRadius: 12, fontSize: 14,
              fontWeight: 500, fontFamily: 'inherit', cursor: 'pointer',
              background: 'var(--bg-surface)', border: '1px solid var(--border-strong)',
              color: 'var(--text-secondary)', transition: 'all 0.15s ease',
            }}
          >
            ← New Query
          </button>
        )}

      </div>
    </div>
  )
}
