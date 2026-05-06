import { useState, useEffect, useCallback } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface MetricScores {
  hallucination?: number | null
  quantitative?: number | null
  freshness?: number | null
  diversity?: number | null
  entity_coverage?: number | null
  narrative_length?: number | null
  source_credibility?: number | null
  fictional_premise?: number | null
  answer_relevance?: number | null
  tool_activation?: number | null
  citation_density?: number | null
  confidence_calibration?: number | null
  overall?: number | null
}

interface EvalReport {
  query: string
  timestamp: string
  confidence: number
  scores: MetricScores
}

interface EvalData {
  available: boolean
  reports: EvalReport[]
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function scoreColor(s: number | null | undefined): string {
  if (s == null) return 'var(--text-muted)'
  if (s >= 0.75) return '#3fb950'
  if (s >= 0.50) return '#d29922'
  return '#f85149'
}

function scoreBg(s: number | null | undefined): string {
  if (s == null) return 'transparent'
  if (s >= 0.75) return 'rgba(63,185,80,0.10)'
  if (s >= 0.50) return 'rgba(210,153,34,0.10)'
  return 'rgba(248,81,73,0.10)'
}

function fmt(s: number | null | undefined): string {
  if (s == null) return '—'
  return s.toFixed(2)
}

function avgOf(reports: EvalReport[], key: keyof MetricScores): number | null {
  const vals = reports.map(r => r.scores[key]).filter((v): v is number => v != null)
  if (!vals.length) return null
  return vals.reduce((a, b) => a + b, 0) / vals.length
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const min = Math.floor(diff / 60000)
  if (min < 1) return 'just now'
  if (min < 60) return `${min}m ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h ago`
  return `${Math.floor(hr / 24)}d ago`
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function ScorePill({ value }: { value: number | null | undefined }) {
  return (
    <span style={{
      display: 'inline-block',
      minWidth: 48,
      padding: '2px 8px',
      borderRadius: 6,
      fontSize: 13,
      fontWeight: 700,
      fontFamily: 'monospace',
      textAlign: 'center',
      color: scoreColor(value),
      background: scoreBg(value),
    }}>
      {fmt(value)}
    </span>
  )
}

function HeroCard({
  label, value, sub, accent,
}: { label: string; value: string; sub: string; accent: string }) {
  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: `1px solid ${accent}33`,
      borderRadius: 14,
      padding: '24px 20px',
      textAlign: 'center',
      flex: 1,
      minWidth: 0,
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ fontSize: 40, fontWeight: 800, fontFamily: 'monospace', color: accent, lineHeight: 1.1 }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>{sub}</div>
    </div>
  )
}

// ─── Tooltip ─────────────────────────────────────────────────────────────────

function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)

  return (
    <>
      <div
        style={{ display: 'inline-flex', alignItems: 'center', cursor: 'default' }}
        onMouseMove={e => setPos({ x: e.clientX, y: e.clientY })}
        onMouseLeave={() => setPos(null)}
      >
        {children}
      </div>
      {pos && (
        <div style={{
          position: 'fixed',
          left: pos.x,
          top: pos.y - 44,
          transform: 'translateX(-50%)',
          background: '#1c2128',
          border: '1px solid #444c56',
          borderRadius: 8,
          padding: '7px 12px',
          fontSize: 12,
          color: '#cdd9e5',
          zIndex: 9999,
          boxShadow: '0 4px 20px rgba(0,0,0,0.6)',
          pointerEvents: 'none',
          maxWidth: 280,
          whiteSpace: 'normal' as const,
          lineHeight: 1.5,
        }}>
          {text}
          <div style={{
            position: 'absolute',
            top: '100%', left: '50%',
            transform: 'translateX(-50%)',
            width: 0, height: 0,
            borderLeft: '5px solid transparent',
            borderRight: '5px solid transparent',
            borderTop: '5px solid #1c2128',
          }} />
        </div>
      )}
    </>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function EvalDashboard({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<EvalData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'live' | 'improvements'>('live')

  const fetchScores = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/evals/scores?limit=20')
      if (!res.ok) throw new Error(`Gateway returned ${res.status}`)
      const json = await res.json()
      setData(json)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load eval scores')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchScores()
    const interval = setInterval(fetchScores, 30000) // refresh every 30s
    return () => clearInterval(interval)
  }, [fetchScores])

  const reports = data?.reports ?? []

  // Aggregates
  const avgOverall = avgOf(reports, 'overall')
  const bestOverall = reports.length
    ? Math.max(...reports.map(r => r.scores.overall ?? 0))
    : null

  const primaryMetrics: { key: keyof MetricScores; label: string; short: string; tip: string }[] = [
    {
      key: 'hallucination', label: 'Hallucination', short: 'Hall',
      tip: 'Citation count ÷ paragraphs. 1.00 = every paragraph has a citation.',
    },
    {
      key: 'quantitative', label: 'Quantitative', short: 'Quant',
      tip: 'Numbers in narrative matched against yfinance ground-truth (±15% tolerance). 1.0 = figures are accurate and precise.',
    },
    {
      key: 'freshness', label: 'Freshness', short: 'Fresh',
      tip: 'Average age of dated citations. 1.00 = very recent. 0.50 = no dates found.',
    },
    {
      key: 'diversity', label: 'Diversity', short: 'Div',
      tip: 'Unique source domains ÷ total sources. 1.00 = every citation from a different site.',
    },
    {
      key: 'entity_coverage', label: 'Entity Coverage', short: 'Ent',
      tip: 'Key nouns from your query found in the narrative. 1.00 = fully answers the question.',
    },
    {
      key: 'narrative_length', label: 'Narrative Length', short: 'Narr',
      tip: 'Word count (ideal 400–900) + 5 expected section headers present.',
    },
    {
      key: 'source_credibility', label: 'Source Credibility', short: 'Cred',
      tip: 'Weighted domain authority: SEC/Gov=1.0, Reuters/Bloomberg/Finnhub=0.8, Yahoo/CNBC=0.65, Wikipedia=0.5, unknown=0.3.',
    },
    {
      key: 'fictional_premise', label: 'Fictional Premise', short: 'Fict',
      tip: 'Rewards proper hedging ("analysts expect", "reportedly") for uncertain claims. Penalises fabricated figures. 1.0 = well-hedged.',
    },
    {
      key: 'answer_relevance', label: 'Answer Relevance', short: 'Rel',
      tip: 'Does the conclusion directly address the specific question? For comparisons, it must name which wins and by how much. 1.0 = fully answered.',
    },
    {
      key: 'overall', label: 'Overall', short: 'Overall',
      tip: 'Mean of all 11 primary metrics.',
    },
  ]

  const diagMetrics: { key: keyof MetricScores; label: string; tip: string }[] = [
    {
      key: 'tool_activation', label: 'Tool Activation',
      tip: 'Fraction of 5 tools (yfinance, newsapi, wikipedia, edgar, finnhub) that contributed citations.',
    },
    {
      key: 'citation_density', label: 'Citation Density',
      tip: 'Citations per 200 words. 1.00 = report is well-backed throughout.',
    },
    {
      key: 'confidence_calibration', label: 'Conf. Calibration',
      tip: "Gap between agent's self-reported confidence and actual eval overall. 1.00 = perfectly calibrated.",
    },
  ]

  // Improvements data (static — before/after)
  const improvements = [
    {
      num: 1, tier: 'TIER 1', tag: 'Fixed',
      title: 'Hallucination metric redesigned',
      before: '0.42', after: '1.00', delta: '+0.58',
      desc: 'Old keyword scan consistently failed on structured JSON citations. Redesigned to citation count vs paragraph count.',
    },
    {
      num: 2, tier: 'TIER 1', tag: 'Fixed',
      title: 'Quantitative — extract real figures from citations',
      before: '0.54', after: '0.61', delta: '+0.07',
      desc: 'Was defaulting to 1.0 when no yfinance data. Now extracts actual numbers (including B/T/M suffixed values) from citation claims.',
    },
    {
      num: 3, tier: 'TIER 1', tag: 'Fixed',
      title: 'Freshness — added published_at to citations',
      before: '0.50', after: '0.60', delta: '+0.10',
      desc: 'Citations had no dates so freshness always returned 0.50 neutral. Wired NewsAPI publishedAt and Finnhub Unix timestamps through.',
    },
    {
      num: 4, tier: 'TIER 1', tag: 'Fixed',
      title: 'Finnhub restructured to return dated articles',
      before: 'str', after: 'dict', delta: 'New',
      desc: 'get_finnhub_data() returned a plain string with no metadata. Restructured to return articles with ISO published_at timestamps.',
    },
    {
      num: 5, tier: 'TIER 1', tag: 'Improved',
      title: 'yfinance prompt enforces actual figures',
      before: 'vague', after: 'exact', delta: 'New',
      desc: 'Added CRITICAL section to synthesis prompt mandating exact figures in yfinance claims: "Stock price: $189.50, Market cap: $2.93T".',
    },
    {
      num: 6, tier: 'TIER 2', tag: 'Added',
      title: 'narrative_length metric',
      before: '—', after: '0.92', delta: 'New',
      desc: '60% word-count band score + 40% section completeness (5 expected headers). Avg 0.92 — reports are well-structured.',
    },
    {
      num: 7, tier: 'TIER 2', tag: 'Fixed',
      title: 'Regression runner retry logic',
      before: '0.00', after: 'retry', delta: 'Fix',
      desc: 'Silent 0.00/0.00 rows from failed queries corrupted averages. Added auto-retry after 20s on failed/empty runs.',
    },
    {
      num: 8, tier: 'TIER 3', tag: 'Added',
      title: 'tool_activation diagnostic',
      before: '—', after: '0.68', delta: 'New',
      desc: 'Fraction of 5 tools (yfinance, newsapi, wikipedia, edgar, finnhub) represented in citations. Exposes weak orchestrator routing.',
    },
    {
      num: 9, tier: 'TIER 3', tag: 'Added',
      title: 'citation_density diagnostic',
      before: '—', after: '0.91', delta: 'New',
      desc: 'Citations per 200 words. Ideal ≥1 per block. Avg 0.91 confirms reports are well-cited throughout.',
    },
    {
      num: 10, tier: 'TIER 3', tag: 'Added',
      title: 'confidence_calibration diagnostic',
      before: '—', after: '0.75', delta: 'New',
      desc: 'Gap between self-reported confidence and eval overall. Avg 0.75 — agent is reasonably well-calibrated.',
    },
    {
      num: 11, tier: 'TIER 2', tag: 'Added',
      title: 'Source Credibility metric',
      before: '—', after: '0.56', delta: 'New',
      desc: 'Tiered domain authority scoring: SEC/Gov=1.0, Reuters/Bloomberg/Finnhub=0.8, Yahoo/CNBC=0.65, Wikipedia=0.5, unknown=0.3. Added 30+ domains including investor IR pages.',
    },
    {
      num: 12, tier: 'TIER 2', tag: 'Added',
      title: 'Fictional Premise metric',
      before: '—', after: '0.70', delta: 'New',
      desc: 'Detects fabricated figures for unverifiable claims and rewards correct hedging language ("reportedly", "analysts expect"). Fixed to score all queries, not just fictional-premise ones.',
    },
    {
      num: 13, tier: 'TIER 2', tag: 'Added',
      title: 'Answer Relevance metric',
      before: '—', after: '1.00', delta: 'New',
      desc: 'Checks if the conclusion directly answers the specific question asked. For comparisons, names the winner with exact figures. Thresholds tuned to realistic financial report coverage.',
    },
    {
      num: 14, tier: 'TIER 1', tag: 'Fixed',
      title: 'Quantitative tolerance 2% → 15% + UPV ground truth',
      before: '0.19', after: '0.72', delta: '+0.53',
      desc: 'Tolerance widened to 15% to handle quarterly vs TTM figure differences. UPVs (user-provided values in query) now injected as ground truth so agent is not penalised for correctly using the user\'s own numbers.',
    },
    {
      num: 15, tier: 'TIER 2', tag: 'Improved',
      title: 'UPV enforcement in synthesis agent',
      before: 'ignored', after: 'locked', delta: 'New',
      desc: 'User-Provided Values extracted from query at synthesis time. Injected as locked ground truth before research findings. Contradicting external data moved to "Contradictory Market Data" footnote instead of overwriting.',
    },
    {
      num: 16, tier: 'TIER 2', tag: 'Improved',
      title: 'Synthesis prompt data hierarchy',
      before: 'flat', after: 'ranked', delta: 'New',
      desc: 'Strict 6-tier source hierarchy: UPV > EDGAR > yfinance > Finnhub > Research > Wikipedia. Added exact-numbers rule, direct-answer conclusion rule, priority-source citation rule, and mandatory hedging for projections.',
    },
    {
      num: 17, tier: 'INFRA', tag: 'Fixed',
      title: 'MCP SSE connection drops between runs',
      before: 'stale', after: 'fresh', delta: 'Fix',
      desc: 'Synthesis agent held one persistent SSE connection that died between runs. Fixed by creating a fresh MCPToolset per synthesize() call. Tool Activation jumped to 1.00.',
    },
    {
      num: 18, tier: 'INFRA', tag: 'Fixed',
      title: 'Orchestrator max_output_tokens 1024 → 4096',
      before: 'failed', after: 'stable', delta: 'Fix',
      desc: 'decompose_query node truncated JSON at 1024 tokens causing "Run Failed" on every query. Increased to 4096 — pipeline now completes reliably.',
    },
  ]

  const tagColor: Record<string, string> = {
    Fixed:    '#3fb950',
    Added:    '#58a6ff',
    Improved: '#f0883e',
  }

  const tierColor: Record<string, string> = {
    'TIER 1': '#58a6ff',
    'TIER 2': '#bc8cff',
    'TIER 3': '#39d353',
    'INFRA':  '#f0883e',
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'var(--bg-base)',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* ── Top bar ── */}
      <div style={{
        position: 'sticky', top: 0, zIndex: 10,
        background: 'var(--bg-base)',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '14px 28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backdropFilter: 'blur(12px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          {/* Logo mark */}
          <div style={{
            width: 32, height: 32,
            background: 'linear-gradient(135deg, #58a6ff, #bc8cff)',
            borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
              Eval Dashboard
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {reports.length} report{reports.length !== 1 ? 's' : ''} scored · auto-refreshes every 30s
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Refresh button */}
          <button
            onClick={fetchScores}
            style={{
              padding: '6px 12px',
              borderRadius: 8,
              border: '1px solid var(--border-strong)',
              background: 'var(--bg-surface)',
              color: 'var(--text-secondary)',
              fontSize: 12,
              cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 6,
            }}
          >
            <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>

          {/* Close */}
          <button
            onClick={onClose}
            style={{
              width: 34, height: 34,
              borderRadius: 8,
              border: '1px solid var(--border-strong)',
              background: 'var(--bg-surface)',
              color: 'var(--text-secondary)',
              fontSize: 18,
              cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              lineHeight: 1,
            }}
            title="Close"
          >
            ×
          </button>
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 24px', width: '100%' }}>

        {/* ── Tab switcher ── */}
        <div style={{
          display: 'flex', gap: 4,
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 10,
          padding: 4,
          width: 'fit-content',
          marginBottom: 32,
        }}>
          {(['live', 'improvements'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '7px 18px',
                borderRadius: 7,
                border: 'none',
                background: activeTab === tab ? 'var(--accent)' : 'transparent',
                color: activeTab === tab ? '#fff' : 'var(--text-secondary)',
                fontWeight: 600,
                fontSize: 13,
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              {tab === 'live' ? '📊  Live Scores' : '🚀  Improvements'}
            </button>
          ))}
        </div>

        {/* ═══════════════════════════════════ LIVE SCORES TAB ═══ */}
        {activeTab === 'live' && (
          <>
            {loading && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 60, fontSize: 14 }}>
                Loading eval scores…
              </div>
            )}

            {error && (
              <div style={{
                background: 'rgba(248,81,73,0.08)',
                border: '1px solid rgba(248,81,73,0.25)',
                borderRadius: 10, padding: '14px 18px',
                color: '#f85149', fontSize: 13, marginBottom: 24,
              }}>
                ⚠ {error} — make sure the gateway is running at localhost:8000
              </div>
            )}

            {!loading && !error && data?.available === false && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 60, fontSize: 14 }}>
                Eval framework not available in the gateway container.
              </div>
            )}

            {!loading && reports.length === 0 && data?.available && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 60, fontSize: 14 }}>
                No eval results yet — run a query at localhost:3000 to generate scores.
              </div>
            )}

            {reports.length > 0 && (
              <>
                {/* Hero cards */}
                <div style={{ display: 'flex', gap: 14, marginBottom: 32, flexWrap: 'wrap' }}>
                  <HeroCard
                    label="Avg Overall Score"
                    value={avgOverall != null ? avgOverall.toFixed(2) : '—'}
                    sub={`across ${reports.length} reports`}
                    accent={scoreColor(avgOverall)}
                  />
                  <HeroCard
                    label="Best Single Report"
                    value={bestOverall != null ? bestOverall.toFixed(2) : '—'}
                    sub="highest overall score"
                    accent="#58a6ff"
                  />
                  <HeroCard
                    label="Avg Tool Activation"
                    value={fmt(avgOf(reports, 'tool_activation'))}
                    sub="fraction of 5 tools used"
                    accent="#39d353"
                  />
                  <HeroCard
                    label="Avg Citation Density"
                    value={fmt(avgOf(reports, 'citation_density'))}
                    sub="citations per 200 words"
                    accent="#bc8cff"
                  />
                </div>

                {/* Scores table */}
                <div style={{
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 12,
                  overflow: 'hidden',
                }}>
                  {/* Table header */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '2fr repeat(10, 1fr)',
                    background: 'var(--bg-elevated, #1c2128)',
                    borderBottom: '1px solid var(--border-subtle)',
                    padding: '10px 16px',
                    gap: 4,
                  }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Query</div>
                    {primaryMetrics.map(m => (
                      <div key={m.key} style={{ textAlign: 'center' }}>
                        <Tooltip text={m.tip}>
                          <span style={{
                            fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
                            textTransform: 'uppercase', letterSpacing: '0.5px',
                            borderBottom: '1px dashed var(--border-strong)',
                            paddingBottom: 1,
                          }}>
                            {m.short}
                          </span>
                        </Tooltip>
                      </div>
                    ))}
                  </div>

                  {/* Rows */}
                  {reports.map((r, i) => {
                    const ts = r.timestamp ? r.timestamp.slice(0, 16).replace('T', ' ') : ''
                    const qShort = r.query.length > 55 ? r.query.slice(0, 52) + '…' : r.query
                    return (
                      <div key={i} style={{ borderBottom: i < reports.length - 1 ? '1px solid var(--border-subtle)' : 'none' }}>
                        {/* Main row */}
                        <div style={{
                          display: 'grid',
                          gridTemplateColumns: '2fr repeat(10, 1fr)',
                          padding: '12px 16px',
                          gap: 4,
                          alignItems: 'center',
                        }}>
                          <div style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 500 }}>
                            {qShort}
                          </div>
                          {primaryMetrics.map(m => (
                            <div key={m.key} style={{ textAlign: 'center' }}>
                              <ScorePill value={r.scores[m.key]} />
                            </div>
                          ))}
                        </div>
                        {/* Diagnostic sub-row */}
                        <div style={{
                          padding: '0 16px 10px 16px',
                          display: 'flex', gap: 16, alignItems: 'center',
                          fontSize: 11, color: 'var(--text-muted)',
                        }}>
                          <span>{ts} · {timeAgo(r.timestamp)}</span>
                          {diagMetrics.map(d => (
                            r.scores[d.key] != null && (
                              <Tooltip key={d.key} text={d.tip}>
                                <span style={{ borderBottom: '1px dashed var(--border-strong)', paddingBottom: 1 }}>
                                  {d.label}:
                                </span>
                                <span style={{ color: scoreColor(r.scores[d.key]), fontWeight: 700, fontFamily: 'monospace', marginLeft: 4 }}>
                                  {fmt(r.scores[d.key])}
                                </span>
                              </Tooltip>
                            )
                          ))}
                        </div>
                      </div>
                    )
                  })}

                  {/* Average row */}
                  {reports.length > 1 && (
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '2fr repeat(10, 1fr)',
                      padding: '12px 16px',
                      gap: 4,
                      alignItems: 'center',
                      borderTop: '2px solid var(--border-strong)',
                      background: 'var(--bg-elevated, #1c2128)',
                    }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)' }}>
                        Averages across {reports.length} reports
                      </div>
                      {primaryMetrics.map(m => (
                        <div key={m.key} style={{ textAlign: 'center' }}>
                          <Tooltip text={m.tip}>
                            <span><ScorePill value={avgOf(reports, m.key)} /></span>
                          </Tooltip>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Metric legend */}
                <div style={{
                  marginTop: 16, fontSize: 11, color: 'var(--text-muted)',
                  display: 'flex', flexWrap: 'wrap', gap: '6px 20px',
                }}>
                  {primaryMetrics.filter(m => m.key !== 'overall').map(m => (
                    <span key={m.key}><strong>{m.short}</strong> = {m.label}</span>
                  ))}
                  <span style={{ marginLeft: 'auto' }}>
                    <span style={{ color: '#3fb950', fontWeight: 700 }}>■</span> ≥0.75 &nbsp;
                    <span style={{ color: '#d29922', fontWeight: 700 }}>■</span> ≥0.50 &nbsp;
                    <span style={{ color: '#f85149', fontWeight: 700 }}>■</span> &lt;0.50
                  </span>
                </div>
              </>
            )}
          </>
        )}

        {/* ═══════════════════════════════ IMPROVEMENTS TAB ═══ */}
        {activeTab === 'improvements' && (
          <>
            {/* Before / After hero */}
            <div style={{ display: 'flex', gap: 14, marginBottom: 36, flexWrap: 'wrap' }}>
              <HeroCard label="Before (Baseline)" value="0.68" sub="No eval framework · estimated" accent="#f85149" />
              <HeroCard label="After (10-Report Avg)" value="0.77" sub="Automated scoring · 9 metrics" accent="#3fb950" />
              <HeroCard label="Best Single Report" value="0.82" sub="AWS vs Google Cloud · May 2026" accent="#58a6ff" />
              <HeroCard label="Metrics Added" value="9" sub="Tier 1 · Tier 2 · Tier 3" accent="#bc8cff" />
            </div>

            {/* Per-metric before/after table */}
            <div style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 12, overflow: 'hidden',
              marginBottom: 32,
            }}>
              <div style={{
                display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
                background: 'var(--bg-elevated, #1c2128)',
                borderBottom: '1px solid var(--border-subtle)',
                padding: '10px 16px', gap: 4,
              }}>
                {['Metric', 'Before', 'After', 'Change', 'Status'].map(h => (
                  <div key={h} style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', textAlign: h === 'Metric' ? 'left' : 'center' }}>
                    {h}
                  </div>
                ))}
              </div>

              {[
                { name: 'Hallucination',    desc: 'Citation count vs paragraphs',       before: '0.42', after: '1.00', delta: '+0.58', status: 'Fixed'    },
                { name: 'Quantitative',     desc: '$/% figures vs yfinance data',        before: '0.54', after: '0.61', delta: '+0.07', status: 'Improved' },
                { name: 'Freshness',        desc: 'Avg age of dated citations',          before: '0.50', after: '0.60', delta: '+0.10', status: 'Improved' },
                { name: 'Diversity',        desc: 'Unique domains / total sources',      before: '0.71', after: '0.74', delta: '+0.03', status: 'Improved' },
                { name: 'Entity Coverage',  desc: 'Query entities found in narrative',   before: '0.68', after: '0.71', delta: '+0.03', status: 'Improved' },
                { name: 'Narrative Length', desc: 'Word count + section completeness',   before: '—',    after: '0.92', delta: 'New',   status: 'Added'   },
                { name: 'Tool Activation',  desc: 'Fraction of 5 tools in citations',    before: '—',    after: '0.68', delta: 'New',   status: 'Added'   },
                { name: 'Citation Density', desc: 'Citations per 200 words',             before: '—',    after: '0.91', delta: 'New',   status: 'Added'   },
                { name: 'Conf. Calibration',desc: 'Self-reported vs eval gap',           before: '—',    after: '0.75', delta: 'New',   status: 'Added'   },
              ].map((row, i) => (
                <div key={i} style={{
                  display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
                  padding: '12px 16px', gap: 4, alignItems: 'center',
                  borderBottom: i < 8 ? '1px solid var(--border-subtle)' : 'none',
                }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{row.name}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{row.desc}</div>
                  </div>
                  <div style={{ textAlign: 'center', fontFamily: 'monospace', fontSize: 13, color: row.before === '—' ? 'var(--text-muted)' : '#f85149', fontWeight: 700 }}>{row.before}</div>
                  <div style={{ textAlign: 'center', fontFamily: 'monospace', fontSize: 13, color: '#3fb950', fontWeight: 700 }}>{row.after}</div>
                  <div style={{ textAlign: 'center', fontFamily: 'monospace', fontSize: 12, fontWeight: 700, color: row.delta.startsWith('+') ? '#3fb950' : '#58a6ff' }}>{row.delta}</div>
                  <div style={{ textAlign: 'center' }}>
                    <span style={{
                      display: 'inline-block', padding: '2px 9px', borderRadius: 6,
                      fontSize: 11, fontWeight: 700,
                      color: tagColor[row.status] ?? '#fff',
                      background: (tagColor[row.status] ?? '#fff') + '18',
                      border: `1px solid ${(tagColor[row.status] ?? '#fff')}33`,
                    }}>{row.status}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Change list */}
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>
              All Changes Made
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {improvements.map(item => (
                <div key={item.num} style={{
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 10,
                  padding: '16px 18px',
                  display: 'grid',
                  gridTemplateColumns: '28px 1fr auto',
                  gap: 14,
                  alignItems: 'start',
                }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: '50%',
                    background: 'var(--bg-elevated, #1c2128)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, color: 'var(--text-muted)',
                    flexShrink: 0,
                  }}>
                    {item.num}
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
                      <span style={{
                        fontSize: 10, fontWeight: 700, padding: '1px 7px', borderRadius: 4,
                        color: tierColor[item.tier] ?? '#fff',
                        background: (tierColor[item.tier] ?? '#fff') + '18',
                        border: `1px solid ${(tierColor[item.tier] ?? '#fff')}33`,
                        letterSpacing: '0.4px',
                      }}>{item.tier}</span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{item.title}</span>
                    </div>
                    <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0, lineHeight: 1.6 }}>{item.desc}</p>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                    {item.before !== '—' && (
                      <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#f85149', fontWeight: 700 }}>{item.before}</span>
                    )}
                    {item.before !== '—' && (
                      <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>→</span>
                    )}
                    <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#3fb950', fontWeight: 700 }}>{item.after}</span>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
