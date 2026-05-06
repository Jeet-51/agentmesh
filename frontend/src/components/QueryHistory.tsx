import { useState } from 'react'
import type { HistoryItem } from '../types'

interface Props {
  history: HistoryItem[]
  onSelect: (item: HistoryItem) => void
  onDelete: (id: string) => void
  onClearAll: () => void
  open: boolean
  onClose: () => void
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const m = Math.floor(diff / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

// ─── Confidence pill ──────────────────────────────────────────────────────────

function ConfidencePill({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = pct >= 80 ? 'var(--green)' : pct >= 60 ? 'var(--amber)' : 'var(--red)'
  const bg    = pct >= 80
    ? 'rgba(34,211,160,0.1)'
    : pct >= 60
    ? 'rgba(245,166,35,0.1)'
    : 'rgba(242,92,92,0.1)'

  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 20,
      background: bg, color,
      fontVariantNumeric: 'tabular-nums',
    }}>
      {pct}%
    </span>
  )
}

// ─── History item ─────────────────────────────────────────────────────────────

interface HistoryRowProps {
  item: HistoryItem
  onSelect: () => void
  onDelete: () => void
}

function HistoryRow({ item, onSelect, onDelete }: HistoryRowProps) {
  const [hover, setHover] = useState(false)

  return (
    <div
      onClick={onSelect}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: 'relative', borderRadius: 10, padding: '10px 14px',
        cursor: 'pointer', transition: 'background 0.12s ease',
        background: hover ? 'var(--bg-hover)' : 'var(--bg-elevated)',
        border: '1px solid var(--border)',
      }}
    >
      {/* Query text */}
      <p style={{
        fontSize: 13, fontWeight: 500, color: 'var(--text-primary)',
        margin: '0 0 6px', lineHeight: 1.45,
        paddingRight: hover ? 22 : 0,
        overflow: 'hidden', display: '-webkit-box',
        WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
      }}>
        {item.query}
      </p>

      {/* Meta row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <ConfidencePill score={item.confidence} />
          {item.status === 'partial' && (
            <span style={{ fontSize: 10, color: 'var(--amber)', fontWeight: 500 }}>partial</span>
          )}
        </div>
        <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
          {timeAgo(item.timestamp)}
        </span>
      </div>

      {/* Trash icon — visible on hover */}
      {hover && (
        <button
          onClick={e => { e.stopPropagation(); onDelete() }}
          title="Delete"
          style={{
            position: 'absolute', top: 8, right: 8,
            width: 22, height: 22, borderRadius: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text-tertiary)', transition: 'color 0.12s ease',
          }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--red)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-tertiary)')}
        >
          <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
          </svg>
        </button>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function QueryHistory({ history, onSelect, onDelete, onClearAll, open, onClose }: Props) {
  const [clearHover, setClearHover] = useState(false)

  if (!open) return null

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, zIndex: 40,
          background: 'rgba(0,0,0,0.45)',
          backdropFilter: 'blur(4px)',
          WebkitBackdropFilter: 'blur(4px)',
        }}
      />

      {/* Sidebar panel — slides in from LEFT */}
      <div
        className="animate-slide-left"
        style={{
          position: 'fixed', top: 0, left: 0, bottom: 0, zIndex: 50,
          width: 300, display: 'flex', flexDirection: 'column',
          background: 'var(--bg-surface)',
          borderRight: '1px solid var(--border-strong)',
          boxShadow: 'var(--shadow-lg)',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '16px 16px 14px',
          borderBottom: '1px solid var(--border)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"
              style={{ color: 'var(--accent)' }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
              Research History
            </span>
          </div>
          <button
            onClick={onClose}
            style={{
              width: 28, height: 28, borderRadius: 8, border: 'none',
              background: 'none', cursor: 'pointer', display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              color: 'var(--text-tertiary)', fontSize: 16,
              transition: 'color 0.12s ease, background 0.12s ease',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.color = 'var(--text-primary)'
              e.currentTarget.style.background = 'var(--bg-elevated)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.color = 'var(--text-tertiary)'
              e.currentTarget.style.background = 'none'
            }}
          >
            ✕
          </button>
        </div>

        {/* List */}
        <div style={{
          flex: 1, overflowY: 'auto', padding: '10px 12px',
          display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          {history.length === 0 ? (
            <div style={{ textAlign: 'center', paddingTop: 48 }}>
              <p style={{ fontSize: 14, color: 'var(--text-tertiary)', margin: '0 0 4px' }}>No history yet</p>
              <p style={{ fontSize: 12, color: 'var(--text-tertiary)', margin: 0, opacity: 0.65 }}>
                Completed queries will appear here
              </p>
            </div>
          ) : (
            history.map(item => (
              <HistoryRow
                key={item.id}
                item={item}
                onSelect={() => { onSelect(item); onClose() }}
                onDelete={() => onDelete(item.id)}
              />
            ))
          )}
        </div>

        {/* Footer */}
        {history.length > 0 && (
          <div style={{ padding: '10px 12px', borderTop: '1px solid var(--border)' }}>
            <button
              onClick={onClearAll}
              onMouseEnter={() => setClearHover(true)}
              onMouseLeave={() => setClearHover(false)}
              style={{
                width: '100%', padding: '9px', borderRadius: 10, fontSize: 13,
                fontWeight: 500, fontFamily: 'inherit', cursor: 'pointer',
                transition: 'all 0.15s ease',
                background: clearHover ? 'rgba(242,92,92,0.1)' : 'none',
                border: '1px solid var(--border)',
                color: clearHover ? 'var(--red)' : 'var(--text-tertiary)',
              }}
            >
              Clear all history
            </button>
          </div>
        )}
      </div>
    </>
  )
}
