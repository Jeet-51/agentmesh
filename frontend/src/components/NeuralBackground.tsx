import { useEffect, useRef } from 'react'

// ─── Constants ────────────────────────────────────────────────────────────────

const NODE_COUNT   = 28
const MAX_CONNS    = 4        // max connections per node (K-nearest)
const MAX_PULSES   = 6        // simultaneous travelling pulses
const CONN_DIST    = 220      // max pixel distance to draw a connection
const NODE_SPEED   = 0.22     // max drift speed (px/frame at 30fps)
const PULSE_SPEED  = 0.0004   // fraction of connection length per ms
const CONN_RECALC  = 300      // ms between K-NN recalculation
const FRAME_MS     = 1000 / 30  // 30fps cap

// Colours at very low opacity
const INDIGO  = 'rgba(99,102,241,'   // #6366f1
const CYAN    = 'rgba(6,182,212,'    // #06b6d4

// ─── Types ────────────────────────────────────────────────────────────────────

interface Node {
  x: number; y: number
  vx: number; vy: number
}

interface Conn {
  a: number; b: number   // node indices
}

interface Pulse {
  conn: Conn
  t: number              // 0..1 progress along connection
  color: string          // INDIGO or CYAN base
  forward: boolean
}

interface Props {
  theme: 'dark' | 'light'
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function NeuralBackground({ theme }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const themeRef  = useRef(theme)

  // Keep themeRef in sync without restarting the animation loop
  useEffect(() => { themeRef.current = theme }, [theme])

  useEffect(() => {
    const maybeCanvas = canvasRef.current
    if (!maybeCanvas) return
    const canvas: HTMLCanvasElement = maybeCanvas

    // Use alpha:false for faster compositing
    const maybeCtx = canvas.getContext('2d', { alpha: false })
    if (!maybeCtx) return

    // Reassign as non-nullable so TypeScript keeps the type in closures
    const ctx: CanvasRenderingContext2D = maybeCtx

    // ── State ─────────────────────────────────────────────────────────────────

    let W = 0, H = 0
    const nodes: Node[]  = []
    let   conns: Conn[]  = []
    const pulses: Pulse[] = []

    let lastFrame    = 0
    let lastConnCalc = 0
    let rafId        = 0

    // ── Init nodes ────────────────────────────────────────────────────────────

    function initNodes() {
      nodes.length = 0
      for (let i = 0; i < NODE_COUNT; i++) {
        const angle = Math.random() * Math.PI * 2
        const speed = NODE_SPEED * (0.3 + Math.random() * 0.7)
        nodes.push({
          x: Math.random() * W,
          y: Math.random() * H,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
        })
      }
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    function resize() {
      const dpr = Math.min(devicePixelRatio, 2)
      W = window.innerWidth
      H = window.innerHeight
      canvas.width  = W * dpr
      canvas.height = H * dpr
      canvas.style.width  = W + 'px'
      canvas.style.height = H + 'px'
      ctx.scale(dpr, dpr)
      if (nodes.length === 0) initNodes()
    }

    // ── K-nearest connections ─────────────────────────────────────────────────

    function updateConns(now: number) {
      if (now - lastConnCalc < CONN_RECALC) return
      lastConnCalc = now

      const connCount = new Array(NODE_COUNT).fill(0)
      const next: Conn[] = []

      // Build candidate pairs sorted by distance
      const pairs: { a: number; b: number; d: number }[] = []
      for (let i = 0; i < NODE_COUNT; i++) {
        for (let j = i + 1; j < NODE_COUNT; j++) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const d  = Math.sqrt(dx * dx + dy * dy)
          if (d < CONN_DIST) pairs.push({ a: i, b: j, d })
        }
      }
      pairs.sort((x, y) => x.d - y.d)

      for (const p of pairs) {
        if (connCount[p.a] >= MAX_CONNS || connCount[p.b] >= MAX_CONNS) continue
        next.push({ a: p.a, b: p.b })
        connCount[p.a]++
        connCount[p.b]++
      }

      conns = next
    }

    // ── Spawn pulses ──────────────────────────────────────────────────────────

    function maybeSpawnPulse() {
      if (pulses.length >= MAX_PULSES || conns.length === 0) return
      if (Math.random() > 0.04) return  // ~4% chance per frame to spawn
      const conn    = conns[Math.floor(Math.random() * conns.length)]
      const isIndigo = Math.random() > 0.45
      pulses.push({
        conn,
        t: 0,
        color: isIndigo ? INDIGO : CYAN,
        forward: Math.random() > 0.5,
      })
    }

    // ── Draw one frame ────────────────────────────────────────────────────────

    function draw(dt: number) {
      const isDark  = themeRef.current === 'dark'
      const bg      = isDark ? '#080810' : '#fafafa'
      const nodeFill = isDark ? 'rgba(99,102,241,0.35)' : 'rgba(99,102,241,0.25)'
      const lineBase = isDark ? 0.55 : 0.35   // max connection opacity

      // Background
      ctx.fillStyle = bg
      ctx.fillRect(0, 0, W, H)

      // Move nodes
      for (const n of nodes) {
        n.x += n.vx * dt
        n.y += n.vy * dt
        if (n.x < 0)  { n.x = 0;  n.vx = Math.abs(n.vx) }
        if (n.x > W)  { n.x = W;  n.vx = -Math.abs(n.vx) }
        if (n.y < 0)  { n.y = 0;  n.vy = Math.abs(n.vy) }
        if (n.y > H)  { n.y = H;  n.vy = -Math.abs(n.vy) }
      }

      // Draw connections
      for (const c of conns) {
        const na  = nodes[c.a], nb = nodes[c.b]
        const dx  = nb.x - na.x, dy = nb.y - na.y
        const d   = Math.sqrt(dx * dx + dy * dy)
        const op  = lineBase * (1 - d / CONN_DIST)
        ctx.beginPath()
        ctx.moveTo(na.x, na.y)
        ctx.lineTo(nb.x, nb.y)
        ctx.strokeStyle = `rgba(99,102,241,${op.toFixed(3)})`
        ctx.lineWidth   = 0.6
        ctx.stroke()
      }

      // Draw pulses
      for (let i = pulses.length - 1; i >= 0; i--) {
        const p  = pulses[i]
        p.t += PULSE_SPEED * dt
        if (p.t > 1) { pulses.splice(i, 1); continue }

        const na  = nodes[p.conn.a], nb = nodes[p.conn.b]
        const t   = p.forward ? p.t : 1 - p.t
        const px  = na.x + (nb.x - na.x) * t
        const py  = na.y + (nb.y - na.y) * t

        // Glow gradient
        const r  = 6
        const grd = ctx.createRadialGradient(px, py, 0, px, py, r)
        const opMax = isDark ? 0.9 : 0.6
        const opMid = isDark ? 0.45 : 0.25
        grd.addColorStop(0,   p.color + opMax + ')')
        grd.addColorStop(0.5, p.color + opMid + ')')
        grd.addColorStop(1,   p.color + '0)')
        ctx.beginPath()
        ctx.arc(px, py, r, 0, Math.PI * 2)
        ctx.fillStyle = grd
        ctx.fill()
      }

      // Draw nodes (small dots)
      for (const n of nodes) {
        ctx.beginPath()
        ctx.arc(n.x, n.y, 2.2, 0, Math.PI * 2)
        ctx.fillStyle = nodeFill
        ctx.fill()
      }
    }

    // ── Animation loop ────────────────────────────────────────────────────────

    function loop(ts: number) {
      if (document.hidden) { rafId = requestAnimationFrame(loop); return }

      const dt = ts - lastFrame
      if (dt < FRAME_MS) { rafId = requestAnimationFrame(loop); return }

      lastFrame = ts
      updateConns(ts)
      maybeSpawnPulse()
      draw(Math.min(dt, 80))   // cap delta to avoid huge jumps after tab switch

      rafId = requestAnimationFrame(loop)
    }

    // ── Boot ──────────────────────────────────────────────────────────────────

    resize()
    window.addEventListener('resize', resize)
    rafId = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        display: 'block',
      }}
    />
  )
}
