/**
 * components/PolicyRace.jsx
 * Runs all 4 policies simultaneously on the same arm config, overlays cumulative reward.
 */
import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { useBandit } from '../hooks/useBandit'

const RACE_POLICIES = [
  { id: 'epsilon-greedy', label: 'e-Greedy', color: '#E89320' },
  { id: 'ucb',            label: 'UCB1',     color: '#378ADD' },
  { id: 'thompson-sampling', label: 'Thompson', color: '#1D9E75' },
  { id: 'linucb',         label: 'LinUCB',   color: '#D85A30' },
]

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

export default function PolicyRace({ armConfig, driftFn }) {
  const eg = useBandit('epsilon-greedy', armConfig, driftFn)
  const ucb = useBandit('ucb', armConfig, driftFn)
  const ts = useBandit('thompson-sampling', armConfig, driftFn)
  const lin = useBandit('linucb', armConfig, driftFn)
  const hooks = [eg, ucb, ts, lin]

  const [autoOn, setAutoOn] = useState(false)
  const [speed, setSpeed] = useState(5)
  const intervalRef = useRef(null)

  const pullAll = useCallback(async () => {
    await Promise.all(hooks.map(h => h.pull()))
  }, [eg.pull, ucb.pull, ts.pull, lin.pull]) // eslint-disable-line

  const stopAuto = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
    setAutoOn(false)
  }, [])

  const startAuto = useCallback(() => {
    intervalRef.current = setInterval(() => pullAll(), Math.round(1000 / speed))
    setAutoOn(true)
  }, [pullAll, speed])

  const toggleAuto = () => autoOn ? stopAuto() : startAuto()

  useEffect(() => {
    if (autoOn) { stopAuto(); startAuto() }
  }, [speed]) // eslint-disable-line

  useEffect(() => () => stopAuto(), []) // eslint-disable-line

  const handleReset = async () => {
    stopAuto()
    await Promise.all(hooks.map(h => h.reset()))
  }

  // Merge histories for overlay chart
  const maxLen = Math.max(...hooks.map(h => h.state.hist.length))
  const merged = []
  const step = Math.max(1, Math.floor(maxLen / 150))
  for (let i = 0; i < maxLen; i++) {
    if (i % step !== 0 && i !== maxLen - 1) continue
    const point = { t: i + 1 }
    hooks.forEach((h, j) => {
      const entry = h.state.hist[i]
      point[RACE_POLICIES[j].id] = entry ? entry.cumR : null
    })
    // Random baseline from first hook that has data at this index
    const entry = hooks[0].state.hist[i]
    point.random = entry ? entry.cumRandom : null
    merged.push(point)
  }

  // Summary table
  const pulls = hooks[0].state.pulls
  const summary = RACE_POLICIES.map((p, i) => ({
    ...p,
    pulls: hooks[i].state.pulls,
    cumR: hooks[i].state.cumR,
    cumReg: hooks[i].state.cumReg,
  })).sort((a, b) => b.cumR - a.cumR)

  return (
    <div>
      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
        <button
          onClick={toggleAuto}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: 11, padding: '5px 12px',
            borderRadius: 'var(--border-radius-md)',
            border: `0.5px solid #1D9E75`,
            background: autoOn ? '#1D9E75' : 'transparent',
            color: autoOn ? '#fff' : '#1D9E75',
            cursor: 'pointer',
          }}
        >
          {autoOn ? '⏸ Running' : '▶ Race'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          <span>Speed</span>
          <input type="range" min="1" max="20" value={speed} step="1"
            onChange={e => setSpeed(Number(e.target.value))} style={{ width: 80 }} />
          <span>{speed}/s</span>
        </div>

        <button onClick={handleReset} style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '5px 12px', borderRadius: 'var(--border-radius-md)', border: '0.5px solid var(--color-border-secondary)', background: 'transparent', cursor: 'pointer', color: 'var(--color-text-secondary)' }}>
          ⟳ Reset
        </button>

        <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
          {pulls} pulls
        </span>
      </div>

      {/* Overlay chart */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
          cumulative reward — all policies
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={merged}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
            <XAxis dataKey="t" tick={TICK_STYLE} axisLine={false} tickLine={false} />
            <YAxis tick={TICK_STYLE} axisLine={false} tickLine={false} width={36} />
            <Tooltip
              contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              formatter={v => [v !== null ? v.toFixed(2) : '—']}
            />
            {RACE_POLICIES.map(p => (
              <Line key={p.id} type="monotone" dataKey={p.id} name={p.label} stroke={p.color} dot={false} strokeWidth={1.5} connectNulls />
            ))}
            <Line type="monotone" dataKey="random" name="Random" stroke="#8b949e" dot={false} strokeWidth={1} strokeDasharray="2 4" opacity={0.6} />
            <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font-mono)', paddingTop: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary table */}
      {pulls > 0 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
            results
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr 1fr', gap: '4px 16px', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: 'var(--color-text-secondary)' }}>#</span>
            <span style={{ color: 'var(--color-text-secondary)' }}>policy</span>
            <span style={{ color: 'var(--color-text-secondary)' }}>reward</span>
            <span style={{ color: 'var(--color-text-secondary)' }}>regret</span>
            {summary.map((s, i) => (
              <React.Fragment key={s.id}>
                <span style={{ color: 'var(--color-text-secondary)' }}>{i + 1}</span>
                <span style={{ color: s.color }}>{s.label}</span>
                <span>{s.cumR.toFixed(2)}</span>
                <span style={{ color: '#E24B4A' }}>{s.cumReg.toFixed(2)}</span>
              </React.Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
