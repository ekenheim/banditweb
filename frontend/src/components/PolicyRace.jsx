/**
 * components/PolicyRace.jsx
 * Runs selected policies simultaneously on the same arm config, overlays cumulative reward.
 */
import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { useBandit } from '../hooks/useBandit'

const RACE_POLICIES = [
  { id: 'epsilon-greedy',    label: '\u03b5-Greedy',  color: '#E89320' },
  { id: 'ucb',               label: 'UCB1',       color: '#378ADD' },
  { id: 'thompson-sampling', label: 'Thompson',   color: '#1D9E75' },
  { id: 'bayesian-ucb',      label: 'Bayes UCB',  color: '#9B59B6' },
  { id: 'exp3',              label: 'EXP3',       color: '#E74C3C' },
  { id: 'linucb',            label: 'LinUCB',     color: '#D85A30' },
  { id: 'lints',             label: 'LinTS',      color: '#F39C12' },
]

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

// ── Per-policy runner — renders nothing but owns its useBandit hook ────────
// Reports state changes to the parent via onStateChange callback.

function RacePolicyRunner({ policyId, armConfig, driftFn, onStateChange }) {
  const bandit = useBandit(policyId, armConfig, driftFn)
  const pullRef = useRef(bandit.pull)
  const resetRef = useRef(bandit.reset)
  pullRef.current = bandit.pull
  resetRef.current = bandit.reset

  // Notify parent whenever state changes
  useEffect(() => {
    onStateChange(policyId, {
      state: bandit.state,
      pull: () => pullRef.current(),
      reset: () => resetRef.current(),
    })
  }, [bandit.state]) // eslint-disable-line

  return null
}

// ── Main component ────────────────────────────────────────────────────────

export default function PolicyRace({ armConfig, driftFn }) {
  const [selected, setSelected] = useState(() => new Set(RACE_POLICIES.map(p => p.id)))
  const [autoOn, setAutoOn] = useState(false)
  const [speed, setSpeed] = useState(5)
  const intervalRef = useRef(null)

  // Store latest bandit data from each runner
  const runnersRef = useRef({})
  const [tick, setTick] = useState(0) // counter to force re-renders

  const handleStateChange = useCallback((policyId, data) => {
    runnersRef.current[policyId] = data
    setTick(t => t + 1)
  }, [])

  const togglePolicy = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const pullAll = useCallback(() => {
    RACE_POLICIES
      .filter(p => selected.has(p.id))
      .forEach(p => runnersRef.current[p.id]?.pull())
  }, [selected])

  const pullAllRef = useRef(pullAll)
  pullAllRef.current = pullAll

  const stopAuto = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
    setAutoOn(false)
  }, [])

  const startAuto = useCallback(() => {
    intervalRef.current = setInterval(() => pullAllRef.current(), Math.round(1000 / speed))
    setAutoOn(true)
  }, [speed])

  const toggleAuto = () => autoOn ? stopAuto() : startAuto()

  useEffect(() => {
    if (autoOn) { stopAuto(); startAuto() }
  }, [speed]) // eslint-disable-line

  useEffect(() => () => stopAuto(), []) // eslint-disable-line

  const handleReset = () => {
    stopAuto()
    RACE_POLICIES.forEach(p => runnersRef.current[p.id]?.reset())
  }

  // Build chart data from runner states
  const activeRunners = RACE_POLICIES
    .filter(p => selected.has(p.id) && runnersRef.current[p.id])
    .map(p => ({ policy: p, data: runnersRef.current[p.id] }))

  const maxLen = Math.max(0, ...activeRunners.map(r => r.data?.state?.hist?.length || 0))
  const merged = []
  const step = Math.max(1, Math.floor(maxLen / 150))
  for (let i = 0; i < maxLen; i++) {
    if (i % step !== 0 && i !== maxLen - 1) continue
    const point = { t: i + 1 }
    activeRunners.forEach(r => {
      const entry = r.data?.state?.hist?.[i]
      point[r.policy.id] = entry ? entry.cumR : null
    })
    const firstEntry = activeRunners[0]?.data?.state?.hist?.[i]
    point.random = firstEntry ? firstEntry.cumRandom : null
    merged.push(point)
  }

  const pulls = activeRunners[0]?.data?.state?.pulls || 0
  const summary = activeRunners
    .map(r => ({
      ...r.policy,
      pulls: r.data?.state?.pulls || 0,
      cumR: r.data?.state?.cumR || 0,
      cumReg: r.data?.state?.cumReg || 0,
    }))
    .sort((a, b) => b.cumR - a.cumR)

  return (
    <div>
      {/* All runners (headless, always mounted) */}
      {RACE_POLICIES.map(p => (
        <RacePolicyRunner
          key={p.id}
          policyId={p.id}
          armConfig={armConfig}
          driftFn={driftFn}
          onStateChange={handleStateChange}
        />
      ))}

      {/* Policy selector */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
        {RACE_POLICIES.map(p => (
          <button
            key={p.id}
            onClick={() => togglePolicy(p.id)}
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              padding: '3px 8px',
              borderRadius: 'var(--border-radius-md)',
              border: `1px solid ${p.color}`,
              background: selected.has(p.id) ? p.color + '22' : 'transparent',
              color: selected.has(p.id) ? p.color : 'var(--color-text-secondary)',
              cursor: 'pointer',
              opacity: selected.has(p.id) ? 1 : 0.5,
              transition: 'all .12s',
            }}
          >
            {selected.has(p.id) ? '\u2713 ' : ''}{p.label}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
        <button
          onClick={toggleAuto}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: 11, padding: '5px 12px',
            borderRadius: 'var(--border-radius-md)',
            border: '0.5px solid #1D9E75',
            background: autoOn ? '#1D9E75' : 'transparent',
            color: autoOn ? '#fff' : '#1D9E75',
            cursor: 'pointer',
          }}
        >
          {autoOn ? '\u23F8 Running' : '\u25B6 Race'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          <span>Speed</span>
          <input type="range" min="1" max="20" value={speed} step="1"
            onChange={e => setSpeed(Number(e.target.value))} style={{ width: 80 }} />
          <span>{speed}/s</span>
        </div>

        <button onClick={handleReset} style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '5px 12px', borderRadius: 'var(--border-radius-md)', border: '0.5px solid var(--color-border-secondary)', background: 'transparent', cursor: 'pointer', color: 'var(--color-text-secondary)' }}>
          {'\u27F3'} Reset
        </button>

        <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
          {pulls} pulls
        </span>
      </div>

      {/* Overlay chart */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
          cumulative reward — selected policies
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={merged}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
            <XAxis dataKey="t" tick={TICK_STYLE} axisLine={false} tickLine={false} />
            <YAxis tick={TICK_STYLE} axisLine={false} tickLine={false} width={36} />
            <Tooltip
              contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              formatter={v => [v !== null ? v.toFixed(2) : '\u2014']}
            />
            {RACE_POLICIES.filter(p => selected.has(p.id)).map(p => (
              <Line key={p.id} type="monotone" dataKey={p.id} name={p.label} stroke={p.color} dot={false} strokeWidth={1.5} connectNulls isAnimationActive={false} />
            ))}
            <Line type="monotone" dataKey="random" name="Random" stroke="#8b949e" dot={false} strokeWidth={1} strokeDasharray="2 4" opacity={0.6} isAnimationActive={false} />
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
