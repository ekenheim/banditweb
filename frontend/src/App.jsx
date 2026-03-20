/**
 * App.jsx — Bandit Demo Dashboard
 *
 * Connects all components. Maintains one bandit state per policy so
 * switching tabs preserves experiment history for the session.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import ArmPanel from './components/ArmPanel'
import Charts from './components/Charts'
import { useBandit } from './hooks/useBandit'
import * as sim from './lib/simulation'

// ── Constants ─────────────────────────────────────────────────────────────────

const POLICIES = [
  { id: 'epsilon-greedy',    label: 'ε-Greedy',   meta: 'ε=0.1 · non-contextual', color: '#E89320' },
  { id: 'ucb',               label: 'UCB1',        meta: 'c=2 · non-contextual',   color: '#378ADD' },
  { id: 'thompson-sampling', label: 'Thompson',    meta: 'Beta posterior',         color: '#1D9E75' },
  { id: 'linucb',            label: 'LinUCB',      meta: 'α=1.0 · contextual',     color: '#D85A30' },
]

const SIMULATE = import.meta.env.VITE_SIMULATE === 'true'

// ── Policy panel (one per tab) ────────────────────────────────────────────────

function PolicyPanel({ policy }) {
  const { state, status, error, pull, reset, estimatedValues, pullCounts, isSimulating } = useBandit(policy.id)
  const [autoOn, setAutoOn] = useState(false)
  const [speed, setSpeed] = useState(5)
  const intervalRef = useRef(null)

  const stopAuto = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
    setAutoOn(false)
  }, [])

  const startAuto = useCallback(() => {
    intervalRef.current = setInterval(() => pull(), Math.round(1000 / speed))
    setAutoOn(true)
  }, [pull, speed])

  const toggleAuto = () => autoOn ? stopAuto() : startAuto()

  // Restart interval when speed changes while running
  useEffect(() => {
    if (autoOn) { stopAuto(); startAuto() }
  }, [speed]) // eslint-disable-line

  useEffect(() => () => stopAuto(), []) // cleanup on unmount

  const handleReset = async () => { stopAuto(); await reset() }

  return (
    <div>
      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
        {[
          { label: 'Total pulls', value: state.pulls },
          { label: 'Cum. reward', value: state.cumR.toFixed(2) },
          { label: 'Cum. regret', value: state.cumReg.toFixed(2) },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)', padding: '8px 12px' }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 20, fontWeight: 500 }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Arm panel */}
      <div style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
        arms — click to pull manually
      </div>
      <div style={{ marginBottom: 12 }}>
        <ArmPanel
          policy={policy.id}
          values={estimatedValues}
          counts={pullCounts}
          lastArm={state.lastArm}
          onPull={arm => pull(arm)}
          color={policy.color}
          loading={status === 'loading'}
        />
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
        <button
          onClick={toggleAuto}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: 11, padding: '5px 12px',
            borderRadius: 'var(--border-radius-md)',
            border: `0.5px solid ${policy.color}`,
            background: autoOn ? policy.color : 'transparent',
            color: autoOn ? '#fff' : policy.color,
            cursor: 'pointer',
          }}
        >
          {autoOn ? '⏸ Running' : '▶ Auto-run'}
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

        {isSimulating && (
          <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
            ● simulation mode
          </span>
        )}
        {error && (
          <span style={{ fontSize: 10, color: 'var(--color-text-danger)', fontFamily: 'var(--font-mono)' }}>
            ⚠ {error}
          </span>
        )}
      </div>

      {/* Charts */}
      <Charts state={state} values={estimatedValues} counts={pullCounts} color={policy.color} />
    </div>
  )
}

// ── Root App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [activePolicy, setActivePolicy] = useState(POLICIES[0].id)
  const active = POLICIES.find(p => p.id === activePolicy)

  return (
    <div style={{ fontFamily: 'var(--font-mono)', maxWidth: 860, margin: '0 auto', padding: '1.5rem 1rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 600, fontFamily: 'var(--font-sans)', marginBottom: 2 }}>
            Bandit Policy Demo
          </h1>
          <p style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
            Seldon Core v2 · MLflow · Kubernetes · {SIMULATE ? 'offline simulation' : 'live cluster'}
          </p>
        </div>
        <a
          href={import.meta.env.VITE_MLFLOW_URL || '/mlflow'}
          target="_blank"
          rel="noreferrer"
          style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', textDecoration: 'none', border: '0.5px solid var(--color-border-secondary)', padding: '4px 10px', borderRadius: 'var(--border-radius-md)' }}
        >
          MLflow UI ↗
        </a>
      </div>

      {/* True probabilities info bar */}
      <div style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)', padding: '8px 12px', marginBottom: 16, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 10, color: 'var(--color-text-secondary)' }}>true Bernoulli p(reward):</span>
        {sim.TRUE_P.map((p, i) => (
          <span key={i} style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: p === sim.OPTIMAL ? '#1D9E75' : 'var(--color-text-secondary)' }}>
            Arm {String.fromCharCode(65 + i)}: <strong>{p}</strong>{p === sim.OPTIMAL ? ' ★' : ''}
          </span>
        ))}
      </div>

      {/* Policy tabs */}
      <div style={{ borderBottom: '0.5px solid var(--color-border-tertiary)', display: 'flex', marginBottom: 16, gap: 0 }}>
        {POLICIES.map(p => (
          <button
            key={p.id}
            onClick={() => setActivePolicy(p.id)}
            style={{
              fontFamily: 'var(--font-sans)', fontSize: 12, fontWeight: 600,
              padding: '8px 14px', border: 'none', background: 'none', cursor: 'pointer',
              color: activePolicy === p.id ? p.color : 'var(--color-text-secondary)',
              borderBottom: `2px solid ${activePolicy === p.id ? p.color : 'transparent'}`,
              marginBottom: -1, whiteSpace: 'nowrap', transition: 'all .12s',
            }}
          >
            {p.label}
            <span style={{ fontSize: 9, marginLeft: 4, opacity: 0.7 }}>{p.meta}</span>
          </button>
        ))}
      </div>

      {/* Active policy panel — all are mounted to preserve state, only active is shown */}
      {POLICIES.map(p => (
        <div key={p.id} style={{ display: p.id === activePolicy ? 'block' : 'none' }}>
          <PolicyPanel policy={p} />
        </div>
      ))}
    </div>
  )
}
