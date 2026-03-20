/**
 * App.jsx — Bandit Demo Dashboard
 *
 * Connects all components. Maintains one bandit state per policy so
 * switching tabs preserves experiment history for the session.
 */
import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import ArmPanel from './components/ArmPanel'
import Charts from './components/Charts'
import PosteriorChart from './components/PosteriorChart'
import ConvergencePanel from './components/ConvergencePanel'
import ArmConfigurator from './components/ArmConfigurator'
import PolicyRace from './components/PolicyRace'
import { useBandit } from './hooks/useBandit'
import * as sim from './lib/simulation'
import { DRIFT_PATTERNS } from './lib/scenarios'

// ── Constants ─────────────────────────────────────────────────────────────────

const POLICIES = [
  { id: 'epsilon-greedy',    label: '\u03b5-Greedy',   meta: '\u03b5=0.1 \u00b7 non-contextual', color: '#E89320' },
  { id: 'ucb',               label: 'UCB1',        meta: 'c=2 \u00b7 non-contextual',   color: '#378ADD' },
  { id: 'thompson-sampling', label: 'Thompson',    meta: 'Beta posterior',         color: '#1D9E75' },
  { id: 'linucb',            label: 'LinUCB',      meta: '\u03b1=1.0 \u00b7 contextual',     color: '#D85A30' },
]

const SIMULATE = import.meta.env.VITE_SIMULATE === 'true'

// ── Policy panel (one per tab) ────────────────────────────────────────────────

function PolicyPanel({ policy, armConfig, driftFn, labels }) {
  const { state, status, error, pull, reset, estimatedValues, pullCounts, betaParams, isSimulating } = useBandit(policy.id, armConfig, driftFn)
  const [autoOn, setAutoOn] = useState(false)
  const [speed, setSpeed] = useState(5)
  const [threshold, setThreshold] = useState(0.95)
  const intervalRef = useRef(null)
  const pullRef = useRef(pull)
  pullRef.current = pull  // always points to the latest pull function

  const stopAuto = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
    setAutoOn(false)
  }, [])

  const startAuto = useCallback(() => {
    intervalRef.current = setInterval(() => pullRef.current(), Math.round(1000 / speed))
    setAutoOn(true)
  }, [speed])

  const toggleAuto = () => autoOn ? stopAuto() : startAuto()

  // Restart interval when speed changes while running
  useEffect(() => {
    if (autoOn) { stopAuto(); startAuto() }
  }, [speed]) // eslint-disable-line

  useEffect(() => () => stopAuto(), []) // cleanup on unmount

  const handleReset = async () => { stopAuto(); await reset() }

  // Compute effective probabilities for display when drift is active
  const effectiveP = driftFn ? driftFn(armConfig.trueP, state.pulls) : armConfig.trueP

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
          labels={labels}
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
          {autoOn ? '\u23f8 Running' : '\u25b6 Auto-run'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          <span>Speed</span>
          <input type="range" min="1" max="20" value={speed} step="1"
            onChange={e => setSpeed(Number(e.target.value))} style={{ width: 80 }} />
          <span>{speed}/s</span>
        </div>

        <button onClick={handleReset} style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '5px 12px', borderRadius: 'var(--border-radius-md)', border: '0.5px solid var(--color-border-secondary)', background: 'transparent', cursor: 'pointer', color: 'var(--color-text-secondary)' }}>
          {'\u27f3'} Reset
        </button>

        {isSimulating && (
          <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
            {'\u25cf'} simulation mode
          </span>
        )}
        {driftFn && (
          <span style={{ fontSize: 10, color: '#D85A30', fontFamily: 'var(--font-mono)' }}>
            drift active
          </span>
        )}
        {error && (
          <span style={{ fontSize: 10, color: 'var(--color-text-danger)', fontFamily: 'var(--font-mono)' }}>
            {'\u26a0'} {error}
          </span>
        )}
      </div>

      {/* Charts */}
      <Charts state={state} values={estimatedValues} counts={pullCounts} color={policy.color} labels={labels} />

      {/* Posterior distributions */}
      <div style={{ marginTop: 16 }}>
        <PosteriorChart betaParams={betaParams} labels={labels} />
      </div>

      {/* Convergence indicator */}
      <div style={{ marginTop: 16 }}>
        <ConvergencePanel
          betaParams={betaParams}
          pulls={state.pulls}
          labels={labels}
          threshold={threshold}
          onThresholdChange={setThreshold}
        />
      </div>
    </div>
  )
}

// ── Root App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState(POLICIES[0].id)
  const [trueP, setTrueP] = useState(sim.DEFAULT_TRUE_P)
  const [labels, setLabels] = useState(null)
  const [scenarioId, setScenarioId] = useState('custom')
  const [driftId, setDriftId] = useState('none')

  const armConfig = useMemo(() => sim.makeArmConfig(trueP), [trueP])
  const driftPattern = DRIFT_PATTERNS.find(d => d.id === driftId)
  const driftFn = driftPattern?.fn || null

  const allTabs = [...POLICIES.map(p => ({ ...p, type: 'policy' })), { id: 'race', label: 'Race', meta: 'all policies', color: '#8b949e', type: 'race' }]

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
          MLflow UI {'\u2197'}
        </a>
      </div>

      {/* Arm configurator */}
      <ArmConfigurator
        trueP={trueP}
        setTrueP={setTrueP}
        labels={labels}
        setLabels={setLabels}
        scenarioId={scenarioId}
        setScenarioId={setScenarioId}
        driftId={driftId}
        setDriftId={setDriftId}
      />

      {/* True probabilities info bar */}
      <div style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)', padding: '8px 12px', marginBottom: 16, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 10, color: 'var(--color-text-secondary)' }}>true p(reward):</span>
        {armConfig.trueP.map((p, i) => {
          const label = labels && labels[i] ? labels[i] : `Arm ${String.fromCharCode(65 + i)}`
          return (
            <span key={i} style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: p === armConfig.optimal ? '#1D9E75' : 'var(--color-text-secondary)' }}>
              {label}: <strong>{p.toFixed(2)}</strong>{p === armConfig.optimal ? ' \u2605' : ''}
            </span>
          )
        })}
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: '0.5px solid var(--color-border-tertiary)', display: 'flex', marginBottom: 16, gap: 0 }}>
        {allTabs.map(p => (
          <button
            key={p.id}
            onClick={() => setActiveTab(p.id)}
            style={{
              fontFamily: 'var(--font-sans)', fontSize: 12, fontWeight: 600,
              padding: '8px 14px', border: 'none', background: 'none', cursor: 'pointer',
              color: activeTab === p.id ? p.color : 'var(--color-text-secondary)',
              borderBottom: `2px solid ${activeTab === p.id ? p.color : 'transparent'}`,
              marginBottom: -1, whiteSpace: 'nowrap', transition: 'all .12s',
            }}
          >
            {p.label}
            <span style={{ fontSize: 9, marginLeft: 4, opacity: 0.7 }}>{p.meta}</span>
          </button>
        ))}
      </div>

      {/* Policy panels — all mounted to preserve state, only active is shown */}
      {POLICIES.map(p => (
        <div key={p.id} style={{ display: p.id === activeTab ? 'block' : 'none' }}>
          <PolicyPanel policy={p} armConfig={armConfig} driftFn={driftFn} labels={labels} />
        </div>
      ))}

      {/* Race view */}
      <div style={{ display: activeTab === 'race' ? 'block' : 'none' }}>
        <PolicyRace armConfig={armConfig} driftFn={driftFn} />
      </div>
    </div>
  )
}
