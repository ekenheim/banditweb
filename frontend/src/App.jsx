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
import WeightsChart from './components/WeightsChart'
import ConvergencePanel from './components/ConvergencePanel'
import ArmConfigurator from './components/ArmConfigurator'
import PolicyInfoCard from './components/PolicyInfoCard'
import BayesianDeepDive from './components/BayesianDeepDive'
import PolicyRace from './components/PolicyRace'
import { useBandit } from './hooks/useBandit'
import * as sim from './lib/simulation'
import { DRIFT_PATTERNS } from './lib/scenarios'

// ── Constants ─────────────────────────────────────────────────────────────────

const POLICIES = [
  { id: 'epsilon-greedy',    label: '\u03b5-Greedy',      meta: '\u03b5=0.1',          color: '#E89320', category: 'classic' },
  { id: 'ucb',               label: 'UCB1',           meta: 'c=2.0',           color: '#378ADD', category: 'classic' },
  { id: 'thompson-sampling', label: 'Thompson',       meta: 'Beta posterior',  color: '#1D9E75', category: 'classic' },
  { id: 'bayesian-ucb',      label: 'Bayes UCB',      meta: 'credible bound',  color: '#9B59B6', category: 'classic' },
  { id: 'exp3',              label: 'EXP3',           meta: 'adversarial',     color: '#E74C3C', category: 'adversarial' },
  { id: 'linucb',            label: 'LinUCB',         meta: 'contextual UCB',  color: '#D85A30', category: 'contextual' },
  { id: 'lints',             label: 'LinTS',          meta: 'contextual TS',   color: '#F39C12', category: 'contextual' },
]

const CATEGORIES = [
  { id: 'classic',     label: 'CLASSIC' },
  { id: 'adversarial', label: 'ADVERSARIAL' },
  { id: 'contextual',  label: 'CONTEXTUAL' },
]

const SIMULATE = import.meta.env.VITE_SIMULATE === 'true'

// ── Policy panel (one per tab) ────────────────────────────────────────────────

function PolicyPanel({ policy, armConfig, driftFn, labels, onPullsChange }) {
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

  // Report pull count to parent for drift display
  useEffect(() => {
    if (onPullsChange) onPullsChange(state.pulls)
  }, [state.pulls]) // eslint-disable-line

  return (
    <div>
      {/* Policy info card */}
      <PolicyInfoCard policyId={policy.id} />

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

      {/* Posterior distributions or weights chart */}
      <div style={{ marginTop: 16 }}>
        {policy.id === 'exp3' ? (
          <WeightsChart probabilities={sim.getEXP3Probabilities(state)} labels={labels} />
        ) : (
          <PosteriorChart betaParams={betaParams} labels={labels} />
        )}
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

      {/* Bayesian deep dive (PyMC analysis) */}
      <BayesianDeepDive policyId={policy.id} state={state} />
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

  // Track active policy's pull count for drift visualization
  const [activePulls, setActivePulls] = useState(0)
  const pullsRef = useRef({})  // per-policy pull counts
  const handlePullsChange = useCallback((policyId) => (pulls) => {
    pullsRef.current[policyId] = pulls
    if (policyId === activeTab) setActivePulls(pulls)
  }, [activeTab])

  // Update displayed pulls when switching tabs
  useEffect(() => {
    setActivePulls(pullsRef.current[activeTab] || 0)
  }, [activeTab])

  // Effective probabilities after drift
  const effectiveP = driftFn ? driftFn(trueP, activePulls) : trueP
  const effectiveOptimal = Math.max(...effectiveP)

  return (
    <div style={{ fontFamily: 'var(--font-mono)', maxWidth: 1080, margin: '0 auto', padding: '1.5rem 1rem' }}>
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

      {/* True probabilities info bar — shows drifted values when drift is active */}
      <div style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)', padding: '8px 12px', marginBottom: 16, display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: 10, color: 'var(--color-text-secondary)' }}>
          {driftFn ? 'effective p(reward):' : 'true p(reward):'}
        </span>
        {effectiveP.map((p, i) => {
          const label = labels && labels[i] ? labels[i] : `Arm ${String.fromCharCode(65 + i)}`
          const isBest = p === effectiveOptimal
          return (
            <span key={i} style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: isBest ? '#1D9E75' : 'var(--color-text-secondary)', transition: 'color 0.3s' }}>
              {label}: <strong>{p.toFixed(2)}</strong>{isBest ? ' \u2605' : ''}
            </span>
          )
        })}
        {driftFn && (
          <span style={{ fontSize: 9, color: '#D85A30', marginLeft: 'auto' }}>
            step {activePulls}
          </span>
        )}
      </div>

      {/* Sidebar + Content layout */}
      <div style={{ display: 'flex', gap: 20 }}>
        {/* Sidebar navigation */}
        <nav style={{
          width: 170,
          flexShrink: 0,
          position: 'sticky',
          top: 16,
          alignSelf: 'flex-start',
        }}>
          {CATEGORIES.map(cat => {
            const catPolicies = POLICIES.filter(p => p.category === cat.id)
            return (
              <div key={cat.id} style={{ marginBottom: 12 }}>
                <div style={{
                  fontSize: 9,
                  fontWeight: 700,
                  color: 'var(--color-text-secondary)',
                  letterSpacing: '0.08em',
                  padding: '4px 8px',
                  textTransform: 'uppercase',
                }}>
                  {cat.label}
                </div>
                {catPolicies.map(p => (
                  <button
                    key={p.id}
                    onClick={() => setActiveTab(p.id)}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      fontFamily: 'var(--font-sans)',
                      fontSize: 12,
                      fontWeight: activeTab === p.id ? 600 : 400,
                      padding: '6px 8px 6px 16px',
                      border: 'none',
                      background: activeTab === p.id ? p.color + '18' : 'transparent',
                      borderLeft: `2px solid ${activeTab === p.id ? p.color : 'transparent'}`,
                      color: activeTab === p.id ? p.color : 'var(--color-text-secondary)',
                      cursor: 'pointer',
                      borderRadius: '0 var(--border-radius-md) var(--border-radius-md) 0',
                      transition: 'all .12s',
                    }}
                  >
                    {p.label}
                    <span style={{ fontSize: 9, display: 'block', opacity: 0.7, marginTop: 1 }}>{p.meta}</span>
                  </button>
                ))}
              </div>
            )
          })}

          {/* Race link */}
          <div style={{ borderTop: '0.5px solid var(--color-border-tertiary)', paddingTop: 8, marginTop: 4 }}>
            <button
              onClick={() => setActiveTab('race')}
              style={{
                display: 'block',
                width: '100%',
                textAlign: 'left',
                fontFamily: 'var(--font-sans)',
                fontSize: 12,
                fontWeight: activeTab === 'race' ? 600 : 400,
                padding: '6px 8px 6px 16px',
                border: 'none',
                background: activeTab === 'race' ? '#8b949e18' : 'transparent',
                borderLeft: `2px solid ${activeTab === 'race' ? '#8b949e' : 'transparent'}`,
                color: activeTab === 'race' ? '#8b949e' : 'var(--color-text-secondary)',
                cursor: 'pointer',
                borderRadius: '0 var(--border-radius-md) var(--border-radius-md) 0',
                transition: 'all .12s',
              }}
            >
              {'\u26A1'} Race
              <span style={{ fontSize: 9, display: 'block', opacity: 0.7, marginTop: 1 }}>compare all</span>
            </button>
          </div>
        </nav>

        {/* Content area */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Policy panels — all mounted to preserve state, only active is shown */}
          {POLICIES.map(p => (
            <div key={p.id} style={{ display: p.id === activeTab ? 'block' : 'none' }}>
              <PolicyPanel policy={p} armConfig={armConfig} driftFn={driftFn} labels={labels} onPullsChange={handlePullsChange(p.id)} />
            </div>
          ))}

          {/* Race view */}
          <div style={{ display: activeTab === 'race' ? 'block' : 'none' }}>
            <PolicyRace armConfig={armConfig} driftFn={driftFn} />
          </div>
        </div>
      </div>
    </div>
  )
}
