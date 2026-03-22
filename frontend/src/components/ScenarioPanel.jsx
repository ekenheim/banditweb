/**
 * components/ScenarioPanel.jsx
 * Wrapper panel for an ecommerce scenario. Provides:
 *   - Scenario description
 *   - Policy switcher (defaults to scenario's recommended policy)
 *   - Standard bandit controls (auto-run, speed, reset)
 *   - Standard charts + scenario-specific visualization
 */
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import ArmPanel from './ArmPanel'
import Charts from './Charts'
import PosteriorChart from './PosteriorChart'
import UserTypeHeatmap from './UserTypeHeatmap'
import RevenueChart from './RevenueChart'
import CartBreakdownChart from './CartBreakdownChart'
import { useBandit } from '../hooks/useBandit'
import * as sim from '../lib/simulation'

export default function ScenarioPanel({ scenario }) {
  const { id, description, defaultPolicy, policies, labels, trueP, contextFn, rewardFn, rewardLabel } = scenario

  const [activePolicyId, setActivePolicyId] = useState(defaultPolicy)
  const activePolicy = policies.find(p => p.id === activePolicyId) || policies[0]

  const armConfig = useMemo(() => sim.makeArmConfig(trueP), [trueP])

  const {
    state, status, error, pull, reset, lastMeta,
    estimatedValues, pullCounts, betaParams, isSimulating,
  } = useBandit(
    activePolicyId,
    armConfig,
    null,  // no drift for scenarios
    contextFn || null,
    rewardFn || null,
  )

  const [autoOn, setAutoOn] = useState(false)
  const [speed, setSpeed] = useState(5)
  const intervalRef = useRef(null)
  const pullRef = useRef(pull)
  pullRef.current = pull

  const stopAuto = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
    setAutoOn(false)
  }, [])

  const startAuto = useCallback(() => {
    intervalRef.current = setInterval(() => pullRef.current(), Math.round(1000 / speed))
    setAutoOn(true)
  }, [speed])

  const toggleAuto = () => autoOn ? stopAuto() : startAuto()

  useEffect(() => {
    if (autoOn) { stopAuto(); startAuto() }
  }, [speed]) // eslint-disable-line

  useEffect(() => () => stopAuto(), []) // cleanup on unmount

  // Stop auto-run and reset when switching policies
  const handlePolicySwitch = (policyId) => {
    stopAuto()
    setActivePolicyId(policyId)
  }

  const handleReset = async () => { stopAuto(); await reset() }

  return (
    <div>
      {/* Scenario description */}
      <div style={{
        background: 'var(--color-background-secondary)',
        borderRadius: 'var(--border-radius-md)',
        padding: '10px 14px',
        marginBottom: 12,
        fontSize: 12,
        color: 'var(--color-text-secondary)',
        lineHeight: 1.5,
      }}>
        {description}
      </div>

      {/* Policy switcher */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>policy:</span>
        {policies.map(p => (
          <button
            key={p.id}
            onClick={() => handlePolicySwitch(p.id)}
            style={{
              fontSize: 11, fontFamily: 'var(--font-mono)', padding: '4px 10px',
              borderRadius: 'var(--border-radius-md)',
              border: `0.5px solid ${activePolicyId === p.id ? p.color : 'var(--color-border-secondary)'}`,
              background: activePolicyId === p.id ? p.color + '18' : 'transparent',
              color: activePolicyId === p.id ? p.color : 'var(--color-text-secondary)',
              cursor: 'pointer',
            }}
          >
            {p.label}
            <span style={{ fontSize: 9, marginLeft: 4, opacity: 0.7 }}>{p.meta}</span>
          </button>
        ))}
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
        {[
          { label: 'Total pulls', value: state.pulls },
          { label: rewardLabel || 'Cum. reward', value: state.cumR.toFixed(2) },
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
          policy={activePolicyId}
          values={estimatedValues}
          counts={pullCounts}
          lastArm={state.lastArm}
          onPull={arm => pull(arm)}
          color={activePolicy.color}
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
            border: `0.5px solid ${activePolicy.color}`,
            background: autoOn ? activePolicy.color : 'transparent',
            color: autoOn ? '#fff' : activePolicy.color,
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

        <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
          {'\u25cf'} simulation mode
        </span>
        {error && (
          <span style={{ fontSize: 10, color: 'var(--color-text-danger)', fontFamily: 'var(--font-mono)' }}>
            {'\u26a0'} {error}
          </span>
        )}
      </div>

      {/* Standard charts */}
      <Charts state={state} values={estimatedValues} counts={pullCounts} color={activePolicy.color} labels={labels} />

      {/* Scenario-specific visualizations */}
      {id === 'recommendations' && (
        <UserTypeHeatmap
          scenario={scenario}
          estimatedValues={estimatedValues}
          pullCounts={pullCounts}
          lastMeta={lastMeta}
        />
      )}

      {id === 'pricing' && (
        <RevenueChart
          scenario={scenario}
          estimatedValues={estimatedValues}
          pullCounts={pullCounts}
        />
      )}

      {id === 'checkout-recovery' && (
        <CartBreakdownChart
          scenario={scenario}
          estimatedValues={estimatedValues}
          pullCounts={pullCounts}
          lastMeta={lastMeta}
        />
      )}

      {/* Posterior distributions */}
      <div style={{ marginTop: 16 }}>
        <PosteriorChart betaParams={betaParams} labels={labels} />
      </div>
    </div>
  )
}
