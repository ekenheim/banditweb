/**
 * components/ArmConfigurator.jsx
 * Collapsible panel for configuring arm count, probabilities, scenario, and drift.
 */
import React, { useState } from 'react'
import { SCENARIOS, DRIFT_PATTERNS } from '../lib/scenarios'

export default function ArmConfigurator({ trueP, setTrueP, labels, setLabels, scenarioId, setScenarioId, driftId, setDriftId }) {
  const [open, setOpen] = useState(false)
  const nArms = trueP.length
  const bestIdx = trueP.indexOf(Math.max(...trueP))

  const handleScenario = (id) => {
    const s = SCENARIOS.find(sc => sc.id === id)
    if (!s) return
    setScenarioId(id)
    setTrueP([...s.trueP])
    setLabels(s.labels ? [...s.labels] : null)
  }

  const handleProbChange = (i, val) => {
    const next = [...trueP]
    next[i] = val
    setTrueP(next)
    if (scenarioId !== 'custom') setScenarioId('custom')
  }

  const addArm = () => {
    if (nArms >= 10) return
    setTrueP([...trueP, 0.50])
    if (labels) setLabels([...labels, `Arm ${String.fromCharCode(65 + nArms)}`])
    if (scenarioId !== 'custom') setScenarioId('custom')
  }

  const removeArm = () => {
    if (nArms <= 2) return
    setTrueP(trueP.slice(0, -1))
    if (labels) setLabels(labels.slice(0, -1))
    if (scenarioId !== 'custom') setScenarioId('custom')
  }

  const activeDrift = DRIFT_PATTERNS.find(d => d.id === driftId)

  return (
    <div style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)', marginBottom: 12, overflow: 'hidden' }}>
      {/* Header toggle */}
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '8px 12px', border: 'none', background: 'none', cursor: 'pointer',
          fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-text-secondary)',
        }}
      >
        <span>
          configure arms
          {!open && (
            <span style={{ marginLeft: 8, opacity: 0.6 }}>
              {nArms} arms · {SCENARIOS.find(s => s.id === scenarioId)?.name || 'Custom'}
              {driftId !== 'none' && ` · ${activeDrift?.name}`}
            </span>
          )}
        </span>
        <span style={{ fontSize: 9 }}>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div style={{ padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {/* Scenario selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', minWidth: 52 }}>scenario</span>
            {SCENARIOS.map(s => (
              <button
                key={s.id}
                onClick={() => handleScenario(s.id)}
                title={s.description}
                style={{
                  fontSize: 10, fontFamily: 'var(--font-mono)', padding: '3px 8px',
                  borderRadius: 'var(--border-radius-md)',
                  border: `0.5px solid ${scenarioId === s.id ? '#1D9E75' : 'var(--color-border-secondary)'}`,
                  background: scenarioId === s.id ? '#1D9E7518' : 'transparent',
                  color: scenarioId === s.id ? '#1D9E75' : 'var(--color-text-secondary)',
                  cursor: 'pointer',
                }}
              >
                {s.name}
              </button>
            ))}
          </div>

          {/* Arm count */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', minWidth: 52 }}>arms</span>
            <button onClick={removeArm} disabled={nArms <= 2} style={btnSmall}>−</button>
            <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', minWidth: 20, textAlign: 'center' }}>{nArms}</span>
            <button onClick={addArm} disabled={nArms >= 10} style={btnSmall}>+</button>
          </div>

          {/* Probability sliders */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {trueP.map((p, i) => {
              const label = labels && labels[i] ? labels[i] : `Arm ${String.fromCharCode(65 + i)}`
              const isBest = i === bestIdx
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    fontSize: 10, fontFamily: 'var(--font-mono)', minWidth: 80,
                    color: isBest ? '#1D9E75' : 'var(--color-text-secondary)',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {label}{isBest ? ' ★' : ''}
                  </span>
                  <input
                    type="range" min="0.01" max="0.99" step="0.01" value={p}
                    onChange={e => handleProbChange(i, parseFloat(e.target.value))}
                    style={{ flex: 1, height: 4 }}
                  />
                  <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', minWidth: 30, textAlign: 'right', color: isBest ? '#1D9E75' : 'var(--color-text-primary)' }}>
                    {p.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Drift pattern */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', minWidth: 52 }}>drift</span>
            {DRIFT_PATTERNS.map(d => (
              <button
                key={d.id}
                onClick={() => setDriftId(d.id)}
                title={d.description}
                style={{
                  fontSize: 10, fontFamily: 'var(--font-mono)', padding: '3px 8px',
                  borderRadius: 'var(--border-radius-md)',
                  border: `0.5px solid ${driftId === d.id ? '#D85A30' : 'var(--color-border-secondary)'}`,
                  background: driftId === d.id ? '#D85A3018' : 'transparent',
                  color: driftId === d.id ? '#D85A30' : 'var(--color-text-secondary)',
                  cursor: 'pointer',
                }}
              >
                {d.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

const btnSmall = {
  fontSize: 12, fontFamily: 'var(--font-mono)', padding: '2px 8px',
  borderRadius: 'var(--border-radius-md)',
  border: '0.5px solid var(--color-border-secondary)',
  background: 'transparent', cursor: 'pointer',
  color: 'var(--color-text-primary)',
}
