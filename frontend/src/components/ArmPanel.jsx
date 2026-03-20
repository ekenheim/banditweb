/**
 * components/ArmPanel.jsx
 * Renders N clickable arm buttons with estimated reward bars.
 */
import React from 'react'

const ARM_LABELS = ['Arm A', 'Arm B', 'Arm C', 'Arm D', 'Arm E']

export default function ArmPanel({ policy, values, counts, lastArm, onPull, color, loading }) {
  const maxVal = Math.max(...values, 0.001)
  const bestArm = values.indexOf(Math.max(...values))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${values.length}, 1fr)`, gap: 8 }}>
      {values.map((val, i) => {
        const isBest = i === bestArm && counts.reduce((a, b) => a + b, 0) > 0
        const isLast = i === lastArm
        return (
          <button
            key={i}
            onClick={() => onPull(i)}
            disabled={loading}
            style={{
              position: 'relative',
              border: `1px solid ${isBest ? color : 'var(--color-border-secondary)'}`,
              borderRadius: 'var(--border-radius-md)',
              padding: '10px 6px 6px',
              background: isBest ? `${color}11` : 'var(--color-background-primary)',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all .12s',
              overflow: 'hidden',
              opacity: loading ? 0.6 : 1,
              outline: isLast ? `2px solid ${color}44` : 'none',
            }}
          >
            <div style={{ fontSize: 10, color: 'var(--color-text-secondary)', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>
              {ARM_LABELS[i]} (n={counts[i]})
            </div>
            <div style={{ fontSize: 16, fontWeight: 500, color: isBest ? color : 'var(--color-text-primary)', fontFamily: 'var(--font-mono)' }}>
              {val.toFixed(2)}
            </div>
            {/* Reward bar */}
            <div style={{
              position: 'absolute', bottom: 0, left: 0,
              height: 3, width: `${(val / maxVal) * 100}%`,
              background: color, opacity: 0.5, transition: 'width .4s',
            }} />
          </button>
        )
      })}
    </div>
  )
}
