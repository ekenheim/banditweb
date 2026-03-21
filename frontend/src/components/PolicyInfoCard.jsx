/**
 * components/PolicyInfoCard.jsx
 *
 * Collapsible card showing policy description, real-world use cases,
 * and guidance on when to choose this algorithm.
 */
import { useState } from 'react'
import POLICY_INFO from '../lib/policyInfo'

const CATEGORY_COLORS = {
  Classic: '#378ADD',
  Adversarial: '#E74C3C',
  Contextual: '#F39C12',
}

export default function PolicyInfoCard({ policyId }) {
  const [open, setOpen] = useState(false)
  const info = POLICY_INFO[policyId]
  if (!info) return null

  const badgeColor = CATEGORY_COLORS[info.category] || '#888'

  return (
    <div style={{
      border: '1px solid var(--color-border-secondary)',
      borderRadius: 'var(--border-radius-md)',
      marginBottom: 12,
      background: 'var(--color-background-secondary)',
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          padding: '8px 12px',
          background: 'none',
          border: 'none',
          color: 'var(--color-text-primary)',
          cursor: 'pointer',
          fontFamily: 'var(--font-sans)',
          fontSize: 13,
          textAlign: 'left',
        }}
      >
        <span style={{
          transform: open ? 'rotate(90deg)' : 'rotate(0deg)',
          transition: 'transform 0.15s',
          fontSize: 10,
          color: 'var(--color-text-secondary)',
        }}>
          {'\u25B6'}
        </span>
        <span style={{
          background: badgeColor + '22',
          color: badgeColor,
          padding: '2px 8px',
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 600,
          fontFamily: 'var(--font-mono)',
          textTransform: 'uppercase',
        }}>
          {info.category}
        </span>
        <span style={{ fontWeight: 500 }}>About this policy</span>
      </button>

      {open && (
        <div style={{ padding: '0 12px 12px', fontSize: 13, lineHeight: 1.5, color: 'var(--color-text-secondary)' }}>
          <p style={{ margin: '0 0 10px', color: 'var(--color-text-primary)' }}>
            {info.description}
          </p>

          <div style={{ margin: '0 0 10px' }}>
            <strong style={{ color: 'var(--color-text-primary)', fontSize: 12 }}>Real-world deployments</strong>
            <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
              {info.realWorld.map((rw, i) => (
                <li key={i} style={{ marginBottom: 3 }}>
                  <strong style={{ color: 'var(--color-text-primary)' }}>{rw.company}</strong>
                  {' \u2014 '}{rw.useCase}
                </li>
              ))}
            </ul>
          </div>

          <div style={{ margin: '0 0 10px' }}>
            <strong style={{ color: 'var(--color-text-primary)', fontSize: 12 }}>Best for</strong>
            <p style={{ margin: '4px 0 0' }}>{info.bestFor}</p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 10,
          }}>
            <div>
              <strong style={{ color: '#1D9E75', fontSize: 12 }}>Strengths</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {info.tradeoffs.strengths.map((s, i) => (
                  <li key={i} style={{ marginBottom: 2 }}>{s}</li>
                ))}
              </ul>
            </div>
            <div>
              <strong style={{ color: '#E89320', fontSize: 12 }}>Weaknesses</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {info.tradeoffs.weaknesses.map((w, i) => (
                  <li key={i} style={{ marginBottom: 2 }}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
